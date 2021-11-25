import numpy as np
from tensorflow.python.eager.execute import args_to_mixed_eager_tensors
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow.python.ops.variables import trainable_variables
from torch.functional import split

from common_tf.arguments import parse_args
import tensorflow as tf

import os
import sys
import errno

from common_tf.camera import *
from common_tf.model import *
from common_tf.loss import *
from common_tf.generators import ChunkedGenerator, UnchunkedGenerator
import time
from common_tf.utils import deterministic_random
from dataloader import DataLoader

from matplotlib import pyplot as plt
from tqdm import tqdm

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    result = [x.name for x in local_device_protos if x.device_type == 'GPU']
    if len(result) > 0: return result
    return [x.name for x in local_device_protos]

class VideoPose3D(object):
    def __init__(self, args, loader, CHECK_DIR):
        self.checkpoint = CHECK_DIR
        self.architecture = args.architecture
        self.dense = args.dense
        self.stride = args.stride
        self.disable_optimizations = args.disable_optimizations
        self.causal = args.causal
        self.dropout = args.dropout
        self.channels = args.channels
        self.dense = args.dense
        self.batch_size = args.batch_size
        self.linear_projection = args.linear_projection
        self.lr_decay = args.lr_decay
        self.save_every = 1000
        self.args = args
        self.filter_widths = [int(x) for x in self.architecture.split(',')]
        self.bone_length = args.bone_length_term
        
        self.loader = loader

        self.cameras, self.pose3d, self.pose2d = self.loader.prepare(self.args.render, 'valid')

        self.pose_model = TemporalModel(self.pose2d[0].shape[-2], self.pose2d[0].shape[-1], loader.dataset.skeleton().num_joints(),
                                filter_widths = self.filter_widths, causal = self.causal, dropout = self.dropout,
                                channels = self.channels, dense = self.dense)

        self.traj_model = TemporalModel(self.pose2d[0].shape[-2], self.pose2d[0].shape[-1], 1,
                                filter_widths = self.filter_widths, causal = self.causal, dropout = self.dropout,
                                channels = self.channels, dense = self.dense)
    
    def eval_init(self):

        self.input_shape = [None, 17]
        self.input = tf.placeholder( tf.float32, self.input_shape, name='input')  
        self.pred_3d = self.pose_model.forward(self.input)
        self.pred_traj = self.traj_model.forward(self.input)

        self.output = self.pred_traj + self.pred_3d
        #self.seg_map = seg_map #/ 256.0
        self.output = tf.minimum(output, 1e6, name='output')
        print( output.shape )

        if QUANTIZE is True:
            print('Quantization Enabled')
            graph = tf.get_default_graph()
            tf.contrib.quantize.create_eval_graph()
            self.out_graph = 'graph_def.pb'
            out_def = graph.as_graph_def()
            with tf.gfile.GFile(self.out_graph, 'wb') as f:
                f.write(out_def.SerializeToString())

        self.add_saver()

    def train_init(self, cpu_mode =False):

        cameras_train, poses_train, poses_train_2d = self.loader.prepare(self.args.render, 'train_sup')
        cameras_semi, _, poses_semi_2d = self.loader.prepare(self.args.render, 'train_semi')
        
        model_params = 0
        model_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print('INFO: Trainable parameter count:', model_params)

        receptive_field = self.pose_model.receptive_field()
        print('Size of the Receptive Field %d'%receptive_field)
        pad = (receptive_field - 1) // 2 # Padding on each side
        if self.causal:
            print('INFO: Using causal convolutions')
            causal_shift = pad
        else:
            causal_shift = 0

        ########################## data pipeline ####################

        tic = time.time()
        super_generator = ChunkedGenerator(self.batch_size//self.stride, cameras_train, poses_train, poses_train_2d, self.stride,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=self.args.data_augmentation,
                                       kps_left=self.loader.kps_left, kps_right=self.loader.kps_right,\
                                          joints_left=self.loader.joints_left, joints_right=self.loader.joints_right)      
        
        # semi_generator = ChunkedGenerator(self.batch_size//self.stride, cameras_semi, None, poses_semi_2d, self.stride,
        #                                   pad=pad, causal_shift=causal_shift, shuffle=True,
        #                                   random_seed=4321, augment=self.args.data_augmentation,
        #                                   kps_left=self.loader.kps_left, kps_right=self.loader.kps_right,\
        #                                   joints_left=self.loader.joints_left, joints_right=self.loader.joints_right,
        #                                   endless=True)      
        toc = time.time()
        print("###################################")
        print(f'data loading elapsed={toc-tic}sec')

        
        def generate_data():
            _, batch_3d, batch_2d = super_generator.next_epoch()
            #cam, _, batch_2d_semi = None #semi_generator.next_epoch()
            return batch_3d, batch_2d#, cam, batch_2d_semi
        ################################################################


        # batch_3d, batch_2d = tf.py_func(generate_data, [], ('float32', 'float32'), stateful=True)
        batch_3d, batch_2d = generate_data()
        # batch_3d.set_shape(super_generator.batch_3d.shape)
        # batch_2d.set_shape(super_generator.batch_2d.shape)

        if cpu_mode is True:
            gpus = ['/device:CPU:0']
            print(gpus)
        else:
            gpus = get_available_gpus()
            print(gpus)

        num_gpus = len(gpus)
        assert(self.batch_size % num_gpus == 0)
        batch_slice = self.batch_size // num_gpus

        tower_losses = []

        for idx_gpu, gpu in enumerate(gpus):
            print( gpu )
            with tf.device(gpu):
                slice_3d = batch_3d[batch_slice*idx_gpu:batch_slice*(idx_gpu+1),...]
                slice_2d = batch_2d[batch_slice*idx_gpu:batch_slice*(idx_gpu+1),...]
                # slice_cam = cam[batch_slice*idx_gpu:batch_slice*(idx_gpu+1),...]
                # slice_semi = batch_2d_semi[batch_slice*idx_gpu:batch_slice*(idx_gpu+1),...]
                
                # slice_traj = slice_3d[:, :, :1].clone()
                # slice_3d[:, :, 0] = 0
                # split_idx = slice_3d.shape[0]

                # slice_2d_cat =  torch.cat((slice_2d, slice_semi), dim=0) if not skip else slice_2d
                slice_2d_cat = slice_2d

                # slice_2d_cat = self.pose_model.forward(slice_2d_cat)
                pred_3d = self.pose_model.forward( slice_2d_cat)

                # pred_traj = self.traj_model.forward(slice_2d_cat)

                self.pred_3d =pred_3d
                # self.pred_traj = pred_traj

                # if pad > 0:
                #     targ_semi = slice_semi[:,pad:-pad,:,:2]
                # else:
                #     targ_semi = slice_semi[:,:,:,:2]


                with tf.variable_scope('losses'):
                    #mask_slice = tf.expand_dims( mask_slice, axis = -1 )

                    print( pred_3d.shape )
                    # print( pred_traj.shape )

                    # loss_3d = mpjpe(pred_3d[:split_idx], slice_3d)
                    loss_3d = mpjpe(pred_3d, slice_3d)

                    # w = 1 / slice_traj[:,:,:,2]
                    # loss_traj = weighted_mpjpe(pred_traj[:split_idx], slice_traj, w)

                    # assert slice_traj.shape[0]*slice_traj.shape[1] == slice_3d.shape[0]*slice_3d.shape[1]

                    # proj_func = project_to_2d_linear if self.linear_projection else project_to_2d

                    # recn_semi = proj_func(pred_3d)

                    # loss_recn = mpjpe(recn_semi, targ_semi)

                    loss = loss_3d# + loss_traj + loss_recn

                    # if self.bone_length:
                    #     dists = pred_3d[:,:,1:] - pred_3d[:,:, self.loader.dataset.skeleton().parents()[1:]]
                    #     bone_l = tf.reduce_mean(tf.norm(dists, axis=3), axis = 1)
                    #     loss_bone = tf.reduce_mean(tf.abs(tf.reduce_mean(bone_l[:split_idx], axis=0) \
                    #                 - tf.reduce_mean(bone_l[split_idx:], axis=0)))
                    #     loss += loss_bone


                tower_losses.append( loss )

                self.add_summary_per_gpu( idx_gpu = idx_gpu, loss = loss )
        
        # if QUANTIZE is True:
        #     print('Quantization Enabled')
        #     tf.contrib.quantize.create_training_graph(quant_delay = 50000)
        
        self.loss = tf.reduce_mean(tower_losses)
        self.add_gradient()
        
        self.add_summary()
        self.add_saver()


    def add_summary_per_gpu(self, loss, idx_gpu):
        with tf.variable_scope( 'summary_%s'%(idx_gpu) ): 
            tf.summary.scalar("loss", loss)


    def add_summary(self):
        with tf.variable_scope('summary'): 
            tf.summary.scalar("loss", self.loss)
            self.summaries = tf.summary.merge_all()
        pass 
    

    def occasional_jobs( self, sess, global_step ):
        ckpt_filename = os.path.join( self.checkpoint, 'myckpt')

        if global_step % self.save_every == 0:
            save_path = self.saver.save(sess, ckpt_filename, global_step=global_step)     
            tqdm.write( "saved at" + save_path )

    def add_gradient(self ):
        print("add gradient")
        with tf.variable_scope("gradients"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.global_step = tf.Variable( 0, trainable=False)     

            self.learning_rate = tf.train.exponential_decay(
                learning_rate = self.args.learning_rate,
                global_step = self.global_step,
                decay_steps = 1,
                decay_rate = self.lr_decay)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            update_var_list = tf.get_collection( key = tf.GraphKeys.TRAINABLE_VARIABLES )
            #self.print_var_status( update_var_list )

            with tf.control_dependencies(update_ops):
                self.train_op = tf.contrib.training.create_train_op(self.loss,optimizer= optimizer, global_step=self.global_step, update_ops = update_ops )\

    def add_saver(self):
        #var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="ocnet")
        self.saver = tf.train.Saver(max_to_keep=3)

    def train(self, sess):
        self.writer = tf.summary.FileWriter( self.checkpoint, sess.graph)
 
        exp_loss = None
        counter = 0

        print("train starting")        

        print_every = 1000
        self.summary_every = 50
        while True:
            for iter in tqdm( range( print_every ), leave=False ):

                output_feed = { 
                    "train_op": self.train_op,
                    "global_step": self.global_step,
                    "learning_rate": self.learning_rate,
                    "loss": self.loss
                }                

                if iter % self.summary_every == 0:
                    output_feed["summaries"] = self.summaries

                _results = sess.run( output_feed )

                global_step = _results["global_step"]
                learning_rate  = _results["learning_rate"]
                
                if iter % self.summary_every == 0:
                    self.writer.add_summary( _results["summaries"], global_step=global_step )
                        
                cur_loss = _results["loss"]

                if not exp_loss:  # first iter
                    exp_loss = cur_loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * cur_loss                    

                self.occasional_jobs( sess, global_step )                    


            if True: #global_step  % print_every == 0:
                print( f"global_step = {global_step}, learning_rate = {learning_rate:0.6f}")
                print( f"loss = {exp_loss:0.4f}" )

                if global_step == 1000000:
                    break


            """"""
        sys.stdout.flush()
