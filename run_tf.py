# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from tensorflow.python.ops.variables import trainable_variables

from common_tf.arguments import parse_args
import tensorflow as tf

import os
import sys
import errno

from common_tf.camera import *
from common_tf.model import *
from common_tf.loss import *
from common_tf.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common_tf.utils import deterministic_random
from dataloader import DataLoader
from common_tf import training_help_ftns

from init import VideoPose3D


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


MAIN_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(MAIN_DIR, "experiments")


def main(args):
    args = parse_args()
    CHECK_DIR = os.path.join(OUTPUT_DIR, args.checkpoint)

    try:
        # Create checkpoint directory if it does not exist
        os.makedirs(CHECK_DIR)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', CHECK_DIR)

    print('Loading dataset...')


    loader = DataLoader(args.dataset, args.keypoints, args.downsample, args.subjects_train, args.subjects_unlabeled, \
                    args.subjects_test, args.render, args.viz_subject, args.actions)
    print('Loading 3d dataset...')
    loader.prep_3d()
    print('Loading 2d dataset...')
    loader.prep_2d()

    pose3d = VideoPose3D(args, loader, CHECK_DIR)

    cpu_mode = False
    save_video = True

    if cpu_mode is False:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        #config.gpu_options.per_process_gpu_memory_fraction = 0.7
    else:
        config = tf.ConfigProto(device_count = {'GPU': 0})

    if args.mode == "train":
        pose3d.train_init( )
        with tf.Session(config=config) as sess:
            training_help_ftns.initialize_model(sess, pose3d, CHECK_DIR, init_op=True )
            pose3d.train(sess)

    else:
        raise ValueError('Select model to run the model with')

if __name__ == '__main__':
  tf.app.run()