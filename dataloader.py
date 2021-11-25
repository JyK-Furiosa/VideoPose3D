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


class DataLoader:
    def __init__(self, data_dir, key_dir, downsample, train_label, train_unlab, \
                 test, render, test_vis, actions):

        self.dataset_path = 'data/data_3d_' + data_dir + '.npz'

        if data_dir == 'h36m':
            from common_tf.h36m_dataset import Human36mDataset
            self.dataset = Human36mDataset(self.dataset_path)
        elif data_dir.startswith('humaneva'):
            from common_tf.humaneva_dataset import HumanEvaDataset
            self.dataset = HumanEvaDataset(self.dataset_path)
        elif data_dir.startswith('custom'):
            from common_tf.custom_dataset import CustomDataset
            self.dataset = CustomDataset('data/data_2d_' + data_dir + '_' + key_dir + '.npz')
        else:
            raise KeyError('Invalid dataset')


        self.key_data = np.load('data/data_2d_' + data_dir + '_' + key_dir + '.npz', allow_pickle=True)
        self.keypoints_metadata = self.key_data['metadata'].item()
        self.keypoints_symmetry = self.keypoints_metadata['keypoints_symmetry']
        self.kps_left, self.kps_right = list(self.keypoints_symmetry[0]), list(self.keypoints_symmetry[1])
        self.joints_left, self.joints_right = list(self.dataset.skeleton().joints_left()), list(self.dataset.skeleton().joints_right())
        self.keypoints = self.key_data['positions_2d'].item()

        self.downsample = downsample
        self.train_label = train_label
        self.train_unlabel = train_unlab
        self.test = test
        self.test_vis = test_vis
        self.render = render
        self.actions = actions


    def prep_3d(self):
        print('Preparing data...')
        for subject in self.dataset.subjects():

            for action in self.dataset[subject].keys():
                anim = self.dataset[subject][action]
                
                if 'positions' in anim:
                    positions_3d = []
                    for cam in anim['cameras']:
                        pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                        pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                        positions_3d.append(pos_3d)
                    print(subject)
                    anim['positions_3d'] = positions_3d

    def prep_2d(self):
        for subject in self.dataset.subjects():
    
            assert subject in self.keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
            for action in self.dataset[subject].keys():
                assert action in self.keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
                if 'positions_3d' not in self.dataset[subject][action]:
                    continue
                    
                for cam_idx in range(len(self.keypoints[subject][action])):
                    
                    # We check for >= instead of == because some videos in H3.6M contain extra frames
                    mocap_length = self.dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                    assert self.keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                    
                    if self.keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                        # Shorten sequence
                        self.keypoints[subject][action][cam_idx] = self.keypoints[subject][action][cam_idx][:mocap_length]
                    print(subject)
                assert len(self.keypoints[subject][action]) == len(self.dataset[subject][action]['positions_3d'])
                
        for subject in self.keypoints.keys():
        
            for action in self.keypoints[subject]:
                for cam_idx, kps in enumerate(self.keypoints[subject][action]):
                    # Normalize camera frame
                    cam = self.dataset.cameras()[subject][cam_idx]
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                    self.keypoints[subject][action][cam_idx] = kps

    def fetch(self, subjects, action_filter=None, subset=1, parse_3d_poses=True):
        out_poses_3d = []
        out_poses_2d = []
        out_camera_params = []
        for subject in subjects:
            for action in self.keypoints[subject].keys():
                if action_filter is not None:
                    found = False
                    for a in action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue
                    
                poses_2d = self.keypoints[subject][action]
                for i in range(len(poses_2d)): # Iterate across cameras
                    out_poses_2d.append(poses_2d[i])
                    
                if subject in self.dataset.cameras():
                    cams = self.dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for cam in cams:
                        if 'intrinsic' in cam:
                            out_camera_params.append(cam['intrinsic'])
                    
                if parse_3d_poses and 'positions_3d' in self.dataset[subject][action]:
                    poses_3d = self.dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)): # Iterate across cameras
                        out_poses_3d.append(poses_3d[i])
        
        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None
        
        stride = self.downsample
        if subset < 1:
            for i in range(len(out_poses_2d)):
                n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
                start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
                out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
        elif stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]
        
        return out_camera_params, out_poses_3d, out_poses_2d



    def prepare(self, render, options):
        subjects_train = self.train_label.split(',')
        subjects_semi = [] if not self.train_unlabel else self.train_unlabel.split(',')
        if not render:
            subjects_test = self.test.split(',')
        else:
            subjects_test = [self.test_vis]

        semi_supervised = len(subjects_semi) > 0
        if semi_supervised and not self.dataset.supports_semi_supervised():
            raise RuntimeError('Semi-supervised training is not implemented for this dataset')
                    
        

        action_filter = None if self.actions == '*' else self.actions.split(',')
        if action_filter is not None:
            print('Selected actions:', action_filter)
        
        if options == 'valid':
            camera, poses, poses_2d = self.fetch(subjects_test, action_filter)
        elif options == 'train_sup':
            camera, poses, poses_2d = self.fetch(subjects_train, action_filter)
        elif options == 'train_semi':
            camera, poses, poses_2d = self.fetch(subjects_semi, action_filter)
        else:
            raise ValueError('data option is not selected')

        return camera, poses, poses_2d
