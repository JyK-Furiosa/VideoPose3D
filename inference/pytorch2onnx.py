# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Perform inference on a single video or all videos with a certain extension
(e.g., .mp4) in a folder.
"""

import os
import sys
import ast
import cv2
import time
import torch
import numpy as np
import io
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

sys.path.insert(1, os.getcwd()+'/VideoPose3D')

import matplotlib.pyplot as plt
from common.arguments import parse_args
from common.camera import *
from common.model import *
from common.jottue_dataset import CustomDataset
# from common.custom_dataset import CustomDataset

import torch
import os
import time
import glob
import cv2

ROOT_dir = os.path.dirname(os.path.abspath(__file__))

def get_img_from_fig(fig, dpi=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img


class d3_regressor(nn.Module):
    def __init__(self, MODEL_NAME='pretrained_model', TRAJ=True,  device=torch.device('cpu')):
        super(d3_regressor, self).__init__()
        self.MODEL_NAME = MODEL_NAME
        self.dataset = CustomDataset()
        self.TRAJ = TRAJ
        self.cam = self.dataset.cameras()
        self.architecture = '3,3,3,3,3'
        self.causal = True

        self.filter_widths = [int(x) for x in self.architecture.split(',')]

        print('Loading 2D detections...')
        mm = np.load('VideoPose3D/data/metadata.npy', allow_pickle=True)
        self.keypoints_metadata = mm.item()


        self.model_pos = TemporalModel(17, 2, 
                                self.dataset.skeleton().num_joints(),
                                filter_widths= self.filter_widths, 
                                causal= self.causal, 
                                dropout=0.25, 
                                channels=1024,
                                dense=False)

        if self.TRAJ:
            self.model_traj = TemporalModel(17,2, 1,
                                    filter_widths=self.filter_widths, 
                                    causal=self.causal, 
                                    dropout=0.25, 
                                    channels=1024,
                                    dense=False)
        else:
            self.model_traj = None

        
        self.load_model()


        self.model_pos.to(device)
        if self.TRAJ:
            self.model_traj.to(device)




    def load_model(self):
        root = './VideoPose3D/checkpoint/{}'.format(self.MODEL_NAME)

        chk_filename = sorted(glob.glob(os.path.join(root,'epoch_*.bin')))[-1]
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(checkpoint['epoch']))

        self.model_pos.load_state_dict(checkpoint['model_pos'])
        if self.TRAJ:
            self.model_traj.load_state_dict(checkpoint['model_traj'])


        receptive_field = self.model_pos.receptive_field()
        print('INFO: Receptive field: {} frames'.format(receptive_field))

        if self.causal:
            print('INFO: Using causal convolutions')



    def forward(self, keypoints):
        # keypoint_ = normalize_screen_coordinates(keypoints, w=self.cam['res_w'], h=self.cam['res_h'])

        with torch.no_grad():
            self.model_pos.eval()
            if self.TRAJ:
                self.model_traj.eval()

            # Positional model
            predicted_3d_pos = self.model_pos(keypoints[:, :244, :, :])

            if self.TRAJ:
                predicted_3d_traj = self.model_traj(keypoints[:, :244, :, :])
            else:
                predicted_3d_traj = 0

            predicted = predicted_3d_pos + predicted_3d_traj

            prediction = predicted.squeeze(1)




            # rot = self.dataset.cameras()['orientation']
            # prediction = camera_to_world(prediction, R=rot, t=0)


        return prediction



def main():

    predictor = d3_regressor()
    

    import torchsummary
    img_width=288
    img_height= 384
    batch_size=1

    # x = torch.randn(batch_size, 3, img_height, img_width, requires_grad=True)
    x = torch.randn(batch_size, 243, 17, 2, requires_grad=True)
    # torchsummary.summary(hrnet, input_size=(3, 384, 288))
    torch_out =predictor(x)
    print(torch_out)
    # Export the model
    torch.onnx.export(predictor,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "videopose.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=12,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    #   dynamic_axes={'input' : {0 : 'batch_size', 2:'img_height', 3:'img_width'},    # variable length axes
                    #                 'output' : {0 : 'batch_size',1:'height', 2:'width'}}
                    )
    output_path = 'videopose.onnx'
    import numpy as np
    import onnxruntime
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(output_path)
    # ort_session = onnxruntime.InferenceSession("craft__fake_quant.onnx")
    # ort_session = onnxruntime.InferenceSession("craft__int8.onnx")
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(np.amax(ort_outs[0]), np.amin(ort_outs[0]))
    print(np.amax(to_numpy(torch_out)), np.amin(to_numpy(torch_out)))
    print(ort_outs[0].shape)

    print(to_numpy(torch_out).shape)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    main()
