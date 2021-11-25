import os
import cv2
import json
import pathlib
import onnx
import numpy as np
from furiosa_sdk_quantizer.frontend.onnx import optimize_model, calibrate, quantize
from furiosa_sdk_quantizer.frontend.onnx.quantizer import calibrator, quantizer

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img


def pre_process_vgg(img, dims=None, need_transpose=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_height, output_width, _ = dims
    cv2_interpol = cv2.INTER_AREA
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2_interpol)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    mean=(0.485, 0.456, 0.406)
    variance=(0.229, 0.224, 0.225)
    # normalize image
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)

    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return np.expand_dims(img, axis=0)


model_name = 'videopose'
ext = 'jpg'
dataset = 'imagenet'
preproc = pre_process_vgg
input_size = 550
input_name = 'input'

path = f'{model_name}.onnx'
model = onnx.load_model(path)
optimized_model = optimize_model(model)
print('optimized model')

# pathes = [str(path) for path in pathlib.Path('/Videopose/val2017').glob(f'*.{ext}')]
path = 'VideoPose3D/output.npy'
print(path)
dy_path = f'{model_name}_dynamic_ranges.json'

if not os.path.exists(dy_path):
    dataset = []
    pths = np.load(path)
    # print(pths.shape)
    for pth in pths:
        print(pth.shape)
        dataset.append({input_name: pth.reshape((1,243,17,2)).astype(np.float32)})

    # assert len(dataset) == 500
    dynamic_ranges = calibrate(optimized_model, dataset)
    print(dynamic_ranges)
    with open(dy_path, 'w') as f:
        dynamic_ranges = json.dump(dynamic_ranges, f, indent=4, cls=NumpyEncoder)
else:
    with open(dy_path, 'r') as f:
        dynamic_ranges = json.load(f)

print('i8 model')
quant_model = quantize(optimized_model, True, True, quantizer.QuantizationMode.dfg, dynamic_ranges)
onnx.save_model(quant_model, f'{model_name}_int8.onnx')
print('fake quant model')
quant_model = quantize(optimized_model, True, True, quantizer.QuantizationMode.fake, dynamic_ranges)
onnx.save_model(quant_model, f'{model_name}_fake_quant.onnx')