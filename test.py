import cv2
import numpy as np
import sys
import os
import ast
from vidgear.gears import CamGear
import time
import numpy as np

sys.path.insert(1, os.getcwd()+'/simple-HRNet/models/detectors/yolo')
sys.path.insert(1, os.getcwd()+'/simple-HRNet')

from SimpleHRNet_custom import SimpleHRNet
from main_function import d3_regressor, visualization, get_img_from_fig
##########################################################################################################################

video_format = 'MJPG'
video_framerate = 30
save_video = True
video_writer = None

CUSTOM_BOX_USE = False
VISUAL = True
renew_iter = 10

# filename ='videoplayback.mp4'
#filename = 'messi.mp4'
filename = 'suarez.mp4'
#filename = 'video000.mp4'
# filename = '/home/j/Desktop/furi/e/e.MP4'
# filename = 'input_video.mp4'
#filename = None
start_xy = [0,0]
# start_xy = [690,241]
end_xy = [-1,-1]


yolo_model_def="simple-HRNet/models/detectors/yolo/config/yolov3.cfg"
yolo_class_path="simple-HRNet/models/detectors/yolo/data/coco.names"
yolo_weights_path="simple-HRNet/models/detectors/yolo/weights/yolov3.weights"

##########################################################################################################################
def input_crop(image, start_xy = [0,0], end_xy = [-1,-1]):
    return image[start_xy[1]:end_xy[1],start_xy[0]:end_xy[0]]


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if filename is not None:
    rotation_code = None
    video = cv2.VideoCapture(filename)
    nof_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    assert video.isOpened()

else:
    rotation_code = None
    video = CamGear(0).start()
    nof_frames = 1000




model = SimpleHRNet(48, 17, "./simple-HRNet/weights/officialhrnet_w48_384x288.pth",
                    model_name='HRNet',
                    resolution=ast.literal_eval('(384, 288)'),
                    multiperson=True,
                    return_bounding_boxes=True,
                    max_batch_size=16,
                    yolo_model_def=yolo_model_def,
                    yolo_class_path=yolo_class_path,
                    yolo_weights_path=yolo_weights_path,
                    device=device)

facebook = d3_regressor(MODEL_NAME= 'pretrained_model')
vis = visualization(facebook.dataset, facebook.keypoints_metadata)

for i in range ((int)(nof_frames)):
    valid_d = {}
    if filename is not None:
        ret, input_frame = video.read()
        if not ret:
            continue
        if rotation_code is not None:
            input_frame = cv2.rotate(input_frame, rotation_code)
    else:
        input_frame = video.read()
        if input_frame is None:
            break

    input_frame = input_crop(input_frame,start_xy,end_xy)
    IMAGE_SHAPE = [input_frame.shape[0], input_frame.shape[1]]
    # print(np.amax(input_frame))
    if i == 0:
        facebook.dataset._cameras['res_w'] = input_frame.shape[1]
        facebook.dataset._cameras['res_h'] = input_frame.shape[0]
    # input_frame = cv2.resize(input_frame, dsize=(960, 540), interpolation=cv2.INTER_CUBIC) ##########################

    model.predict_custom(input_frame)

    if len(model.pts) == 0:
        model.renew = True
        model.person_ids = np.array((), dtype=np.int32)
        prediction = []


    else:
        for ids in sorted(model.person_ids):
            valid_d[ids] = model.d[ids]

        if len(model.pts) != 0:
            prediction = facebook.predict(np.array(list(valid_d.values())))
        # print(prediction.shape)img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            prediction = []




    if vis.prev_ids != sorted(valid_d.keys()):
        vis.init_canvas(valid_d)

    if VISUAL:
        vis.update_video(input_frame, model.pts, prediction, model.bbox, model.person_ids)
        vis.fig.canvas.flush_events()

        if save_video:
            plot_img = get_img_from_fig(vis.fig)
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*video_format)  # video format
                video_writer = cv2.VideoWriter('football_output.avi', fourcc, video_framerate, (plot_img.shape[1], plot_img.shape[0]))
            video_writer.write(plot_img)

    if CUSTOM_BOX_USE:
        model.renew = False

        if (i+1) % renew_iter == 0:
            model.renew = True

cv2.destroyAllWindows()

if save_video:
    video_writer.release()


