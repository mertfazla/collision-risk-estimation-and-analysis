import sys
sys.path.insert(0, './yolov5')
from modules.init_output import init_out

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import os

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# Specify the actual width of the car
REAL_CAR_WIDTH = 1.8
REAL_CAR_HEIGHT = 1.4
FOCAL_LENGTH = 650  # Specify the focal length of the camera
EDGE_THRESHOLD = 10  # Threshold value for checking if the bounding box is close to edges

MIN_SPEED = -20
MAX_SPEED = 0
MIN_DISTANCE = 3
MAX_DISTANCE = 30

object_size_data = {}
waiting_queue = {}
speed_data = {}

output_directory = "outputs"

def calculate_collision_risk(speed, distance):
    if speed == None:
        return None
    speed = int(speed)
    normalized_speed = (MAX_SPEED - speed) / (MAX_SPEED - MIN_SPEED)
    normalized_distance = (MAX_DISTANCE - distance) / (MAX_DISTANCE - MIN_DISTANCE)
    risk_score = normalized_speed * normalized_distance
    return risk_score

def calculate_distance(width_in_pixels):
    distance = (FOCAL_LENGTH * REAL_CAR_WIDTH) / width_in_pixels
    return distance

def measure_speed(ObjectID, high_value, l):     
    if object_size_data.get(ObjectID) is not None:
        if(waiting_queue[ObjectID] == 0):
            #print(high_value, "-> ", object_size_data[ObjectID], "Speed-> ",object_size_data[ObjectID]-high_value)
            speed_data[ObjectID] = object_size_data[ObjectID]-high_value
            object_size_data[ObjectID] = high_value
            waiting_queue[ObjectID] = 6
            return speed_data[ObjectID]
        else:
            waiting_queue[ObjectID] = waiting_queue[ObjectID]-1
            if speed_data.get(ObjectID) is not None:
                return speed_data[ObjectID]
    else:
        object_size_data[ObjectID] = high_value
        waiting_queue[ObjectID] = 6


  

def detect(opt):
    source, yolo_weights, imgsz = opt.source, opt.yolo_weights, opt.img_size

    # Initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    #attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, 
                        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, 
                        n_init=cfg.DEEPSORT.N_INIT, 
                        nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    
    print("CUDA availability: ",torch.cuda.is_available())
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())
    image_size = check_img_size(imgsz, s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    if half:
        model.half()  # to FP16
   
    dataset = LoadImages(source, img_size=image_size, stride=stride)
    t0 = time.time()

    vid_cap = cv2.VideoCapture(source)
    frame_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = init_out(output_directory, frame_width, frame_height)

    # Each frame in the video
    for i, (path, img, frames, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        frameWidth = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        rectangle1_x1 = (int)(frameWidth/2 - (int)(frameWidth/100)*5)
        rectangle1_y1 = frameHeight - (int)(frameHeight/100)*41
        rectangle1_x2 = (int)(frameWidth/2 + (int)(frameWidth/100)*5)
        rectangle1_y2 = frameHeight - (int)(frameHeight/100)*37

        rectangle2_x1 = (int)(frameWidth/2 - (int)(frameWidth/100)*8)
        rectangle2_y1 = frameHeight - (int)(frameHeight/100)*36
        rectangle2_x2 = (int)(frameWidth/2 + (int)(frameWidth/100)*8)
        rectangle2_y2 = frameHeight - (int)(frameHeight/100)*29

        rectangle3_x1 = (int)(frameWidth/2 - (int)(frameWidth/100)*12)
        rectangle3_y1 = frameHeight - (int)(frameHeight/100)*28
        rectangle3_x2 = (int)(frameWidth/2 + (int)(frameWidth/100)*12)
        rectangle3_y2 = frameHeight - (int)(frameHeight/100)*21

        rectangle4_x1 = (int)(frameWidth/2 - (int)(frameWidth/100)*17)
        rectangle4_y1 = frameHeight - (int)(frameHeight/100)*20
        rectangle4_x2 = (int)(frameWidth/2 + (int)(frameWidth/100)*17)
        rectangle4_y2 = frameHeight - (int)(frameHeight/100)*10


        # Inference
        t1 = time_sync()
        predictions = model(img)[0]
        predictions = non_max_suppression(predictions, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        detection = predictions[0]
        t2 = time_sync()

        # Process detections
        p, frame = path, frames
        cv2.namedWindow(p, cv2.WINDOW_NORMAL)
        (screen_x, screen_y, windowWidth, windowHeight) = cv2.getWindowImageRect(p)
    
        if detection is not None and len(detection):
            # Rescale boxes from img_size to frame size
            detection[:, :4] = scale_coords(img.shape[2:], detection[:, :4], frame.shape).round()

            # Pass detections to deepsort
            xywhs = xyxy2xywh(detection[:, 0:4])
            confs = detection[:, 4]
            clss = detection[:, 5]
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
            
            # Draw boxes for visualization
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)): 
                    bboxes = output[0:4]
                    x1, y1, x2, y2 = int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3])
                    frame_height, frame_width = frame.shape[:2]
                    car_aspect_ratio = REAL_CAR_WIDTH / REAL_CAR_HEIGHT
                    box_height = y2 - y1
                    box_width = x2 - x1
                    visible_area_ratio = box_width / (box_height * car_aspect_ratio)
                    estimated_real_width = REAL_CAR_WIDTH * visible_area_ratio
                    car_width_in_pixels = box_width / visible_area_ratio
                    distance_vector = calculate_distance(car_width_in_pixels)
                    
                    
                    # Check if the bounding box is close to the edges of the frame
                    #if x1 <= EDGE_THRESHOLD or y1 <= EDGE_THRESHOLD or x2 >= frame_width - EDGE_THRESHOLD or y2 >= frame_height - EDGE_THRESHOLD:
                    ObjectID = output[4]
                    classString = output[5]
                    speed_string = ""
                    collision_risk = None
                    speed_int = 0
                    dangerColor = (255,255,0)
                    collisionWarningColor = (255,255,0)
                    carArea = (y2-y1)

                    if (rectangle1_x1 <= x1 <= rectangle1_x2 or rectangle1_x1 <= x2 <= rectangle1_x2) and rectangle1_y1 <= y2 <= rectangle1_y2:
                        speed_string = measure_speed(ObjectID, carArea, 1)
                        collision_risk = calculate_collision_risk(speed_string, distance_vector)
                        dangerColor = (124,252,0) if distance_vector < 5 else (124,252,0) 
                    elif (rectangle2_x1 <= x1 <= rectangle2_x2 or rectangle2_x1 <= x2 <= rectangle2_x2) and rectangle1_y2 <= y2 <= rectangle2_y1:
                        speed_string = measure_speed(ObjectID, carArea, 2)
                        collision_risk = calculate_collision_risk(speed_string, distance_vector)
                        dangerColor = (124,252,0) if distance_vector < 5 else (124,252,0) 
                    elif (rectangle2_x1 <= x1 <= rectangle2_x2 or rectangle2_x1 <= x2 <= rectangle2_x2) and rectangle2_y1 <= y2 <= rectangle2_y2:
                        speed_string = measure_speed(ObjectID, carArea, 3)
                        collision_risk = calculate_collision_risk(speed_string, distance_vector)
                        dangerColor = (0,255,0) if distance_vector < 5 else (124,252,0) 
                    elif (rectangle2_x1 <= x1 <= rectangle2_x2 or rectangle2_x1 <= x2 <= rectangle2_x2) and rectangle2_y2 <= y2 <= rectangle3_y1:
                        speed_string = measure_speed(ObjectID, carArea, 4)
                        collision_risk = calculate_collision_risk(speed_string, distance_vector)
                        dangerColor = (0,255,0) if distance_vector < 5 else (124,252,0) 
                    elif (rectangle3_x1 <= x1 <= rectangle3_x2 or rectangle3_x1 <= x2 <= rectangle3_x2) and rectangle3_y1 <= y2 <= rectangle3_y2:
                        speed_string = measure_speed(ObjectID, carArea, 5)
                        collision_risk = calculate_collision_risk(speed_string, distance_vector)
                        dangerColor = (0,255,0) if distance_vector < 5 else (124,252,0) 
                    elif (rectangle2_x1 <= x1 <= rectangle2_x2 or rectangle2_x1 <= x2 <= rectangle2_x2) and rectangle3_y2 <= y2 <= rectangle4_y1:
                        speed_string = measure_speed(ObjectID, carArea, 6)
                        collision_risk = calculate_collision_risk(speed_string, distance_vector)
                        dangerColor = (0,255,0) if distance_vector < 5 else (124,252,0) 
                    elif (rectangle4_x1 <= x1 <= rectangle4_x2 or rectangle4_x1 <= x2 <= rectangle4_x2) and rectangle4_y1 <= y2 <= rectangle4_y2:
                        speed_string = measure_speed(ObjectID, carArea, 7)
                        collision_risk = calculate_collision_risk(speed_string, distance_vector)
                        dangerColor = (0,255,0) if distance_vector < 5 else (124,252,0) 
                    elif (rectangle2_x1 <= x1 <= rectangle2_x2 or rectangle2_x1 <= x2 <= rectangle2_x2) and y2 > rectangle2_y2:
                        speed_string = measure_speed(ObjectID, carArea, 8)
                        collision_risk = calculate_collision_risk(speed_string, distance_vector)
                        dangerColor = (0,255,0) if distance_vector < 9 else (124,252,0) 
                    else: 
                        if object_size_data.get(ObjectID) is not None:
                            object_size_data.pop(ObjectID)
                    speed_string = speed_string if speed_string != None else ""
                    forwardSpeed = (255, 255, 255)
                    if speed_string != "":
                        forwardSpeed = (0, 255, 0) if int(speed_string) >= 0 else (0, 0, 255)
    
                    if collision_risk != None:
                        if collision_risk > 0.80:
                            collisionWarningColor = (255, 0, 0)
                        elif collision_risk > 0.5:
                            collisionWarningColor = (255, 255, 0)
                        else: collisionWarningColor = (0, 255, 0)
                        cv2.putText(frame, f"{round(collision_risk, 2)}"  , (x1, y2 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, collisionWarningColor, 2)    
                            
                    cv2.rectangle(frame, (rectangle1_x1, rectangle1_y1), (rectangle1_x2, rectangle1_y2), (255, 128, 0), 2)
                    cv2.rectangle(frame, (rectangle2_x1, rectangle2_y1), (rectangle2_x2, rectangle2_y2), (255, 128, 0), 2)
                    cv2.rectangle(frame, (rectangle3_x1, rectangle3_y1), (rectangle3_x2, rectangle3_y2), (255, 128, 0), 2)
                    cv2.rectangle(frame, (rectangle4_x1, rectangle4_y1), (rectangle4_x2, rectangle4_y2), (255, 128, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), dangerColor, 2)
                    cv2.putText(frame, f"{ObjectID} | {distance_vector:.2f}"  , (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2,)
                    cv2.putText(frame, f"{speed_string}"  , (x1, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, forwardSpeed, 2)
                    
        else:
            deepsort.increment_ages()
        #print('%sDone. (%.3fs)' % (s, t2 - t1))

        # Stream results
        cv2.imshow(p, frame)
        writer.write(frame)
        if cv2.waitKey(1) == ord('q'):  # Q to quit
            exit()
    #print('Done. (%.3fs)' % (time.time() - t0))
    writer.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', nargs='+', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    parser.add_argument('--source', type=str, default='video.mp4', help='source')
    parser.add_argument('--img-size', type=int, default=720, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
