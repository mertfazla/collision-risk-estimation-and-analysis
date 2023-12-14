import sys
sys.path.insert(0, './yolov5')
from modules.init_output import init_out
from modules.detect_output import detect_out
from modules.send_to_server import send_frame_to_server
from modules.line_detection import pipeline, perspective_warp, sliding_window, inv_perspective_warp, getM

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import argparse
import time
import cv2
import torch
import numpy as np

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Specify the actual width and height of the car
REAL_CAR_WIDTH = 1.8
REAL_CAR_HEIGHT = 1.4

# Specify the focal length of the camera
FOCAL_LENGTH = 650 

# Threshold value for checking if the bounding box is close to edges
EDGE_THRESHOLD = 10 

MIN_SPEED = -3
MAX_SPEED = 0
MIN_DISTANCE = 3
MAX_DISTANCE = 15

object_size_data = {}
waiting_queue = {}
speed_data = {}

# OpenCV RGB color list
RED= (0, 0, 255)
GREEN= (0, 255, 0)
BLUE= (255, 0, 0)
YELLOW= (0, 255, 255)
ORANGE= (0, 165, 255)

DIR_OUTPUT = "outputs"
DIR_DETECTED_IMAGE = 'detected_frames'

global dangerStatus;

def calculate_collision_risk(speed, distance):
    if speed == None:
        return None
    speed = float(speed)
    normalized_speed = (MAX_SPEED - speed) / (MAX_SPEED - MIN_SPEED)
    normalized_distance = (MAX_DISTANCE - distance) / (MAX_DISTANCE - MIN_DISTANCE)
    risk_score = normalized_speed * normalized_distance
    if(risk_score > 1):
        risk_score = 1
    return risk_score

def calculate_distance(width_in_pixels):
    distance = (FOCAL_LENGTH * REAL_CAR_WIDTH) / width_in_pixels
    return distance

def measure_speed(ObjectID, high_value, fps):     
    if object_size_data.get(ObjectID) is not None:
        if(waiting_queue[ObjectID] == 0):
            estimated_speed = round((((high_value-object_size_data[ObjectID])/5))/(1/30),2)
            speed_data[ObjectID] = estimated_speed if speed_data.get(ObjectID) is None else round(((estimated_speed)*60 + speed_data[ObjectID]*40)/100,2)
            #speed_data[ObjectID] = round((high_value-object_size_data[ObjectID])/(1/fps),2)
            object_size_data[ObjectID] = high_value
            waiting_queue[ObjectID] = 0
            return speed_data[ObjectID]
        else:
            waiting_queue[ObjectID] = waiting_queue[ObjectID]-1
            if speed_data.get(ObjectID) is not None:
                return speed_data[ObjectID]
    else:
        object_size_data[ObjectID] = high_value
        waiting_queue[ObjectID] = 0

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
    model = attempt_load(yolo_weights, map_location=device)
    stride = int(model.stride.max())
    image_size = check_img_size(imgsz, s=stride)
   
    dataset = LoadImages(source, img_size=image_size, stride=stride)

    vid_cap = cv2.VideoCapture(source)
    frame_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = init_out(DIR_OUTPUT, frame_width, frame_height)

    t0 = time.time()
    dangerStatus = 0

    # Each frame in the video
    for i, (path, img, frames, vid_cap) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        frameWidth = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Inference
        predictions = model(img)[0]
        predictions = non_max_suppression(predictions, 0.4, 0.5, classes=None, agnostic=False)
        detection = predictions[0]

        # Process detections
        p, frame = path, frames
        cv2.namedWindow(p, cv2.WINDOW_NORMAL)
        (screen_x, screen_y, windowWidth, windowHeight) = cv2.getWindowImageRect(p)

        img_lane = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_lane = cv2.resize(img_lane, (1920, 1080))

        #apply sobel filter
        dst = pipeline(img_lane)

        #bring the frame to the specified perspective.
        dst = perspective_warp(dst)

        out_img, curves, lanes, ploty = sliding_window(dst);

        left_fit, right_fit = curves[0], curves[1]

        ploty = np.linspace(0, img_lane.shape[0]-1, img_lane.shape[0])
        color_img = np.zeros_like(img_lane)
        
        left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
        points = np.hstack((left, right))

        # left[:, :, 0] = np.clip(left[:, :, 0], 300, 1500)
        # right[:, :, 0] = np.clip(left[:, :, 0], 1200, 1500)

        # limit the matrix the left and right y value min 900 px, if less than 900 px, delete the matrix line
        left = left[left[:, :, 1] > 700]
        right = right[right[:, :, 1] > 700]

        inv_perspective = inv_perspective_warp(color_img)
        # Assuming `M` is the perspective transform matrix obtained during the perspective warp
        # `left` and `right` are the coordinates of the polylines in the original perspective image

        # Reshape the coordinates to match the expected input shape for perspective transform
        left = left.reshape((-1, 1, 2))
        right = right.reshape((-1, 1, 2))

        M = getM()

        # Apply perspective transform to the coordinates
        left_transformed = cv2.perspectiveTransform(left, M[0])
        right_transformed = cv2.perspectiveTransform(right, M[0])

        # Convert the transformed coordinates to integers
        left_transformed = np.int32(left_transformed)
        right_transformed = np.int32(right_transformed)

        # x_min = 100
        x_max = 500

        all_transformed = np.concatenate((left_transformed, right_transformed), axis=0)

        # Draw the polylines on the image
        cv2.polylines(inv_perspective, [all_transformed], isClosed=True, color=(255,0,255), thickness=5)
        img_ = inv_perspective
        t2 = time_sync()

        if detection is not None and len(detection):
            # Assuming "car" class has index 2 and "truck" class has index 7
            car_truck_detections = detection[(detection[:, 5] == 2) | (detection[:, 5] == 7)] 
            clss = car_truck_detections[:, 5]

            # Proceed only if car and truck detections are present
            if len(car_truck_detections) > 0:
                # Rescale boxes from img_size to frame size
                car_truck_detections[:, :4] = scale_coords(img.shape[2:], car_truck_detections[:, :4], frame.shape).round()

                # Pass car and truck detections to deepsort
                xywhs = xyxy2xywh(car_truck_detections[:, 0:4])
                confs = car_truck_detections[:, 4]
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
                    forwardSpeed = (255, 255, 255)
                    carArea = (y2-y1)

                    processing_time = t2 - t1
                    fps = 1 / processing_time
                    
                    if (cv2.pointPolygonTest(all_transformed, (x2, y2), False) >= 0) or (cv2.pointPolygonTest(all_transformed, (x1, y2), False) >= 0):
                        speed_string = measure_speed(ObjectID, distance_vector, fps)
                        collision_risk = calculate_collision_risk(speed_string, distance_vector)
                        dangerColor = YELLOW if distance_vector < 5 else GREEN
                    else: 
                        if object_size_data.get(ObjectID) is not None:
                            object_size_data.pop(ObjectID)   
                    speed_string = speed_string if speed_string != None else ""
                    
                    if speed_string != "":
                        forwardSpeed = GREEN if float(speed_string) >= 0 else RED
                    
                    if collision_risk != None:     
                        if collision_risk >= 0.5 and collision_risk < 0.80:
                            cv2.putText(frame, f"{round((collision_risk*100), 1)}%"  , (x1, y2 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 2)
                            cv2.putText(frame, f"WARNING !", (x1, (int)(y1 + (y2-y1)/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 2)
                            cv2.putText(frame, "Medium collision risk detected", (x1, (int)(y1 + (y2-y1)/2 + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 2)
                            dangerColor = ORANGE
                            if(dangerStatus != 2): dangerStatus = 1
                        elif collision_risk >= 0.80:
                            cv2.putText(frame, f"{round((collision_risk*100), 2)}%"  , (x1, y2 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
                            cv2.putText(frame, f"WARNING !", (x1, (int)(y1 + (y2-y1)/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
                            cv2.putText(frame, "High collision risk detected", (x1, (int)(y1 + (y2-y1)/2 + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
                            dangerColor = RED
                            dangerStatus = 2
                        else: 
                            cv2.putText(frame, f"{round((collision_risk*100), 2)}%"  , (x1, y2 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
       
                    # cv2.polylines(frame, [pts], isClosed=True, color=(255,0,0), thickness = 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), dangerColor, 2)
                    cv2.putText(frame, f"{ObjectID} | {distance_vector:.2f}"  , (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2,)
                    cv2.putText(frame, f"{speed_string}"  , (x1, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, forwardSpeed, 2)
                    
        else:
            deepsort.increment_ages()

        # Stream results
        processing_time = t2 - t1
        fps = 1 / processing_time
        print("FPS:", round(fps, 2))
        frame = cv2.addWeighted(frame, 1, inv_perspective, 0.7, 0)

        if(dangerStatus != 0):
            isWritten = detect_out(DIR_DETECTED_IMAGE, frame, dangerStatus)
            if isWritten:
                dangerMessage = "High collision risk detected" if dangerStatus == 2 else "Medium collision risk detected"
                print(dangerMessage)
                # post this frame the /localhost:8080/api/detect
                #send_frame_to_server(frame)
                dangerStatus = 0

        cv2.imshow(p, frame)
        writer.write(frame)
        if cv2.waitKey(1) == ord('q'):  # Q to quit
            exit()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', nargs='+', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    parser.add_argument('--source', type=str, default='video.mp4', help='source')
    parser.add_argument('--img-size', type=int, default=1080, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
