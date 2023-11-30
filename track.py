import sys
sys.path.insert(0, './yolov5')
from modules.init_output import init_out
from modules.detect_output import detect_out
from modules.send_to_server import send_frame_to_server

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
import cv2
import torch
import torch.backends.cudnn as cudnn
import os
import numpy as np
import glob
import pickle

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Specify the actual width of the car
REAL_CAR_WIDTH = 1.8
REAL_CAR_HEIGHT = 1.4
FOCAL_LENGTH = 650  # Specify the focal length of the camera
EDGE_THRESHOLD = 10  # Threshold value for checking if the bounding box is close to edges

MIN_SPEED = 1
MAX_SPEED = 2
MIN_DISTANCE = 3
MAX_DISTANCE = 15

object_size_data = {}
waiting_queue = {}
speed_data = {}
M = []

# OpenCV RGB color list
RED= (0, 0, 255)
GREEN= (0, 255, 0)
BLUE= (255, 0, 0)
YELLOW= (0, 255, 255)
ORANGE= (0, 165, 255)


output_directory = "outputs"
detected_image_path = 'detected_frames'

def calculate_collision_risk(speed, distance):
    if speed == None:
        return None
    speed = float(speed)
    normalized_speed = (MAX_SPEED - speed) / (MAX_SPEED - MIN_SPEED)
    normalized_distance = (MAX_DISTANCE - distance) / (MAX_DISTANCE - MIN_DISTANCE)
    risk_score = normalized_speed * normalized_distance
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

def undistort(img, cal_dir='camera_cal/cal_pickle.p'):
    #cv2.imwrite('camera_cal/test_cal.jpg', dst)
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def pipeline(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = undistort(img)
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float32)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def perspective_warp(img, 
                     dst_size=(1920,1080),
                     src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     dst=np.float32([(0,0.4), (1, 0.4), (0,1), (1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def inv_perspective_warp(img, 
                     dst_size=(1920,1080),
                     src=np.float32([(0,0.4), (1, 0.4), (0,1), (1,1)]),
                     dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M.clear()
    Mx = cv2.getPerspectiveTransform(src, dst)
    M.append(Mx)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, Mx, dst_size)
    # get the coordinate of the perspective lane
    return warped

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]

def sliding_window(img, nwindows=30, margin=60, minpix = 1, draw_windows=True):
    global left_a, left_b, left_c,right_a, right_b, right_c 
    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    histogram = get_hist(img)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Set height of windows
    window_height = np.int16(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (100,255,255), 3) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (100,255,255), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int16(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int16(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    
    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

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

    vid_cap = cv2.VideoCapture(source)
    frame_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = init_out(output_directory, frame_width, frame_height)

    t0 = time.time()

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
        predictions = non_max_suppression(predictions, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        detection = predictions[0]

        # Process detections
        p, frame = path, frames
        cv2.namedWindow(p, cv2.WINDOW_NORMAL)
        (screen_x, screen_y, windowWidth, windowHeight) = cv2.getWindowImageRect(p)

        img_lane = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_lane = cv2.resize(img_lane, (1920, 1080))
        dst = pipeline(img_lane)
        dst = perspective_warp(dst)
        out_img, curves, lanes, ploty = sliding_window(dst) 

        left_fit, right_fit = curves[0], curves[1]
        ploty = np.linspace(0, img_lane.shape[0]-1, img_lane.shape[0])
        color_img = np.zeros_like(img_lane)
        
        left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
        points = np.hstack((left, right))
     
        inv_perspective = inv_perspective_warp(color_img)
        # Assuming `M` is the perspective transform matrix obtained during the perspective warp
        # `left` and `right` are the coordinates of the polylines in the original perspective image

        # Reshape the coordinates to match the expected input shape for perspective transform
        left = left.reshape((-1, 1, 2))
        right = right.reshape((-1, 1, 2))

        # Apply perspective transform to the coordinates
        left_transformed = cv2.perspectiveTransform(left, M[0])
        right_transformed = cv2.perspectiveTransform(right, M[0])

        # Convert the transformed coordinates to integers
        left_transformed = np.int32(left_transformed)
        right_transformed = np.int32(right_transformed)

        all_transformed = np.concatenate((left_transformed, right_transformed), axis=0)
        # Draw the polylines on the image
        cv2.polylines(inv_perspective, [all_transformed], isClosed=True, color=(255,0,255), thickness=5)
        img_ = inv_perspective
        t2 = time_sync()

        if detection is not None and len(detection):
            car_truck_detections = detection[(detection[:, 5] == 2) | (detection[:, 5] == 7)]  # Assuming "car" class has index 2 and "truck" class has index 7
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
                    #if clss is not car class, continue

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
                    # define an array of three points on image to draw the polylines
                    # shape of point array [3,2]

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
                            cv2.putText(frame, f"{round(collision_risk, 1)}"  , (x1, y2 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 2)
                            cv2.putText(frame, f"WARNING !", (x1, (int)(y1 + (y2-y1)/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 2)
                            cv2.putText(frame, "Medium collision risk detected", (x1, (int)(y1 + (y2-y1)/2 + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 2)
                            dangerColor = ORANGE
                            isWritten = detect_out(detected_image_path, frame, 2)
                            if isWritten:
                                print('Medium collision risk detected and frame is successfully saved')
                        elif collision_risk >= 0.80:
                            cv2.putText(frame, f"{round(collision_risk, 2)}"  , (x1, y2 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
                            cv2.putText(frame, f"WARNING !", (x1, (int)(y1 + (y2-y1)/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
                            cv2.putText(frame, "High collision risk detected", (x1, (int)(y1 + (y2-y1)/2 + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
                            dangerColor = RED
                            isWritten = detect_out(detected_image_path, frame, 1)
                            if isWritten:
                                print('High collision risk detected and frame is successfully saved')
                                # post this frame the /localhost:8080/api/detect
                                send_frame_to_server(frame)

                        else: 
                            cv2.putText(frame, f"{round(collision_risk, 2)}"  , (x1, y2 - 30),
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
        cv2.imshow(p, frame)
        writer.write(frame)
        if cv2.waitKey(1) == ord('q'):  # Q to quit
            exit()
    writer.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', nargs='+', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
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
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
