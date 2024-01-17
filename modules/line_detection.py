import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
# video3
# 0.45,0.51 -  0.55,0.51  -  0.11,1   -   0.95,1


M = []

def undistort(img, matrix_file='./camera_cal/cameraMatrix.pkl', dist_file='./camera_cal/dist.pkl'):
    with open(matrix_file, mode='rb') as f:
        camera_matrix = pickle.load(f)
    
    with open(dist_file, mode='rb') as f:
        distortion_coeff = pickle.load(f)
    
    dst = cv2.undistort(img, camera_matrix, distortion_coeff, None, camera_matrix)
    return dst

def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 255)):
    #img = undistort(img)
    img = np.copy(img)

    #apply gauss blur filter to the image for noise reduction
    img = cv2.GaussianBlur(img, (5, 5), 0)

    imgg = perspective_warp(img)
    imgg = cv2.resize(imgg, (640, 360))
    cv2.imshow('perspective_wrap2', imgg)

    # Convert to HLS color space (Hue, Lightness, Saturation)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float32)

    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]

    l_channel = np.uint8(255*l_channel/np.max(l_channel))
    s_channel = np.uint8(255*s_channel/np.max(s_channel))
    h_channel = np.uint8(255*h_channel/np.max(h_channel))

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # cv2.imshow('perspective_wrap2o', sobelx)
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary

def perspective_warp(img, 
                     dst_size=(1920,1080),
                     src=np.float32([(0.46,0.51),(0.54,0.51),(0.08,1),(1.1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
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
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dst=np.float32([(0.45,0.51),(0.55,0.51),(0.07,1),(1.05,1)])):
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
    #convert binary to image the warped
    return warped

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]

def sliding_window(img, nwindows=30, margin=80, minpix = 1, draw_windows=True):
    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)

    out_img = np.dstack((img, img, img))*255

    histogram = get_hist(img)

    imageeHis = np.dstack((histogram, histogram, histogram)).astype(np.uint8)
    imageeH = cv2.resize(imageeHis, (640, 360))
    cv2.imshow('dsfkjkl', imageeH)

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

def getM():
    return M