#########################################################
#               COMPUTER VISION SUMMATIVE               #
#########################################################
#
#
#########################################################

import os
import cv2
import numpy as np

# Change this to specify absolute path with images
master_path_to_dataset = "TTBB-durham-02-10-17-sub10"
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"

orb = cv2.ORB_create()


#########################################################
#                  KEYPOINT DETECTION                   #
#########################################################

def detect_keypoints(image):
    kp = orb.detect(image, None)
    kp, des = orb.compute(image, kp)

    img2 = cv2.drawKeypoints(image, kp, None, color=(0, 0, 255), flags=0)
    return img2, kp


#########################################################
#               DISPARITY CALCULATIONS                  #
#########################################################

# remember to convert to grayscale (as the disparity matching works on grayscale)
def get_disparity(left_image, right_image):
    disparity = stereoProcessor.compute(left_image, right_image)
    # filter out noise and speckles (adjust parameters as needed)

    dispNoiseFilter = 5  # increase for more aggressive filtering
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 and max_disparity
    # as disparity=-1 means no disparity available
    _, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
    scaled = (disparity / 16.).astype(np.uint8)
    # crop disparity to chop out left part where there are with no disparity
    # as this area is not seen by both cameras and also
    # chop out the bottom area (where we see the front of car bonnet)
    width = np.size(scaled, 1)
    scaled = scaled[0:390, 135:width]
    return scaled


#########################################################
#                     MAIN METHOD                       #
#########################################################

if __name__ == '__main__':
    full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left)
    full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right)

    left_file_list = sorted(os.listdir(full_path_directory_left))

    # uses a modified H. Hirschmuller algorithm [Hirschmuller, 2008] that differs (see opencv manual)
    # parameters can be adjusted, current ones from [Hamilton / Breckon et al. 2013]
    max_disparity = 128
    stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)

    for filename_left in left_file_list:
        # from the left image filename get the corresponding right image

        filename_right = filename_left.replace("_L", "_R")
        full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
        full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

        # Check if valid image, and has corresponding R
        if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)):
            imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
            # imgL_kp = detect_keypoints(imgL)

            imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
            imgR_kp, kp = detect_keypoints(imgR)

            # N.B. need to do for both as both are 3-channel images
            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

            blur = cv2.blur(grayR, (5,5))
            laplacian = cv2.Sobel(blur, cv2.CV_8U, 1, 0)
            # image, contours, hierarchy = cv2.findContours(laplacian, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            # laplacian = cv2.drawContours(laplacian, contours, -1, (255,0,0), 3)

            cv2.imshow('left image', laplacian)
            cv2.imshow('right image', imgR_kp)

            print("-- files loaded successfully\n")

            disparity_scaled = get_disparity(grayL, grayR)
            # display image (scaling it to the full 0->255 range based on the number
            # of disparities in use for the stereo part)
            dsp = (disparity_scaled * (256. / max_disparity)).astype(np.uint8)
            ret, thresh = cv2.threshold(dsp, 20, 255, cv2.THRESH_TOZERO)
            x_offset = grayR.shape[1] - thresh.shape[1]
            # y_offset = grayR.shape[1] - thresh.shape[1]
            cropped = grayR[0:thresh.shape[0], x_offset:x_offset + thresh.shape[1]]
            bitwise_and = cv2.bitwise_and(cropped, thresh)
            cv2.drawKeypoints(bitwise_and, kp, bitwise_and, color=(0, 0, 255), flags=0)
            cv2.imshow("disparity", bitwise_and)
            cv2.waitKey(0)
        else:
            print("-- files skipped (perhaps one is missing or not PNG)\n")

# close all windows

cv2.destroyAllWindows()
