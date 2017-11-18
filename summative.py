#########################################################
#               COMPUTER VISION SUMMATIVE               #
#########################################################
#
#
#########################################################

import os
import cv2
import math
import numpy as np
import random

# Change this to specify absolute path with images
master_path_to_dataset = "TTBB-durham-02-10-17-sub10"
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"

orb = cv2.ORB_create()


#########################################################
#                  PRE-PROCESS IMAGES                   #
#########################################################
# NOTE: Apply a mask to the image, so not all points    #
# are processed. Cropping is difficult because the      #
# disparity calculations are based upon the image       #
# dimensions.                                           #
#########################################################

def preprocess(imgL, imgR):
    # N.B. need to do for both as both are 3-channel images
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Blur to improve results and reduce computation
    blur_R = cv2.blur(grayR, (5, 5))
    blur_L = cv2.blur(grayL, (5, 5))
    ret, thresh_R = cv2.threshold(blur_R, 20, 75, cv2.THRESH_TOZERO)
    ret, thresh_L = cv2.threshold(blur_L, 20, 75, cv2.THRESH_TOZERO)

    return thresh_L, thresh_R


#########################################################
#                  KEYPOINT DETECTION                   #
#########################################################

def detect_keypoints(image):
    kp = orb.detect(image, None)
    kp, des = orb.compute(image, kp)

    img2 = cv2.drawKeypoints(image, kp, None, color=(0, 0, 255), flags=0)
    return img2, kp


#########################################################
#                AUTO CANNY FUNCTION                    #
#########################################################
# https://www.pyimagesearch.com/2015/04/06/             #
# zero-parameter-automatic-canny-edge-detection-with    #
# -python-and-opencv/                                   #
#########################################################

def auto_canny(image, sigma=0.99):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


#########################################################
#             PROJECTION TO 3D POINTS                   #
#########################################################

#####################################################################

# fixed camera parameters for this stereo setup (from calibration)

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0
image_centre_w = 474.5

#####################################################################


def project_disparity_to_3d(disparity, rgb=[]):
    points = []
    f = camera_focal_length_px
    B = stereo_camera_baseline_m

    height, width = disparity.shape[:2]

    for y in range(height):  # 0 - height is the y axis index
        for x in range(width):  # 0 - width is the x axis index

            # if we have a valid non-zero disparity
            if disparity[y, x] > 0:

                # calculate corresponding 3D point [X, Y, Z]
                # stereo lecture - slide 22 + 25
                Z = (f * B) / disparity[y, x]

                X = ((x - image_centre_w) * Z) / f
                Y = ((y - image_centre_h) * Z) / f

                points.append([X, Y, Z])

    return points


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
#                PLANE FITTING FUNCTION                 #
#########################################################

def fit_plane(points):
    cross_product_check = np.array([0,0,0])
    while cross_product_check[0] == 0 and cross_product_check[1] == 0 and cross_product_check[2] == 0:
        [P1, P2, P3] = np.array([points[i] for i in sorted(random.sample(range(len(points)), 3))])
        # make sure they are non-collinear
        cross_product_check = np.cross(P1 - P2, P2 - P3)

    # how to - calculate plane coefficients from these points
    coefficients_abc = np.dot(np.linalg.inv(np.array([P1, P2, P3])), np.ones([3,1]))
    coefficient_d = math.sqrt(coefficients_abc[0]*coefficients_abc[0]+coefficients_abc[1]*coefficients_abc[1]+coefficients_abc[2]*coefficients_abc[2])
    return coefficients_abc, coefficient_d


def get_distance_from_plane(points, coefficients_abc, coefficient_d):
    # how to - measure distance of all points from plane given the plane coefficients calculated
    dist = abs((np.dot(points, coefficients_abc) - 1)/coefficient_d)

    return dist


def project_3D_points_to_2D_image_points(points):
    points2 = []

    for i1 in range(len(points)):
        # Keep points within visible range on 2D image
        if points[i1][0] > -0.12 and points[i1][1] > -0.12:
            # reverse earlier projection for X and Y to get x and y again
            x = ((points[i1][0] * camera_focal_length_px) / points[i1][2]) + image_centre_w
            y = ((points[i1][1] * camera_focal_length_px) / points[i1][2]) + image_centre_h
            points2.append([x,y])

    return np.array(points2)


#########################################################
#                  RANSAC CALCULATION                   #
#########################################################

# Returns distances of all points from plane
def ransac_plane(points, max_iterations=30):
    good_model = None
    best_min_distance = 100
    iterations = 0
    while iterations < max_iterations:
        plane_abc, plane_d = fit_plane(points)
        distances = get_distance_from_plane(points, plane_abc, plane_d)
        total_distances = np.sum(distances)

        # Run many iterations and find smallest distance
        if total_distances < best_min_distance:
            best_min_distance = total_distances
            good_model = distances

        iterations += 1

    return good_model


# Optimize with math
# Create new array with flags specifying if element is within threshold
# Element-wise multiplication such that only element within threshold is kept
# Everything else is 0
def find_inliers(points, distances, threshold):
    inliers = np.select([distances <= threshold, distances > threshold], [1, 0])
    return np.multiply(points, inliers)


#########################################################
#                     MAIN METHOD                       #
#########################################################

if __name__ == '__main__':
    mask = cv2.imread('image_assets/_general_mask.png', 0)
    left_mask = cv2.imread('image_assets/left_mask.png', 0)
    right_mask = cv2.imread('image_assets/right_mask.png', 0)

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
            imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

            preprocessed_L, preprocessed_R = preprocess(imgL, imgR)
            # DEBUG
            # cv2.imshow("preprocessed_L", preprocessed_L)
            # cv2.imshow("preprocessed_R", preprocessed_R)

            # Apply mask to only look at region in front of car hood;
            # reduces computation
            preprocessed_L = cv2.bitwise_and(preprocessed_L, left_mask)
            preprocessed_R = cv2.bitwise_and(preprocessed_R, right_mask)
            # DEBUG
            # cv2.imshow("masked_L", preprocessed_L)
            # cv2.imshow("masked_R", preprocessed_R)
            print("-- files loaded successfully\n")

            disparity_scaled = get_disparity(preprocessed_L, preprocessed_R)
            # display image (scaling it to the full 0->255 range based on the number
            # of disparities in use for the stereo part)
            dsp = (disparity_scaled * (256. / max_disparity)).astype(np.uint8)
            # Apply Canny to the disparity image to reduce points in point cloud
            canny_dsp = auto_canny(dsp)
            depth_points = project_disparity_to_3d(canny_dsp)

            # Run RANSAC, keep only inliers
            threshold = 0.1
            best_plane = ransac_plane(depth_points)
            points_below_threshold = find_inliers(depth_points, best_plane, threshold)
            points_to_draw = project_3D_points_to_2D_image_points(points_below_threshold)
            # Convert from floating point array to numpy int array
            # so that can draw on pixels
            points_to_draw = np.array(points_to_draw, np.int32)






            # DEBUG
            # hull = cv2.convexHull(points_to_draw)
            # cv2.polylines(imgL, hull, True, (0,255,0), 3)
            # cv2.imshow("drawn plane", imgL)
            cv2.waitKey(0)
        else:
            print("-- files skipped (perhaps one is missing or not PNG)\n")

# close all windows

cv2.destroyAllWindows()
