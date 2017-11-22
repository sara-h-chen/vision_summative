#########################################################
#               COMPUTER VISION SUMMATIVE               #
#########################################################
#
#
#########################################################

# TODO: Highlight anything that appears not near the front
# TODO: Output the file name and the road surface normal (a, b, c)
# TODO: Find out how to calculate normal
# TODO: Make sure that plane can be plotted; if it is vertical, throw away
# TODO: To wrap-up, allow the script to cycle through the images without keypress
# TODO: Draw glyph with normal
# TODO: Write up report
# TODO: Clean up code

import os
import cv2
import math
import numpy as np
import random

# Change this to specify absolute path with images
master_path_to_dataset = "TTBB-durham-02-10-17-sub10"
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"

#####################################################################

orb = cv2.ORB_create()


#########################################################
#             ILLUMINATION PRE-PROCESSING               #
#########################################################
# https://stackoverflow.com/questions/18452438/how-can- #
# i-remove-drastic-brightness-variations-in-a-video     #
#########################################################

def remove_illumination(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(image)
    y = cv2.equalizeHist(y)
    # Remove low frequency details of the image
    blur_y = cv2.GaussianBlur(y, (23,23), 0)
    y = y - blur_y
    image = cv2.merge((y, u, v))
    return cv2.cvtColor(image, cv2.COLOR_YUV2BGR)


#########################################################
#                 HISTOGRAM MATCHING                    #
#########################################################
# http://vzaguskin.github.io/histmatching1/             #
#########################################################

def match(imsrc, imdest, nbr_bins=255):
    if len(imsrc.shape) < 3:
        imsrc = imsrc[:, :, np.newaxis]
        imdest = imdest[:, :, np.newaxis]

    imres = imsrc.copy()
    for d in range(imsrc.shape[2]):
        imhist, bins = np.histogram(imsrc[:, :, d].flatten(), nbr_bins, normed=True)
        tinthist, bins = np.histogram(imdest[:, :, d].flatten(), nbr_bins, normed=True)

        cdfsrc = imhist.cumsum()  # cumulative distribution function
        cdfsrc = (255 * cdfsrc / cdfsrc[-1]).astype(np.uint8)  # normalize

        cdftint = tinthist.cumsum()  # cumulative distribution function
        cdftint = (255 * cdftint / cdftint[-1]).astype(np.uint8)  # normalize

        im2 = np.interp(imsrc[:, :, d].flatten(), bins[:-1], cdfsrc)
        im3 = np.interp(im2, cdftint, bins[:-1])
        imres[:, :, d] = im3.reshape((imsrc.shape[0], imsrc.shape[1]))
    return imres


#########################################################
#                   OBJECT DETECTION                    #
#########################################################

def detect_keypoints(image):
    kp = orb.detect(image, None)
    kp, des = orb.compute(image, kp)

    img2 = cv2.drawKeypoints(image, kp, None, color=(0, 0, 255), flags=0)
    return img2, kp


def extract_keypoints(keypoints):
    keypoints_array = [keypoints[i].pt for i in range(len(keypoints))]
    keypoints_int = np.array(keypoints_array, np.float32)
    return keypoints_int


def cluster_keypoints(extracted_kp):
    img = cv2.imread('image_assets/plain_black.png', 0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 10)
    temp, classified_points, centers = cv2.kmeans(extracted_kp, K=10, bestLabels=None,
                                                  criteria=crit, attempts=1,
                                                  flags=cv2.KMEANS_RANDOM_CENTERS)
    for point, allocation in zip(extracted_kp, classified_points):
        color = (255, 255, 255)
        cv2.circle(img, (int(point[0]), int(point[1])), 8, color, -1)

    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.arcLength(contour, False) > 90:
            x, y, w, h = cv2.boundingRect(contour)
            box = np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], dtype=np.int32)
            cv2.fillPoly(img, box, (255, 255, 255))

    return img


#####################################################################

#########################################################
#                  PRE-PROCESS IMAGES                   #
#########################################################
# NOTE: Apply a mask to the image, so not all points    #
# are processed. Cropping is difficult because the      #
# disparity calculations are based upon the image       #
# dimensions.                                           #
#########################################################

def preprocess(imgL, imgR):
    illum_removed_l = remove_illumination(imgL)
    illum_removed_r = remove_illumination(imgR)
    # N.B. need to do for both as both are 3-channel images
    grayL = cv2.cvtColor(illum_removed_l, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(illum_removed_r, cv2.COLOR_BGR2GRAY)

    ret, thresh_R = cv2.threshold(grayR, 20, 100, cv2.THRESH_TOZERO)
    ret, thresh_L = cv2.threshold(grayL, 20, 100, cv2.THRESH_TOZERO)
    # cv2.imshow("threshed_r", thresh_R)
    # cv2.imshow("threshed_l", thresh_L)

    grayR_matched = match(thresh_R, thresh_L)
    grayL_matched = match(thresh_L, thresh_R)
    cv2.imshow("matched_r", grayR_matched)

    return grayL_matched, grayR_matched


#########################################################
#             PROJECTION TO 3D POINTS                   #
#########################################################

# fixed camera parameters for this stereo setup (from calibration)

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0
image_centre_w = 474.5

#########################################################


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
    return scaled


#########################################################
#                PLANE FITTING FUNCTION                 #
#########################################################

def fit_plane(points):
    [p1, p2, p3] = [0, 0, 0]
    cross_product_check = np.array([0, 0, 0])
    while cross_product_check[0] == 0 and cross_product_check[1] == 0 and cross_product_check[2] == 0:
        [p1, p2, p3] = np.array([points[i] for i in sorted(random.sample(range(len(points)), 3))])
        # make sure they are non-collinear
        cross_product_check = np.cross(p1 - p2, p2 - p3)

    # how to - calculate plane coefficients from these points
    coefficients_abc = np.dot(np.linalg.inv(np.array([p1, p2, p3])), np.ones([3, 1]))
    coefficient_d = math.sqrt(coefficients_abc[0]*coefficients_abc[0] +
                              coefficients_abc[1]*coefficients_abc[1] +
                              coefficients_abc[2]*coefficients_abc[2])
    return coefficients_abc, coefficient_d


def get_distance_from_plane(points, coefficients_abc, coefficient_d):
    # how to - measure distance of all points from plane given the plane coefficients calculated
    dist = abs((np.dot(points, coefficients_abc) - 1)/coefficient_d)

    return dist


def project_3d_points_to_2d_image_points(points):
    points2 = []

    for i1 in range(len(points)):
        # reverse earlier projection for X and Y to get x and y again
        if points[i1][2]:
            x = ((points[i1][0] * camera_focal_length_px) / points[i1][2]) + image_centre_w
            y = ((points[i1][1] * camera_focal_length_px) / points[i1][2]) + image_centre_h
            points2.append([x, y])

    return np.array(points2)


#########################################################
#                  RANSAC CALCULATION                   #
#########################################################

# Returns distances of all points from plane
def ransac_plane(points, inlier_thresh, max_iterations=50):
    iterations = 0
    best_abc, best_d = 0, 1
    inlier_points = []
    while iterations < max_iterations:
        temp_inliers = []
        plane_abc, plane_d = fit_plane(points)
        distances = get_distance_from_plane(points, plane_abc, plane_d)
        for i in range(len(distances)):
            if distances[i] < inlier_thresh:
                temp_inliers.append(points[i])

        # Run many iterations and find plane with the most points that agree
        if len(temp_inliers) > len(inlier_points):
            inlier_points = temp_inliers

        iterations += 1
    # TODO: Fix this
    print("Normal: ", np.divide(best_abc, best_d))
    return inlier_points


#########################################################
#                     MAIN METHOD                       #
#########################################################

if __name__ == '__main__':
    mask = cv2.imread('image_assets/lm_dilated.png', 0)
    plain = cv2.imread('image_assets/plain_black.png', 0)

    full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left)
    full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right)

    left_file_list = sorted(os.listdir(full_path_directory_left))

    # uses a modified H. Hirschmuller algorithm [Hirschmuller, 2008] that differs (see opencv manual)
    # parameters can be adjusted, current ones from [Hamilton / Breckon et al. 2013]
    max_disparity = 128
    stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)
    kernel = np.ones((5, 5), np.uint8)

    for filename_left in left_file_list:
        # from the left image filename get the corresponding right image
        filename_right = filename_left.replace("_L", "_R")
        full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
        full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

        # Check if valid image, and has corresponding R
        if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)):
            imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
            imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

            # Remove green regions
            hsv_l =cv2.cvtColor(imgL, cv2.COLOR_BGR2HSV)
            sensitivity = 40
            lower_green = np.array([60 - sensitivity, 50, 50])
            upper_green = np.array([60 + sensitivity, 255, 255])
            green_region = cv2.inRange(hsv_l, lower_green, upper_green)
            ret, thresh_green = cv2.threshold(green_region, 1, 255, cv2.THRESH_BINARY)
            inv_green = cv2.bitwise_not(thresh_green)
            inv_green = cv2.dilate(inv_green, kernel, iterations=3)
            # DEBUG
            # cv2.imshow("green regions", inv_green)

            # Remove red regions
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            red_region = cv2.inRange(hsv_l, lower_red, upper_red)
            ret, thresh_red = cv2.threshold(red_region, 1, 255, cv2.THRESH_BINARY)
            inv_red = cv2.bitwise_not(thresh_red)
            inv_red = cv2.dilate(inv_red, kernel, iterations=3)
            ## DEBUG
            # cv2.imshow("red regions", inv_red)

            preprocessed_L, preprocessed_R = preprocess(imgL, imgR)
            # DEBUG
            # cv2.imshow("preprocessed_L", preprocessed_L)
            # cv2.imshow("preprocessed_R", preprocessed_R)

            # Apply mask to only look at region in front of car hood;
            # reduces computation
            preprocessed_L = cv2.bitwise_and(preprocessed_L, mask)
            # Remove green and red elements
            # preprocessed_L = cv2.bitwise_and(preprocessed_L, inv_green)
            # preprocessed_L = cv2.bitwise_and(preprocessed_L, inv_red)
            cv2.imshow("red_removed", preprocessed_L)
            preprocessed_R = cv2.bitwise_and(preprocessed_R, mask)
            # DEBUG
            # cv2.imshow("masked_L", preprocessed_L)
            # cv2.imshow("masked_R", preprocessed_R)
            print("-- files loaded successfully\n")

            disparity_scaled = get_disparity(preprocessed_L, preprocessed_R)
            # display image (scaling it to the full 0->255 range based on the number
            # of disparities in use for the stereo part)
            dsp = (disparity_scaled * (256. / max_disparity)).astype(np.uint8)
            depth_points = project_disparity_to_3d(dsp)
            # DEBUG
            cv2.imshow("depth", dsp)

            # Run RANSAC, keep only inliers
            threshold = 0.05
            best_plane = ransac_plane(depth_points, threshold)
            points_to_draw = project_3d_points_to_2d_image_points(best_plane)

            dsp_x, dsp_y = dsp.shape[1], dsp.shape[0]
            img_x, img_y = imgL.shape[1], imgL.shape[0]
            for point in points_to_draw:
                cv2.circle(plain, (int(point[0]) + (img_x - dsp_x), int(point[1])), 1, (255,255,255), 1)
            # Draw circles that make up plain
            # DEBUG
            # cv2.imshow("drawn circles", plain)

            # TODO: Put this in function
            # Find objects in scene
            img, kp = detect_keypoints(imgL)
            # DEBUG
            # cv2.imshow("keypoints", img)
            clusters = extract_keypoints(kp)
            cluster_mask = cluster_keypoints(clusters)
            # DEBUG
            # cv2.imshow("clustered", cluster_mask)
            remove_clusters = plain - cluster_mask
            remove_clusters = cv2.bitwise_and(remove_clusters, mask)
            remove_clusters = cv2.bitwise_and(remove_clusters, inv_red)
            remove_clusters = cv2.bitwise_and(remove_clusters, inv_green)
            cv2.imshow("remove_clusters", remove_clusters)
            ret, thresh1 = cv2.threshold(remove_clusters, 1, 255, cv2.THRESH_BINARY)
            # DEBUG
            # cv2.imshow("removed clusters", remove_clusters)
            # cv2.imshow("threshed", thresh1)

            # TODO: Put this in a function
            img_erosion = cv2.erode(thresh1, kernel, iterations=5)
            img_dilation = cv2.dilate(img_erosion, kernel, iterations=2)

            _, contours, _ = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.05 * cv2.arcLength(max_contour, True)
            # approx = cv2.approxPolyDP(max_contour, epsilon, True)
            cv2.drawContours(imgL, [max_contour], 0, (0,0,255), 2)
            # cv2.drawContours(imgL, approx, 0, (0, 0, 255), 2)
            cv2.imshow("contoured", imgL)

            cv2.waitKey(0)
        else:
            print("-- files skipped (perhaps one is missing or not PNG)\n")

# close all windows

cv2.destroyAllWindows()
