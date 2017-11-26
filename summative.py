#########################################################
#               COMPUTER VISION SUMMATIVE               #
#########################################################
# Before running the code, specify the path to your     #
# dataset in the variable below. Then run the program   #
# by calling the main method, i.e.                      #
#     python3 summative.py                              #
#########################################################
# NOTE: The code will automatically cycle through the   #
# images in the dataset, but does not support any key-  #
# press functionality. It takes a while to process so   #
# do not be alarmed if it appears to have stopped.      #
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

#####################################################################

orb = cv2.ORB_create()


#########################################################
#             ILLUMINATION PRE-PROCESSING               #
#########################################################
# Reference: https://stackoverflow.com/questions/       #
# 18452438/how-can-i-remove-drastic-brightness-         #
# variations-in-a-video                                 #
#########################################################

def remove_illumination(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(image)
    y = cv2.equalizeHist(y)
    # Remove low frequency details of the image
    blur_y = cv2.GaussianBlur(y, (23, 23), 0)
    y = y - blur_y
    image = cv2.merge((y, u, v))
    return cv2.cvtColor(image, cv2.COLOR_YUV2BGR)


#########################################################
#                 HISTOGRAM MATCHING                    #
#########################################################
# Reference: http://vzaguskin.github.io/histmatching1/  #
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


# Create a mask that blacks out all the detected objects
def cluster_keypoints(extracted_kp):
    img = cv2.imread('image_assets/plain_black.png', 0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 10)
    temp, classified_points, centers = cv2.kmeans(extracted_kp, K=10, bestLabels=None,
                                                  criteria=crit, attempts=1,
                                                  flags=cv2.KMEANS_RANDOM_CENTERS)
    for pt, allocation in zip(extracted_kp, classified_points):
        color = (255, 255, 255)
        cv2.circle(img, (int(pt[0]), int(pt[1])), 8, color, -1)

    _, conts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in conts:
        if cv2.arcLength(contour, False) > 90:
            x, y, w, h = cv2.boundingRect(contour)
            box = np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], dtype=np.int32)
            cv2.fillPoly(img, box, (255, 255, 255))

    return img, conts


def remove_objects(black_bg, left_img):
    # Find objects in scene
    img, kp = detect_keypoints(left_img)
    # DEBUG
    # cv2.imshow("keypoints", img)
    clusters = extract_keypoints(kp)
    cluster_mask, contours_found = cluster_keypoints(clusters)
    # DEBUG
    cv2.imshow("clustered", cluster_mask)
    remove_clusters = black_bg - cluster_mask
    remove_clusters = cv2.bitwise_and(remove_clusters, mask)
    remove_clusters = cv2.bitwise_and(remove_clusters, red_mask)
    remove_clusters = cv2.bitwise_and(remove_clusters, green_mask)
    # DEBUG
    # cv2.imshow("remove_clusters", remove_clusters)
    ret, thresh1 = cv2.threshold(remove_clusters, 1, 255, cv2.THRESH_BINARY)
    # DEBUG
    # cv2.imshow("removed clusters", remove_clusters)
    # cv2.imshow("threshed", thresh1)

    img_erosion = cv2.erode(thresh1, kernel, iterations=5)
    dilation = cv2.dilate(img_erosion, kernel, iterations=2)
    return dilation, contours_found


def collision_detected(cluster_box, top_left, top_right, top_y, bottom_y):
    if cluster_box['bottom_y'] < top_y + 5:
        return False

    cut_off_point = top_y + ((bottom_y - top_y) // 2)
    if cluster_box['top_y'] > cut_off_point:
        return False

    if cluster_box['right_x'] <= top_left or \
            cluster_box['left_x'] >= top_right:
        return False

    return True


#########################################################
#                COLOR SPACE PROCESSING                 #
#########################################################

# Remove red and green color regions
# Assumes the car will not be driven on grass
def remove_colors(left_image):
    # Remove any green regions
    hsv_l = cv2.cvtColor(left_image, cv2.COLOR_BGR2HSV)
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
    # DEBUG
    # cv2.imshow("red regions", inv_red)
    return inv_green, inv_red


#####################################################################

#########################################################
#                  PRE-PROCESS IMAGES                   #
#########################################################

def preprocess(img_l, img_r):
    # DEBUG
    # cv2.imshow("img_l", img_l)
    illum_removed_l = remove_illumination(img_l)
    illum_removed_r = remove_illumination(img_r)
    # DEBUG
    # cv2.imshow("illum_removed_l", illum_removed_l)
    # N.B. need to do for both as both are 3-channel images
    gray_l = cv2.cvtColor(illum_removed_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(illum_removed_r, cv2.COLOR_BGR2GRAY)

    ret, thresh_r = cv2.threshold(gray_r, 20, 100, cv2.THRESH_TOZERO)
    ret, thresh_l = cv2.threshold(gray_l, 20, 100, cv2.THRESH_TOZERO)
    # DEBUG
    # cv2.imshow("threshed_r", thresh_R)
    # cv2.imshow("threshed_l", thresh_L)

    gray_r_matched = match(thresh_r, thresh_l)
    gray_l_matched = match(thresh_l, thresh_r)
    # DEBUG
    # cv2.imshow("matched_r", gray_r_matched)

    return gray_l_matched, gray_r_matched


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


def project_disparity_to_3d(disparity):
    points = []
    f = camera_focal_length_px
    b = stereo_camera_baseline_m

    height, width = disparity.shape[:2]

    for h in range(height):  # 0 - height is the y axis index
        for w in range(width):  # 0 - width is the x axis index

            # if we have a valid non-zero disparity
            if disparity[h, w] > 0:

                # calculate corresponding 3D point [X, Y, Z]
                # stereo lecture - slide 22 + 25
                z = (f * b) / disparity[h, w]
                x = ((w - image_centre_w) * z) / f
                y = ((h - image_centre_h) * z) / f

                points.append([x, y, z])

    return points


#########################################################
#               DISPARITY CALCULATIONS                  #
#########################################################

# remember to convert to grayscale (as the disparity matching works on grayscale)
def get_disparity(left_image, right_image):
    disparity = stereoProcessor.compute(left_image, right_image)
    # filter out noise and speckles (adjust parameters as needed)

    disp_noise_filter = 5  # increase for more aggressive filtering
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - disp_noise_filter)

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
    inlier_points = []
    best_normal = np.array([0, 0, 0])
    smallest_distance = float("inf")
    closest_to_plane = None
    while iterations < max_iterations:
        temp_inliers = []
        # Put in try block, in case plane cannot be found
        try:
            plane_abc, plane_d = fit_plane(points)
            distances = get_distance_from_plane(points, plane_abc, plane_d)
            for i in range(len(distances)):
                if distances[i] < inlier_thresh:
                    temp_inliers.append(points[i])

                    if distances[i] < smallest_distance:
                        smallest_distance = distances[i]
                        closest_to_plane = points[i]

            # Run many iterations and find plane with the most points that agree
            if len(temp_inliers) > len(inlier_points):
                inlier_points = temp_inliers
                best_normal = np.divide(plane_abc, plane_d)

            iterations += 1
        except Exception:
            break

    return inlier_points, best_normal, closest_to_plane


#########################################################
#         OBJECT DETECTION HELPER FUNCTIONS             #
#########################################################

# Get the centre of the trapezoid that we want to use
# O(n)
# Checks for objects that enters the trapezoidal region
def draw_polygon(points, cnt, image):
    min_x = float("inf")
    max_x = 0
    min_y = float("inf")
    max_y = 0
    for pnt in points:
        if pnt[0][0] < min_x:
            min_x = pnt[0][0]
        elif pnt[0][0] > max_x:
            max_x = pnt[0][0]

        if pnt[0][1] < min_y:
            min_y = pnt[0][1]
        elif pnt[0][1] > max_y:
            max_y = pnt[0][1]

    mid_x = (min_x + max_x) // 2
    trapezium_length = (max_y - min_y) * (16/9)
    image = draw_trapezium(image, cnt, mid_x, min_y, trapezium_length)
    return image


# Creates invisible trapezoid ROI
def draw_trapezium(img, cnts, midpoint_x, top_y, shape_length):
    # Top x-axes
    tl = midpoint_x - (shape_length // 2.5)
    tr = midpoint_x + (shape_length // 2.5)

    # DEBUG
    # Bottom x-axes
    # bl = midpoint_x - shape_length
    # br = midpoint_x + shape_length

    # Draw cluster bounding boxes first
    for contour in cnts:
        if cv2.arcLength(contour, False) > 90:
            x, y, w, h = cv2.boundingRect(contour)
            box = {
                'left_x': x,
                'top_y': y,
                'right_x': x + w,
                'bottom_y': y + h
            }

            if collision_detected(box, tl, tr, top_y):
                cv2.rectangle(imgL, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # DEBUG
            # else:
                # cv2.rectangle(imgL, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # DEBUG
    # vertices = np.array([[tl, top_y], [tr, top_y], [br, bottom_y], [bl, bottom_y]], np.int32)
    # vertices = vertices.reshape((-1, 1, 2))
    # img = cv2.polylines(img, [vertices], True, (0, 0, 255), 1)

    return img


#########################################################
#                 NORMAL CALCULATIONS                   #
#########################################################
# Reference: http://mlikihazar.blogspot.com.au/2013/    #
# 02/draw-arrow-opencv.html                             #
#########################################################

def draw_normal_vector(constant, image, plane_point, normal):
    to_subtract = np.multiply(np.absolute(normal), constant)
    end_of_point = plane_point - to_subtract.flatten()
    results = project_3d_points_to_2d_image_points([plane_point, end_of_point])
    # Get coordinate values; standardize location
    p_x, p_y = 150, 100
    q_x, q_y = results[1][0] - (results[0][0] - 150), results[1][1] - (results[0][1] - 100)
    normalized_x, normalized_y = make_unit((p_x, p_y), (q_x, q_y))

    # Draw glyph box
    cv2.rectangle(image, (100, 50), (200, 100), (254, 117, 31), 3)
    cv2.putText(image, 'NORMAL', (103, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (254, 117, 31), 2, cv2.LINE_AA)

    # Draw arrow in box
    draw_arrow(image, (int(p_x), int(p_y)), (int(normalized_x), int(normalized_y)), (254, 117, 31))
    return image


# Normalize the normal vector
def make_unit(start_point, end_point):
    translated_to_origin = (end_point[0] - start_point[0], end_point[1] - start_point[1])
    magnitude = math.sqrt(translated_to_origin[0] ** 2 + translated_to_origin[1] ** 2)
    scaled_translated_to_origin = (translated_to_origin[0] / magnitude * 35, translated_to_origin[1] / magnitude * 35)
    scaled_original = (scaled_translated_to_origin[0] + start_point[0], scaled_translated_to_origin[1] + start_point[1])
    return scaled_original


def draw_arrow(image, p, q, color, arrow_magnitude=9, thickness=2, line_type=8, shift=0):
    # draw arrow tail
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # calc angle of the arrow
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    # starting point of first line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
         int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # starting point of second line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
         int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    return image


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

            red_mask, green_mask = remove_colors(imgL)

            # Histogram equalization
            preprocessed_L, preprocessed_R = preprocess(imgL, imgR)
            # DEBUG
            # cv2.imshow("preprocessed_L", preprocessed_L)
            # cv2.imshow("preprocessed_R", preprocessed_R)

            # Apply mask to only look at region in front of car hood;
            # reduces computation
            preprocessed_L = cv2.bitwise_and(preprocessed_L, mask)
            preprocessed_R = cv2.bitwise_and(preprocessed_R, mask)
            # DEBUG
            # cv2.imshow("masked_L", preprocessed_L)
            # cv2.imshow("masked_R", preprocessed_R)
            # print("-- files loaded successfully\n")

            disparity_scaled = get_disparity(preprocessed_L, preprocessed_R)
            # display image (scaling it to the full 0->255 range based on the number
            # of disparities in use for the stereo part)
            dsp = (disparity_scaled * (256. / max_disparity)).astype(np.uint8)
            depth_points = project_disparity_to_3d(dsp)
            # DEBUG
            # cv2.imshow("depth", dsp)

            # Run RANSAC, keep only inliers
            threshold = 0.05
            best_plane, normal_coeffs, closest_pt_to_plane = ransac_plane(depth_points, threshold)
            points_to_draw = project_3d_points_to_2d_image_points(best_plane)

            # Numbers used to generate offset values
            dsp_x, dsp_y = dsp.shape[1], dsp.shape[0]
            img_x, img_y = imgL.shape[1], imgL.shape[0]
            for point in points_to_draw:
                cv2.circle(plain, (int(point[0]) + (img_x - dsp_x), int(point[1])), 1, (255, 255, 255), 1)
            # DEBUG
            # cv2.imshow("drawn circles", plain)

            # Remove noise
            img_dilation, img_contours = remove_objects(plain, imgL)

            # Draw the plane contour
            _, contours, _ = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            max_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.002 * cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, epsilon, True)
            cv2.drawContours(imgL, [approx], -1, (0, 0, 255), 2)

            # Find objects in plane in front of car
            imgL = draw_polygon(approx, img_contours, imgL)

            # Draw glyph for normal
            draw_normal_vector(0.2, imgL, closest_pt_to_plane, normal_coeffs)

            # Print coefficients of plane normal
            normal_coeffs = [str(x) for x in np.array(normal_coeffs).flatten()]
            print(filename_left)
            print(filename_right + " : road surface normal (" +
                  normal_coeffs[0] + ", " +
                  normal_coeffs[1] + ", " +
                  normal_coeffs[2] + ")"
                  )
            cv2.imshow("Output", imgL)
            cv2.waitKey(5)  # wait 5 s before going to next frame
        else:
            print("-- files skipped (perhaps one is missing or not PNG)\n")

# close all windows
cv2.destroyAllWindows()
