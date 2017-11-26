# # Reduce re-computation
# # Create new array with flags specifying if element is within threshold
# # Element-wise multiplication such that only element within threshold is kept
# # Everything else is 0
# def find_inliers(points, distances, thrsh):
#     inliers = np.select([distances <= thrsh, distances > thrsh], [1.0, 0.0])
#     return np.multiply(points, inliers)
#
#
# ########################################################
# #             DRAWING HELPER FUNCTIONS                 #
# ########################################################
# #  IGNORE THIS FILE: Unused/refactored functions       #
# ########################################################
#
# # Get the centre of the shape that we want to plot
# # O(n)
# # Plot the shape
# def draw_polygon(points, cnt, image):
#     min_x = float("inf")
#     max_x = 0
#     min_y = float("inf")
#     max_y = 0
#     for point in points:
#         if point[0] < min_x:
#             min_x = point[0]
#         elif point[0] > max_x:
#             max_x = point[0]
#
#         if point[1] < min_y:
#             min_y = point[1]
#         elif point[1] > max_y:
#             max_y = point[1]
#
#     mid_x = (min_x + max_x) // 2
#     mid_y = (min_y + max_y) // 2
#     image = draw_trapezium(image, cnt, mid_x, mid_y)
#     return image
#
#
# # Fix size of trapezium
# def draw_trapezium(img, contours, midpoint_x, midpoint_y):
#
#     # Top x-axes
#     tl = midpoint_x - 100
#     tr = midpoint_x + 100
#
#     # Bottom x-axes
#     bl = midpoint_x - 300
#     br = midpoint_x + 300
#
#     # y-axes
#     ty = midpoint_y - 75
#     by = midpoint_y + 50
#
#     # Prepare to bump plane
#     bump_to_left = 0
#     bump_to_right = 0
#     # Draw cluster bounding boxes first
#     for contour in contours:
#         if cv2.arcLength(contour, False) > 90:
#             x, y, w, h = cv2.boundingRect(contour)
#             box = {
#                 'left_x': x,
#                 'top_y': y,
#                 'right_x': x + w,
#                 'bottom_y': y + h
#             }
#
#             if top_collision_detected(box, tl, tr, ty):
#                 cv2.rectangle(imgL, (x, y), (x+w, y+h), (0, 0, 255), 2)
#             # DEBUG
#             # else:
#                 # cv2.rectangle(imgL, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#             left_shift, right_shift = side_collision_detection(box, tl, tr, bl, br, ty)
#             if left_shift > bump_to_right:
#                 bump_to_right = min(left_shift, 50)
#             if right_shift > bump_to_left:
#                 bump_to_left = min(right_shift, 50)
#
#     vertices = np.array([[tl + bump_to_right, ty], [tr - bump_to_left, ty],
#                          [br - bump_to_left, by], [bl + bump_to_right, by]], np.int32)
#     vertices = vertices.reshape((-1, 1, 2))
#     img = cv2.polylines(img, [vertices], True, (0, 0, 255), 1)
#
#     return img
#
#
# def top_collision_detected(cluster_box, left, right, top_y):
#     if cluster_box['right_x'] <= left or cluster_box['left_x'] >= right:
#         return False
#     if cluster_box['bottom_y'] < top_y:
#         return False
#     return True
#
#
# def side_collision_detection(cluster_box, tl, tr, bl, br, top_y):
#     if cluster_box['bottom_y'] < top_y:
#         return 0, 0
#
#     left_shift = 0
#     right_shift = 0
#
#     # Shift plane to right
#     if tr < cluster_box['left_x'] < br:
#         box_bottom_intersection = trapezium_linear_x_right(cluster_box['bottom_y'])
#         left_shift = cluster_box['left_x'] - box_bottom_intersection
#
#     # Shift left
#     if bl < cluster_box['right_x'] < tl:
#         box_bottom_intersection = trapezium_linear_x_left(cluster_box['bottom_y'])
#         right_shift = cluster_box['right_x'] - box_bottom_intersection
#
#     return left_shift, right_shift
#
#
# # Find x-intersection of bottom of box with trapezium
# #        /
# #       |
# #      /|
# #     / |    E.g. left of trapezium
# #  --x--|
# #   /
# #  /
# def trapezium_linear_x_left(y_intersection):
#     return int((424.25 - y_intersection) * 8/5)
#
#
# # Find x-intersection of box on right
# def trapezium_linear_x_right(y_intersection):
#     return int((y_intersection + 133.25) * 8/5)
#
#
# def make_length(start_point, end_point, length):
# 	translated_to_origin = (end_point[0] - start_point[0], end_point[1] - start_point[1])
# 	print(translated_to_origin)
# 	magnitude = math.sqrt(translated_to_origin[0]**2 + translated_to_origin[1]**2)
# 	print(magnitude)
# 	scaled_translated_to_origin = (translated_to_origin[0] * 1. / magnitude * length,
#                                  translated_to_origin[1] * 1. / magnitude * length)
# 	print(scaled_translated_to_origin)
# 	scaled_original = (scaled_translated_to_origin[0] + start_point[0],
#                      scaled_translated_to_origin[1] + start_point[1])
# 	return scaled_original
