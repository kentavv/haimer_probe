#!/usr/bin/env python3

# Copyright 2019 Kent A. Vander Velden <kent.vandervelden@gmail.com>
#
# If you use this software, please consider contacting me. I'd like to hear
# about your work.
#
# This file is part of Haimer-Probe.
#
#     Haimer-Probe is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     Haimer-Probe is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with Haimer-Probe.  If not, see <https://www.gnu.org/licenses/>.


# Ideas and observations:
# 1) The long black pointer passes over the top of the short red pointer.
# 2) The blue dot created by the LED on the camera changes the hue of the pointer tha
#    passes under it and then some amount of the pointer is lost.
# 3) Identify unchanging areas of the image, while the pointer is moving, to
#    know what features can be subtracted. E.g., the dial face.
# 4) For each pixel generate a probability of color to detect change.
# 5) Normalize the image using the uniform dial face.
# 6) Edge detection of the wedge shape arrows before Hough transform is not as
#    convenient as skipping edge detection, in which case Hough transform is
#    a thinning operation is more representative of the midline of the pointer.
# 7) There must be no glare on the dial face. Glare is directly reflected light
#    and easily saturates the camera sensor and obscures all details. If the
#    glare obscures the pointer hands, no measurements are possible. Auto
#    exposure helps as the light intensity changes, but will help with glare
#    and may be hindered by glare.

import math
import os
import sys
import time

import cv2
import numpy as np

c_haimer_ball_diam = 4  # millimeters

c_dial_outer_mask_r = 220

c_red_angle_start = 1.8946996875705893
c_red_angle_end = -1.9406482682728394 + 2 * math.pi
c_initial_image_rot = -.07361130624483032714

c_rho_resolution = 1 / 2  # 1/2 pixel
c_theta_resolution = np.pi / 180 / 4  # 1/4 degree

c_black_outer_mask_r = 130
c_black_outer_mask_e = (120, 130)
c_inner_mask_r = 20
c_red_outer_mask_r = 88

c_black_hough_threshold = 5
c_black_hough_min_line_length = 42  # needs to be larger than the height of the HAIMER label
c_black_hough_max_line_gap = 5
c_black_drawn_line_length = 200
c_red_hough_threshold = 5
c_red_hough_min_line_length = 10
c_red_hough_max_line_gap = 2
c_red_drawn_line_length = 140

c_final_image_scale_factor = 1.

c_label_font = cv2.FONT_HERSHEY_SIMPLEX
c_label_color = (255, 255, 255)
c_label_s = .8

c_line_color = (0, 200, 0)
c_line_s = 2

c_center_offset = [25, -3]
c_image_center = lambda w, h: (w // 2 + c_center_offset[0], h // 2 + c_center_offset[1])


# Decorator for static variables, from
# https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def append_v(lst, v, n=1):
    if v is None:
        return lst
    if len(lst) > n:
        lst.pop(0)
    lst.append(v)


def mean_angles(lst):
    # Because the list of angles can contain both 0 and 2pi,
    # however, 0 and pi are also contained and will average to pi/2,
    # this is thus probably not the best way to do this.
    # https://en.wikipedia.org/wiki/Mean_of_circular_quantities
    return math.atan2(np.mean(np.sin(lst)), np.mean(np.cos(lst)))


def difference_of_angles(theta1, theta2):
    dt = theta1 - theta2
    return math.atan2(math.sin(dt), math.cos(dt))


def list_camera_properties(video_cap):
    capture_properties = [('cv2.CAP_PROP_POS_MSEC', True),
                          ('cv2.CAP_PROP_POS_FRAMES', False),
                          ('cv2.CAP_PROP_POS_AVI_RATIO', False),
                          ('cv2.CAP_PROP_FRAME_WIDTH', True),
                          ('cv2.CAP_PROP_FRAME_HEIGHT', True),
                          ('cv2.CAP_PROP_FPS', True),
                          ('cv2.CAP_PROP_FOURCC', True),
                          ('cv2.CAP_PROP_FRAME_COUNT', False),
                          ('cv2.CAP_PROP_FORMAT', True),
                          ('cv2.CAP_PROP_MODE', True),
                          ('cv2.CAP_PROP_BRIGHTNESS', True),
                          ('cv2.CAP_PROP_CONTRAST', True),
                          ('cv2.CAP_PROP_SATURATION', True),
                          ('cv2.CAP_PROP_HUE', False),
                          ('cv2.CAP_PROP_GAIN', False),
                          ('cv2.CAP_PROP_EXPOSURE', True),
                          ('cv2.CAP_PROP_CONVERT_RGB', True),
                          ('cv2.CAP_PROP_WHITE_BALANCE_BLUE_U', False),
                          ('cv2.CAP_PROP_RECTIFICATION', False),
                          ('cv2.CAP_PROP_MONOCHROME', False),
                          ('cv2.CAP_PROP_SHARPNESS', True),
                          ('cv2.CAP_PROP_AUTO_EXPOSURE', True),
                          ('cv2.CAP_PROP_GAMMA', False),
                          ('cv2.CAP_PROP_TEMPERATURE', True),
                          ('cv2.CAP_PROP_TRIGGER', False),
                          ('cv2.CAP_PROP_TRIGGER_DELAY', False),
                          ('cv2.CAP_PROP_WHITE_BALANCE_RED_V', False),
                          ('cv2.CAP_PROP_ZOOM', True),
                          ('cv2.CAP_PROP_FOCUS', True),
                          ('cv2.CAP_PROP_GUID', False),
                          ('cv2.CAP_PROP_ISO_SPEED', False),
                          ('cv2.CAP_PROP_BACKLIGHT', True),
                          ('cv2.CAP_PROP_PAN', True),
                          ('cv2.CAP_PROP_TILT', True),
                          ('cv2.CAP_PROP_ROLL', False),
                          ('cv2.CAP_PROP_IRIS', False),
                          ('cv2.CAP_PROP_SETTINGS', False),
                          ('cv2.CAP_PROP_BUFFERSIZE', True),
                          ('cv2.CAP_PROP_AUTOFOCUS', True),
                          ('cv2.CAP_PROP_SAR_NUM', False),
                          ('cv2.CAP_PROP_SAR_DEN', False),
                          ('cv2.CAP_PROP_BACKEND', True),
                          ('cv2.CAP_PROP_CHANNEL', True),
                          ('cv2.CAP_PROP_AUTO_WB', True),
                          ('cv2.CAP_PROP_WB_TEMPERATURE', True)
                          ]

    for nm, v in capture_properties:
        if v:
            print(nm, video_cap.get(eval(nm)))


def set_camera_properties(video_cap):
    # capture_properties = [('cv2.CAP_PROP_FRAME_WIDTH', 1280),
    #                       ('cv2.CAP_PROP_FRAME_HEIGHT', 720)
    #                       ]

    capture_properties = [('cv2.CAP_PROP_FRAME_WIDTH', 640),
                          ('cv2.CAP_PROP_FRAME_HEIGHT', 480)
                          ]

    for nm, v in capture_properties:
        if not video_cap.set(eval(nm), v):
            print('Unable to set', nm, v)


# Surprisingly, there is no skeletonization method in OpenCV. It seems common
# that people implement topological skeleton, i.e., thinning using mathematical
# morphology operators. This method may leave many small branches to be pruned.
# Scikit-image, in the morphology module, has skeletonize and medial_axis,
# these are both slower than the hand-coded OpenCV method, especially medial_axis.
# https://en.wikipedia.org/wiki/Topological_skeleton
# https://en.wikipedia.org/wiki/Morphological_skeleton
# https://en.wikipedia.org/wiki/Pruning_(morphology)
# https://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html
# https://stackoverflow.com/questions/25968200/morphology-skeleton-differences-betwen-scikit-image-pymorph-opencv-python
# The following code is from
# https://stackoverflow.com/questions/42845747/optimized-skeleton-function-for-opencv-with-python
def find_skeleton(img):
    skeleton = np.zeros(img.shape, np.uint8)
    eroded = np.zeros(img.shape, np.uint8)
    temp = np.zeros(img.shape, np.uint8)

    _, thresh = cv2.threshold(img, 127, 255, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    iters = 0
    while True:
        cv2.erode(thresh, kernel, eroded)
        cv2.dilate(eroded, kernel, temp)
        cv2.subtract(thresh, temp, temp)
        cv2.bitwise_or(skeleton, temp, skeleton)
        thresh, eroded = eroded, thresh  # Swap instead of copy

        iters += 1
        if cv2.countNonZero(thresh) == 0:
            return skeleton, iters


def filter_lines(lines, image_center, cutoff=5):
    lines2 = []
    for lst in lines:
        x1, y1, x2, y2 = lst[0]
        x0, y0 = image_center

        # Distance between a point (image center) and a line defined by two
        # points, as returned from HoughLinesP
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        d = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        lines2 += [[d < cutoff, lst]]

    return lines2


def plot_lines(lines, theta, drawn_line_len, image, image_center):

    if lines is not None:
        for i in range(len(lines)):
            inc, (x1, y1, x2, y2) = lines[i][0], lines[i][1][0]

            pt1 = (x1, y1)
            pt2 = (x2, y2)
            pt0 = image_center

            cv2.line(image, pt0, pt1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.line(image, pt0, pt2, (255, 0, 0), 1, cv2.LINE_AA)

            cv2.line(image, pt1, pt2, (0, 0, 255) if inc else (255, 0, 0), 3, cv2.LINE_AA)

    if theta is not None:
        a = math.cos(theta)
        b = math.sin(theta)
        x0, y0 = image_center
        pt2 = (round(x0 - drawn_line_len * -b), round(y0 - drawn_line_len * a))
        cv2.line(image, image_center, pt2, c_line_color, c_line_s, cv2.LINE_AA)


def summarize_lines(lines, image_center):
    aa = []

    for lst in lines:
        inc, (x1, y1, x2, y2) = lst[0], lst[1][0]

        if inc:
            pt1 = (x1, y1)
            pt2 = (x2, y2)

            pt0 = image_center

            def h(pt0, pt):
                delta_x = pt[0] - pt0[0]
                delta_y = pt[1] - pt0[1]
                return math.atan2(delta_y, delta_x) + math.pi / 2

            aa += [h(pt0, pt1), h(pt0, pt2)]

    theta = None
    if aa:
        theta = mean_angles(aa)

    return theta


def black_arrow_mask(image, image_center):
    mask = np.zeros(image.shape, dtype=image.dtype)

    # cv2.circle(mask, image_center, c_black_outer_mask_r, (255, 255, 255), -1)
    cv2.ellipse(mask, image_center, c_black_outer_mask_e, 0, 0, 360, (255, 255, 255), -1)

    cv2.circle(mask, image_center, c_inner_mask_r, (0, 0, 0), -1)

    return mask


def red_arrow_mask(image, image_center):
    mask = np.zeros(image.shape, dtype=image.dtype)

    cv2.circle(mask, image_center, c_red_outer_mask_r, (255, 255, 255), -1)
    cv2.circle(mask, image_center, c_inner_mask_r, (0, 0, 0), -1)

    return mask


def black_arrow_segment(image, image_center):
    mask = black_arrow_mask(image, image_center)
    image = cv2.bitwise_and(image, mask)

    if False:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        sat = hsv[:, :, 1] < 80
        val = hsv[:, :, 2] < 180
        seg = sat * val * mask[:, :, 0]
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, red_arrow_mask = red_arrow_segment(image, image_center)

        m = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        red_arrow_mask = cv2.morphologyEx(red_arrow_mask, cv2.MORPH_OPEN, m, iterations=1)
        red_arrow_mask = cv2.morphologyEx(red_arrow_mask, cv2.MORPH_DILATE, m, iterations=2)

        # rv, thres = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        # rv, thres = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        # rv, thres = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE + cv2.THRESH_BINARY_INV)
        # print('threshold_value', rv)

        # thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 5)
        thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 5)
        thres = thres * 255

        thres = np.clip(thres.astype(np.int16) - red_arrow_mask, 0, 255).astype(np.uint8)

        seg = thres * mask[:, :, 0]

    return image, seg


def red_arrow_segment(image, image_center):
    mask = red_arrow_mask(image, image_center)
    image = cv2.bitwise_and(image, mask)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    red = (hsv[:, :, 0] < 15) + (hsv[:, :, 0] > 255 - 15)
    sat = hsv[:, :, 1] > 50
    seg = red * sat * mask[:, :, 0]

    return image, seg


def arrow_common(image, image_center, seg_func, hough_threshold, hough_min_line_length, hough_max_line_gap, ll):
    image, seg0 = seg_func(image, image_center)

    m = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    seg = cv2.morphologyEx(seg0, cv2.MORPH_OPEN, m, iterations=1)

    skel, it = find_skeleton(seg)

    # Edge detection (such as cv2.Canny) returns the edges of the wedge shaped
    # pointer, and the edges point to neither the dial value nor to the center
    # of the dial. So, skip edge detection, and immediately call skeletonization,
    # which is similar to the medial axis of the pointer and immediately useful.

    # Use cv2.HoughLinesP, which compared to cv2.HoughLines, may be faster and has
    # options for minimal line length.
    lines = cv2.HoughLinesP(skel, c_rho_resolution, c_theta_resolution, hough_threshold,
                            minLineLength=hough_min_line_length, maxLineGap=hough_max_line_gap)

    theta = None
    if lines is not None:
        lines = filter_lines(lines, image_center, c_inner_mask_r // 4)
        theta = summarize_lines(lines, image_center)
        plot_lines(lines, theta, ll, image, image_center)

    return theta, image, seg0, skel


def black_arrow(image, image_center):
    return arrow_common(image, image_center, black_arrow_segment, c_black_hough_threshold, c_black_hough_min_line_length, c_black_hough_max_line_gap, c_black_drawn_line_length)


def red_arrow(image, image_center):
    return arrow_common(image, image_center, red_arrow_segment, c_red_hough_threshold, c_red_hough_min_line_length, c_red_hough_max_line_gap, c_red_drawn_line_length)


@static_vars(tare_lst=[], tare_on=False)
def calc_mm(theta_b, theta_r):
    # Blend the course and find measurements of the red and black hands,
    # respectively and return final measurement.

    if calc_mm.tare_on:
        append_v(calc_mm.tare_lst, (theta_b, theta_r), 200)
        print('Tare', len(calc_mm.tare_lst), mean_angles([x[0] for x in calc_mm.tare_lst]), mean_angles([x[1] for x in calc_mm.tare_lst]))

    # Thetas come in as [0, Pi] and [-Pi, 0] so change that to [0, 2Pi]
    if theta_b < 0.:
        theta_b += math.pi * 2
    if theta_r < 0.:
        theta_r += math.pi * 2

    theta_b = max(0., min(math.pi * 2, theta_b))
    theta_r = max(0., min(math.pi * 2, theta_r))

    # Change thetas to millimeters
    mm_b = theta_b / (math.pi * 2) * 1
    mm_r = (theta_r - c_red_angle_start) / (c_red_angle_end - c_red_angle_start) * c_haimer_ball_diam  # - c_haimer_ball_diam / 2

    # The decimal portion of mm_r, to be updated.
    mm_r_d = math.modf(mm_r)[0]

    # Find minimal distance between the black hand, which measure 0-1, and the
    # decimal part of the red hand, which measures [-2, 2], by treating them
    # as angles between [0, 2Pi].
    theta_d = difference_of_angles(theta_b, mm_r_d * math.pi * 2)
    mm_offset = theta_d / (2 * math.pi)

    # Adding the offset to mm_r updates the course red hand measurement with the
    # finer measurement of the black hand. The two estimates of the [0-1] part,
    # the decimal portion of mm_r and mm_b, could be weighted, but here only
    # mm_b is used. Effectively, after the offset is applied, mm_r counts the
    # number of times the black hand has revolved and mm_b measures the fraction
    # of rotation of the black hand.
    mm_blended = mm_r + mm_offset

    # Offset the semifinal measurement by half the probe ball diameter.
    mm_final = mm_blended - c_haimer_ball_diam / 2

    # print(f'{mm_r:8.4f} {mm_b:8.4f} {mm_offset:8.4f} {mm_blended:8.4f} {mm_final:8.4f}')

    return mm_final, mm_b, mm_r


def draw_labels(image, image_b, image_r, theta_b, theta_r, mm_b, mm_r, mm_final):
    cv2.putText(image_b, f'{theta_b:5.2f} rad {mm_b:6.3f} mm', (20, 30 * 1), c_label_font, c_label_s, c_label_color)
    cv2.putText(image_r, f'{theta_r:5.2f} rad {mm_r:6.3f} mm', (20, 30 * 1), c_label_font, c_label_s, c_label_color)
    cv2.putText(image, f'{mm_final:6.3f} mm', (20, 30 * 1), c_label_font, c_label_s, c_label_color)


@static_vars(fps_lst=[], fps_t1=None)
def draw_fps(image):
    if draw_fps.fps_t1 is None:
        draw_fps.fps_t1 = time.time()
        return
    t2 = time.time()
    append_v(draw_fps.fps_lst, 1. / (t2 - draw_fps.fps_t1), 90)
    draw_fps.fps_t1 = t2

    fps = np.mean(draw_fps.fps_lst)

    cv2.putText(image, f'{fps:.2f} fps', (20, 30 * 2), c_label_font, c_label_s, c_label_color)


@static_vars(theta_b_l=[], theta_r_l=[], pause_updates=False)
def get_measurement(video_capture):
    mm_final, mm_b, mm_r = None, None, None

    retval, image0 = video_capture.read()
    if not retval:
        print('rv is false')
        sys.exit(1)
    if image0.size == 0:
        print('image0 is empty')
        sys.exit(1)

    h, w = image0.shape[:2]
    image_center = c_image_center(w, h)

    m = cv2.getRotationMatrix2D(image_center, c_initial_image_rot / math.pi * 180., 1.0)
    image1 = cv2.warpAffine(image0, m, (w, h))
    image2 = image1.copy()

    theta_b, image_b, seg_b, skel_b = black_arrow(image1, image_center)
    seg_b = cv2.cvtColor(seg_b, cv2.COLOR_GRAY2BGR)
    skel_b = cv2.cvtColor(skel_b, cv2.COLOR_GRAY2BGR)

    theta_r, image_r, seg_r, skel_r = red_arrow(image1, image_center)
    seg_r = cv2.cvtColor(seg_r, cv2.COLOR_GRAY2BGR)
    skel_r = cv2.cvtColor(skel_r, cv2.COLOR_GRAY2BGR)

    # Maintain a list of valid thetas for times when no measurements are
    # available, such as when the black hand passes over the red hand, and
    # to use for noise reduction.
    append_v(get_measurement.theta_b_l, theta_b)
    append_v(get_measurement.theta_r_l, theta_r)

    if get_measurement.theta_b_l and get_measurement.theta_r_l:
        theta_b = mean_angles(get_measurement.theta_b_l)
        theta_r = mean_angles(get_measurement.theta_r_l)

        mm_final, mm_b, mm_r = calc_mm(theta_b, theta_r)

    # Draw outer circle dial and crosshairs on dial pivot.
    cv2.circle(image1, (image_center), c_dial_outer_mask_r, c_line_color, c_line_s)
    cv2.line(image1,
             (image_center[0] - c_inner_mask_r, image_center[1] - c_inner_mask_r),
             (image_center[0] + c_inner_mask_r, image_center[1] + c_inner_mask_r),
             c_line_color, 1)
    cv2.line(image1,
             (image_center[0] - c_inner_mask_r, image_center[1] + c_inner_mask_r),
             (image_center[0] + c_inner_mask_r, image_center[1] - c_inner_mask_r),
             c_line_color, 1)

    # Draw black arrow mask
    cv2.circle(image1, image_center, c_black_outer_mask_r, c_line_color, c_line_s)
    cv2.ellipse(image1, image_center, c_black_outer_mask_e, 0, 0, 360, c_line_color, c_line_s)
    cv2.circle(image1, image_center, c_inner_mask_r, c_line_color, c_line_s)

    # Draw red arrow mask
    cv2.circle(image1, image_center, c_red_outer_mask_r, c_line_color, c_line_s)
    cv2.circle(image1, image_center, c_inner_mask_r, c_line_color, c_line_s)

    # Draw final marked up image
    mask = np.zeros(image2.shape, dtype=image2.dtype)
    cv2.circle(mask, image_center, c_dial_outer_mask_r, (255, 255, 255), -1)
    image2 = cv2.bitwise_and(image2, mask)

    # Draw calculated red and black arrows
    if get_measurement.theta_b_l and get_measurement.theta_r_l:
        plot_lines(None, theta_b, c_black_drawn_line_length, image2, image_center)
        plot_lines(None, theta_r, c_red_drawn_line_length, image2, image_center)

        draw_labels(image2, image_b, image_r, theta_b, theta_r, mm_b, mm_r, mm_final)

    draw_fps(image2)

    # Build and display composite image
    img_all0 = np.vstack([image0, image1, image2])
    img_all1 = np.vstack([seg_b, skel_b, image_b])
    img_all2 = np.vstack([seg_r, skel_r, image_r])
    img_all = np.hstack([img_all0, img_all1, img_all2])
    img_all_resized = cv2.resize(img_all, None, fx=c_final_image_scale_factor, fy=c_final_image_scale_factor)

    if not get_measurement.pause_updates:
        cv2.imshow("Live", img_all_resized)
    key = cv2.waitKey(5)
    if key == ord('p'):
        get_measurement.pause_updates = not get_measurement.pause_updates
    elif key == ord('s'):
        for i in range(100):
            fn1 = f'raw_{i:03}.png'
            if not os.path.exists(fn1):
                cv2.imwrite(fn1, image0)
                fn2 = f'all_{i:03}.png'
                cv2.imwrite(fn2, img_all)
                print(f'Wrote images {fn1} and {fn2}')
                break
    elif key == ord('t'):
        if calc_mm.tare_on:
            calc_mm.tare_lst = []
            calc_mm.tare_on = False
        else:
            calc_mm.tare_lst = []
            calc_mm.tare_on = True
    elif 81 <= key <= 84:
        if key == 81:  # KEY_LEFT
            c_center_offset[0] -= 1
        elif key == 82:  # KEY_UP
            c_center_offset[1] -= 1
        elif key == 83:  # KEY_RIGHT
            c_center_offset[0] += 1
        elif key == 84:  # KEY_DOWN
            c_center_offset[1] += 1
        print('c_center_offset:', c_center_offset)
    elif key == ord('q'):
        sys.exit(1)
    elif key >= 0:
        pass
        # print(key)

    return mm_final


def gauge_vision_setup():
    np.set_printoptions(precision=2)

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print('camera is not open')
        sys.exit(1)

    set_camera_properties(video_capture)
    # list_camera_properties(video_capture)

    return video_capture


def main():
    video_capture = gauge_vision_setup()

    while True:
        mm_final = get_measurement(video_capture)
        # print('mm_final:', mm_final)


if __name__ == "__main__":
    main()
