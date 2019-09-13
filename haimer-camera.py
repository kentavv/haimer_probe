#!/usr/bin/env python3

# Cpyright Kent A. Vander Velden, 2019
# kent.vandervelden@gmail.com

import math
import sys
import time

import cv2
import numpy as np

c_red_angle_start = 1.8780948507158541
c_red_angle_end = 4.387011637081153
c_initial_image_rot = -0.062469504755442426
def c_image_center(w, h):
    return (w // 2 + 25 - 3 + 1 - 5 - 4, h // 2 + 2 - 1 - 5)


# Interesting challenges
# 1) The long black pointer passes over the top of the short red pointer.
# 2) The blue dot created by the LED on the camera changes the hue of the pointer tha
#    passes under it and then some amount of the pointer is lost.

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
    capture_properties = [('cv2.CAP_PROP_AUTO_EXPOSURE', True, 1.),  # 1. (manual) and 3. (auto) are valid
                          ('cv2.CAP_PROP_EXPOSURE', True, 20.),  # 'cv2.CAP_PROP_AUTO_EXPOSURE' must be 1. first.

                          ('cv2.CAP_PROP_AUTO_WB', True, 0.),
                          ('cv2.CAP_PROP_WB_TEMPERATURE', True, 2800.),  # 'cv2.CAP_PROP_AUTO_WB' must be 0. first
                          ('cv2.CAP_PROP_TEMPERATURE', True, 2800.),

                          ('cv2.CAP_PROP_AUTOFOCUS', True, 0.),
                          ('cv2.CAP_PROP_FOCUS', True, 22.),  # 'cv2.CAP_PROP_AUTOFOCUS' must be 0. first

                          ('cv2.CAP_PROP_ZOOM', True, 0.),  # Valid values [0, 10], but do not change image

                          ('cv2.CAP_PROP_FRAME_WIDTH', True, 1280),
                          ('cv2.CAP_PROP_FRAME_HEIGHT', True, 720),
                          ('cv2.CAP_PROP_FPS', True, 15),

                          ('cv2.CAP_PROP_BRIGHTNESS', True, -100.),
                          ('cv2.CAP_PROP_CONTRAST', True, 5.),
                          ('cv2.CAP_PROP_SATURATION', True, 83.),

                          ('cv2.CAP_PROP_CONVERT_RGB', True, 1.),  # valid values, 0 and 1. If 0, depth will be 2
                          ('cv2.CAP_PROP_SHARPNESS', True, 25.),

                          ('cv2.CAP_PROP_BACKLIGHT', True, 0.),
                          ('cv2.CAP_PROP_BUFFERSIZE', True, 4.),

                          ('cv2.CAP_PROP_POS_MSEC', False),
                          ('cv2.CAP_PROP_POS_FRAMES', False),
                          ('cv2.CAP_PROP_POS_AVI_RATIO', False),
                          ('cv2.CAP_PROP_FOURCC', False),
                          ('cv2.CAP_PROP_FRAME_COUNT', False),
                          ('cv2.CAP_PROP_FORMAT', False),
                          ('cv2.CAP_PROP_MODE', False),
                          ('cv2.CAP_PROP_HUE', False),
                          ('cv2.CAP_PROP_GAIN', False),

                          ('cv2.CAP_PROP_WHITE_BALANCE_BLUE_U', False),
                          ('cv2.CAP_PROP_RECTIFICATION', False),
                          ('cv2.CAP_PROP_MONOCHROME', False),

                          ('cv2.CAP_PROP_GAMMA', False),
                          ('cv2.CAP_PROP_TRIGGER', False),
                          ('cv2.CAP_PROP_TRIGGER_DELAY', False),
                          ('cv2.CAP_PROP_WHITE_BALANCE_RED_V', False),

                          ('cv2.CAP_PROP_GUID', False),
                          ('cv2.CAP_PROP_ISO_SPEED', False),
                          ('cv2.CAP_PROP_PAN', False),
                          ('cv2.CAP_PROP_TILT', False),
                          ('cv2.CAP_PROP_ROLL', False),
                          ('cv2.CAP_PROP_IRIS', False),
                          ('cv2.CAP_PROP_SETTINGS', False),

                          ('cv2.CAP_PROP_SAR_NUM', False),
                          ('cv2.CAP_PROP_SAR_DEN', False),
                          ('cv2.CAP_PROP_BACKEND', False),
                          ('cv2.CAP_PROP_CHANNEL', False)
                          ]

    for lst in capture_properties:
        nm, enabled = lst[:2]
        if enabled:
            v = lst[2]
            print(nm, video_cap.set(eval(nm), v))

def set_camera_properties2(video_cap):
    capture_properties = [('cv2.CAP_PROP_AUTO_EXPOSURE', True, 1.),  # 1. and 3. are valid
                          ('cv2.CAP_PROP_EXPOSURE', True, 20.),  # 'cv2.CAP_PROP_AUTO_EXPOSURE' must be 1. first.

                          ('cv2.CAP_PROP_AUTO_WB', True, 0.),
                          #('cv2.CAP_PROP_WB_TEMPERATURE', True, 2800.),  # 'cv2.CAP_PROP_AUTO_WB' must be 0. first
                          #('cv2.CAP_PROP_TEMPERATURE', True, 2800.),

                          ('cv2.CAP_PROP_AUTOFOCUS', True, 1.),
                          #('cv2.CAP_PROP_FOCUS', True, 22.),  # 'cv2.CAP_PROP_AUTOFOCUS' must be 0. first

                          #('cv2.CAP_PROP_ZOOM', True, 0.),  # Valid values [0, 10], but do not change image

                          ('cv2.CAP_PROP_FRAME_WIDTH', True, 1280),
                          ('cv2.CAP_PROP_FRAME_HEIGHT', True, 720),
                          #('cv2.CAP_PROP_FPS', True, 15),

                          ('cv2.CAP_PROP_BRIGHTNESS', True, 200.),
                          ('cv2.CAP_PROP_CONTRAST', True, 5.),
                          ('cv2.CAP_PROP_SATURATION', True, 83.),

                          #('cv2.CAP_PROP_CONVERT_RGB', True, 1.),  # valid values, 0 and 1. If 0, depth will be 2
                          #('cv2.CAP_PROP_SHARPNESS', True, 25.),

                          #('cv2.CAP_PROP_BACKLIGHT', True, 0.),
                          #('cv2.CAP_PROP_BUFFERSIZE', True, 4.),

                          ('cv2.CAP_PROP_POS_MSEC', False),
                          ('cv2.CAP_PROP_POS_FRAMES', False),
                          ('cv2.CAP_PROP_POS_AVI_RATIO', False),
                          ('cv2.CAP_PROP_FOURCC', False),
                          ('cv2.CAP_PROP_FRAME_COUNT', False),
                          ('cv2.CAP_PROP_FORMAT', False),
                          ('cv2.CAP_PROP_MODE', False),
                          ('cv2.CAP_PROP_HUE', False),
                          ('cv2.CAP_PROP_GAIN', False),

                          ('cv2.CAP_PROP_WHITE_BALANCE_BLUE_U', False),
                          ('cv2.CAP_PROP_RECTIFICATION', False),
                          ('cv2.CAP_PROP_MONOCHROME', False),

                          ('cv2.CAP_PROP_GAMMA', False),
                          ('cv2.CAP_PROP_TRIGGER', False),
                          ('cv2.CAP_PROP_TRIGGER_DELAY', False),
                          ('cv2.CAP_PROP_WHITE_BALANCE_RED_V', False),

                          ('cv2.CAP_PROP_GUID', False),
                          ('cv2.CAP_PROP_ISO_SPEED', False),
                          ('cv2.CAP_PROP_PAN', False),
                          ('cv2.CAP_PROP_TILT', False),
                          ('cv2.CAP_PROP_ROLL', False),
                          ('cv2.CAP_PROP_IRIS', False),
                          ('cv2.CAP_PROP_SETTINGS', False),

                          ('cv2.CAP_PROP_SAR_NUM', False),
                          ('cv2.CAP_PROP_SAR_DEN', False),
                          ('cv2.CAP_PROP_BACKEND', False),
                          ('cv2.CAP_PROP_CHANNEL', False)
                          ]

    for lst in capture_properties:
        nm, enabled = lst[:2]
        if enabled:
            v = lst[2]
            print(nm, video_cap.set(eval(nm), v))


# https://stackoverflow.com/questions/42845747/optimized-skeleton-function-for-opencv-with-python
def find_skeleton3(img):
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


def standard_form(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    A = (y2 - y1) / (x2 - x1)
    B = 1
    C1 = A * x1 - y1
    C2 = A * x2 - y2
    # print(y2, y1, x2, x1)
    # print(type(C1), type(C2), type(x1), type(x2), type(y1), type(y2))
    # print(pt1, pt2, C1, C2)
    # print(pt1, pt2, A, B, C1)
    return A, B, C1


def test_intersection(a, b, c, r):
    error = .00001
    return not c ** 2 > r ** 2 * (a ** 2 + b ** 2) + error


def filter_lines(lines, image_center):
    lines2 = []
    for lst in lines:
        rho, theta = lst[0]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho

        pt1 = [int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))]
        pt2 = [int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))]

        pt1[0] -= image_center[0]
        pt2[0] -= image_center[0]
        pt1[1] -= image_center[1]
        pt2[1] -= image_center[1]
        try:
            a, b, c = standard_form(pt1, pt2)
            t = test_intersection(a, b, c, 20)
        except ZeroDivisionError:
            x1, y1 = pt1
            x2, y2 = pt2
            # A = (y2 - y1) / (x2 - x1)
            A = 0
            B = y1
            C1 = - y1
            # C2 = A * x2 - y2
            # print(C1, C2)
            # print(A, B, C1)
            a, b, c = A, B, C1
            t = test_intersection(a, b, c, 20)
        if t:
            lines2 += [lst]
        # print(rho, theta, x0, y0, t)

    return lines2


def filter_lines2(lines, image_center, cutoff=5):
    lines2 = []
    d_lst = []
    for lst in lines:
        x1, y1, x2, y2 = lst[0]

        x0, y0 = image_center
        pt1 = [x1, y1]
        pt2 = [x2, y2]

        # Distance between a point (image center) and a line defined by two
        # points (line found by HoughLinesP)
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        d = abs((y2 - y1)*x0 - (x2 - x1) * y0 + x2*y1 - y2*x1) / math.sqrt((y2 - y1)**2 + (x2 - x1)**2)

        d_lst += [d]

        if d < cutoff:
            lines2 += [lst]

        # pt1[0] -= image_center[0]
        # pt2[0] -= image_center[0]
        # pt1[1] -= image_center[1]
        # pt2[1] -= image_center[1]
        #
        # # cast the numpy.int64 (created by the subtraction?) so divisions by zero creates an exception
        # pt1[0] = int(pt1[0])
        # pt1[1] = int(pt1[1])
        # pt2[0] = int(pt2[0])
        # pt2[1] = int(pt2[1])
        # # a, b, c = standard_form(pt1, pt2)
        # try:
        #     a, b, c = standard_form(pt1, pt2)
        #     t = test_intersection(a, b, c, 20)
        # except ZeroDivisionError:
        #     x1, y1 = pt1
        #     x2, y2 = pt2
        #     # A = (y2 - y1) / (x2 - x1)
        #     A = 0
        #     B = y1
        #     C1 = - y1
        #     # C2 = A * x2 - y2
        #     # print(C1, C2)
        #     # print(A, B, C1)
        #     a, b, c = A, B, C1
        #     t = test_intersection(a, b, c, 20)
        # if t:
        #     lines2 += [lst]
        # # print(rho, theta, x0, y0, t)

    d_lst = sorted(d_lst)
    print('d_lst', d_lst)

    return lines2


def plot_lines(lines, image, reject_f):
    aa = []
    bb = []
    med = []
    if lines is not None:
        for i in range(0, len(lines)):
            rho, theta = lines[i][0]

            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho

            if reject_f:
                # print(i, rho, theta, theta / (2 * np.pi) * 360, x0, y0, 'asdf')
                reject = 135 < rho < 170 and 1.5 < theta < 1.8
                # print(i, reject, rho, theta)

                if reject:
                    continue

            # if rho < 200:
            #     continue

            aa += [rho]
            bb += [theta]
            angle = theta / (2 * np.pi) * 360
            med += [angle]

            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    return aa, bb, med


def plot_lines2(lines, image, reject_f):
    aa = []
    bb = []
    med = []
    if lines is not None:
        for i in range(0, len(lines)):
            x1, y1, x2, y2 = lines[i][0]

            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

    return aa, bb, med


def summarize(aa, bb, med, image):
    if aa:
        # print(', '.join(f'({a:.2f}, {b:.2f})' for a, b in sorted(zip(aa, bb))))

        # The angle list can contain d and d+180 deg, where the latter will also have
        # a position that's negative. So, fix that.
        for i, (a, b) in enumerate(zip(aa, bb)):
            # if abs(b - theta2) > np.pi / 2:
            if a < 0:
                aa[i] *= -1
                bb[i] -= math.pi
            # if b > math.pi:
            #     aa[i] *= -1
            #     bb[i] -= math.pi

        # print(', '.join(f'({a:.2f}, {b:.2f})' for a, b in sorted(zip(aa, bb))))
        # print(list(zip(aa, bb)))

        # Averaging the angles may now be OK, without the need to perform
        # the more advance version below that handles "similar" angles like 0, 180, 360
        rho = np.mean(aa)
        theta = np.mean(bb)
        # med = np.mean(med)

        # Because the list of angles can contain both 0 and 2pi,
        # however, 0 and pi are also contained and will average to pi/2,
        # this is thus probably not the best way to do this.
        # https://en.wikipedia.org/wiki/Mean_of_circular_quantities
        theta2 = math.atan2(np.mean(np.sin(bb)), np.mean(np.cos(bb)))
        # print(f'theta {theta:.4f} - theta2 {theta2:.4f} = {theta - theta2:.4f}')
        theta = theta2

        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(image, pt1, pt2, (0, 255, 255), 3, cv2.LINE_AA)


def summarize2(lines, image, image_center, ll):
    aa = []
    for lst in lines:
        x1, y1, x2, y2 = lst[0]
        pt1 = (x1, y1)
        pt2 = (x2, y2)

        pt0 = image_center

        def h(pt0, pt):
            delta_x = pt[0] - pt0[0]
            delta_y = pt[1] - pt0[1]

            theta_radians = math.atan2(delta_y, delta_x) + math.pi / 2
            theta_deg = theta_radians / (2 * math.pi) * 360

            return theta_deg

        aa += [h(pt0, pt1), h(pt0, pt2)]
        # print('calc:', aa[-2:])

        cv2.line(image, pt0, pt1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.line(image, pt0, pt2, (255, 0, 0), 1, cv2.LINE_AA)

    theta = None
    if aa:
        # print(np.mean(aa), np.median(aa))

        theta = np.mean(aa) / 360. * 2. * math.pi

        a = math.cos(theta)
        b = math.sin(theta)
        x0, y0 = image_center
        # pt1 = (int(x0 + ll * (-b)), int(y0 + ll * (a)))
        pt2 = (int(x0 - ll * (-b)), int(y0 - ll * (a)))
        cv2.line(image, image_center, pt2, (0, 255, 255), 1, cv2.LINE_AA)

    return theta

    bb = []
    for lst in lines:
        x1, y1, x2, y2 = lst[0]

        pt1 = [x1, y1]
        pt2 = [x2, y2]

        delta_x = x2 - x1
        delta_y = y2 - y1

        theta_radians = math.atan2(delta_y, delta_x) + math.pi / 2
        theta_deg = theta_radians / (2 * math.pi) * 360
        # print('hi111', theta_radians, theta_deg)

        bb += [theta_radians]

    # print('bb', bb)
    theta = None
    if bb:
        # Because the list of angles can contain both 0 and 2pi,
        # however, 0 and pi are also contained and will average to pi/2,
        # this is thus probably not the best way to do this.
        # https://en.wikipedia.org/wiki/Mean_of_circular_quantities
        theta2 = math.atan2(np.mean(np.sin(bb)), np.mean(np.cos(bb)))
        theta = theta2

        a = math.cos(theta)
        b = math.sin(theta)

        x0, y0 = image_center
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(image, pt1, pt2, (0, 255, 255), 3, cv2.LINE_AA)

    return theta


def black_arrow(image, image_center):
    image = image.copy()

    mask = np.zeros(image.shape, dtype=image.dtype)
    if False:
        cv2.circle(mask, image_center, 290, (255, 255, 255), -1)
    else:
        cv2.circle(mask, image_center, 290//2, (255, 255, 255), -1)
    # image = image * (mask.astype(image.dtype))
    # image = image * mask
    image = cv2.bitwise_and(image, mask)

    mask = np.zeros(image.shape, dtype=image.dtype)

    if False:
        # cv2.circle(mask, image_center, 210, (255, 255, 255), -1)
        cv2.ellipse(mask, image_center, (200, 210), 0, 0, 360, (255, 255, 255), -1)

        # cv2.circle(mask, image_center, 80, (255, 255, 255), -1)
        cv2.circle(mask, image_center, 35, (0, 0, 0), -1)
    else:
        # cv2.circle(mask, image_center, 210, (255, 255, 255), -1)
        cv2.ellipse(mask, image_center, (125, 135), 0, 0, 360, (255, 255, 255), -1)

        # cv2.circle(mask, image_center, 80, (255, 255, 255), -1)
        cv2.circle(mask, image_center, 22, (0, 0, 0), -1)

    # image = image * (mask.astype(image.dtype))
    # image = image * mask
    image = cv2.bitwise_and(image, mask)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    val0 = (hsv[:, :, 2] < 120) * mask[:, :, 0]

    m = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # val = cv2.erode(val0, m, iterations=1)
    val = cv2.morphologyEx(val0, cv2.MORPH_OPEN, m, iterations=1)

    val, it = find_skeleton3(val)
    # print('it=', it)

    # val = (val * 255).astype(np.uint8)
    # print(red.shape)
    # dst = cv2.Canny(val, 50, 255, None, 3)
    dst = val.copy()
    # print(np.count_nonzero(dst))
    rho_resolution = 1 / 2  # 1/2 pixel
    theta_resolution = np.pi / 180 / 4  # 1/4 degree
    # lines = cv2.HoughLines(dst, rho_resolution, theta_resolution, 50)
    # minLinLength for black arrow needs to be larger than the height of the HAIMER label
    if False:
        lines = cv2.HoughLinesP(dst, rho_resolution, theta_resolution, 10, minLineLength=45, maxLineGap=5)
    else:
        lines = cv2.HoughLinesP(dst, rho_resolution, theta_resolution, 10//2, minLineLength=45//2, maxLineGap=5//2)
    theta = None
    if lines is not None:
        # print(lines)

        print('----------------------------1')
        # lines = filter_lines(lines, image_center)
        lines = filter_lines2(lines, image_center)

        # aa, bb, med = plot_lines(lines, image, False)
        aa, bb, med = plot_lines2(lines, image, False)
        # summarize(aa, bb, med, image)

        theta = summarize2(lines, image, image_center, 300)
        print('----------------------------2')

    return val0, val, dst, image, theta


def red_arrow(image, image_center):
    image = image.copy()

    mask = np.zeros(image.shape, dtype=image.dtype)
    if False:
        cv2.circle(mask, image_center, 140, (255, 255, 255), -1)
        cv2.circle(mask, image_center, 35, (0, 0, 0), -1)
    else:
        cv2.circle(mask, image_center, 88, (255, 255, 255), -1)
        cv2.circle(mask, image_center, 20, (0, 0, 0), -1)
    # image = image * (mask.astype(image.dtype))
    # image = image * mask
    image = cv2.bitwise_and(image, mask)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    red = (hsv[:, :, 0] < 15) + (hsv[:, :, 0] > 255 - 15)
    sat = hsv[:, :, 1] > 50
    red0 = (red * sat * 255).astype(np.uint8)

    m = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # red = cv2.erode(red0, m, iterations=1)
    red = cv2.morphologyEx(red0, cv2.MORPH_OPEN, m, iterations=1)

    red, it = find_skeleton3(red)
    # print('it=', it)

    # print(red.shape)
    # dst = cv2.Canny(red, 50, 255, None, 3)
    dst = red.copy()
    # print(np.count_nonzero(dst))
    rho_resolution = 1 / 2  # 1/2 pixel
    theta_resolution = np.pi / 180 / 4  # 1/4 degree
    # lines = cv2.HoughLines(dst, rho_resolution, theta_resolution, 30, None, 0, 0)
    if False:
        lines = cv2.HoughLinesP(dst, rho_resolution, theta_resolution, 10, minLineLength=20, maxLineGap=5)
    else:
        lines = cv2.HoughLinesP(dst, rho_resolution, theta_resolution, 10//2, minLineLength=20//2, maxLineGap=5//2)
    theta = None
    if lines is not None:
        # print(lines)

        # lines = filter_lines(lines, image_center)
        lines = filter_lines2(lines, image_center)

        # aa, bb, med = plot_lines(lines, image, False)
        aa, bb, med = plot_lines2(lines, image, False)
        # summarize(aa, bb, med, image)
        theta = summarize2(lines, image, image_center, 200)

    return red0, red, dst, image, theta


def draw_labels(image, theta1, theta2):
    font = cv2.FONT_HERSHEY_DUPLEX
    bb = theta1 / (math.pi * 2) * 1
    rr = (theta2 - c_red_angle_start) / (c_red_angle_end - c_red_angle_start) * 4 - 2
    cv2.putText(image, f'b {theta1:.2f} {bb:.2f}', (20, 30), font, 1, (255, 255, 255))
    cv2.putText(image, f'r {theta2:.2f} {rr:.2f}', (20, 60), font, 1, (255, 255, 255))


lll = []
t1 = None
def draw_fps(image):
    global t1, lll

    font = cv2.FONT_HERSHEY_DUPLEX
    if t1 is None:
        t1 = time.time()
        return
    t2 = time.time()
    if len(lll) > 90:
        lll = lll[1:]
    lll = [1. / (t2 - t1)] + lll
    t1 = t2
    # print('lll', lll)
    llm = np.mean(lll)
    cv2.putText(image, f'fps {llm:.2f}', (20, 90), font, 1, (255, 255, 255))


def main():
    np.set_printoptions(precision=2)

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print('camera is not open')
        sys.exit(1)

    list_camera_properties(video_capture)
    # print()
    # set_camera_properties(video_capture)
    # set_camera_properties2(video_capture)
    # print()
    # list_camera_properties(video_capture)

    # sys.exit(1)

    pimg = [None] * 1
    theta1_l = []
    theta2_l = []
    while True:
        retval, cimg = video_capture.read()
        if not retval:
            print('rv is false')
            sys.exit(1)
        if cimg.size == 0:
            print('image is empty')
            sys.exit(1)

        # cv2.imshow("Live", cimg)
        # if cv2.waitKey(5) >= 0:
        #     break
        # # set_camera_properties2(video_capture)
        # continue

        pimg = pimg[1:] + [cimg]
        if pimg[0] is None:
            continue

        image = np.mean(pimg, axis=0).astype(np.uint8)
        # cv2.imwrite('test.png', image)
        # break
        # print(image.shape)

        h, w = image.shape[:2]
        image_center = c_image_center(w, h)

        m = cv2.getRotationMatrix2D(image_center, c_initial_image_rot / math.pi * 180., 1.0)
        image = cv2.warpAffine(image, m, (w, h))

        # image = cv2.CLAHE.apply(image)

        val01, val1, dst1, image1, theta1 = black_arrow(image, image_center)
        val01 = cv2.cvtColor(val01, cv2.COLOR_GRAY2BGR)
        dst1 = cv2.cvtColor(dst1, cv2.COLOR_GRAY2BGR)
        # val1 = cv2.cvtColor(val1, cv2.COLOR_GRAY2BGR)
        # img_all1 = np.vstack([cimg, val01, val1, dst1, image1])

        val02, val2, dst2, image2, theta2 = red_arrow(image, image_center)
        val02 = cv2.cvtColor(val02, cv2.COLOR_GRAY2BGR)
        dst2 = cv2.cvtColor(dst2, cv2.COLOR_GRAY2BGR)
        # val2 = cv2.cvtColor(val2, cv2.COLOR_GRAY2BGR)
        # img_all2 = np.vstack([cimg, val02, val2, dst2, image2])

        # maintain a list of thetas to average to reduce noise and to fill in
        # during times where not measurements are available such as when the
        # black hand passes over the red hand
        print(theta1, theta2)
        if theta1:
            if theta1 < 0.:
                theta1 += math.pi * 2
            if len(theta1_l) > 1:
                theta1_l = theta1_l[:-1]
            theta1_l = [theta1] + theta1_l
        if theta2:
            if theta2 < 0.:
                theta2 += math.pi * 2
            if len(theta2_l) > 1:
                theta2_l = theta2_l[:-1]
            theta2_l = [theta2] + theta2_l

        image0 = image.copy()
        cv2.circle(image0, (image_center), 15, (0, 0, 255), 1)
        cv2.line(image0, (image_center[0]-20, image_center[1]-20), (image_center[0]+20, image_center[1]+20), (0, 0, 255), 1)
        cv2.line(image0, (image_center[0]-20, image_center[1]+20), (image_center[0]+20, image_center[1]-20), (0, 0, 255), 1)

        # black arrow
        cv2.circle(image0, image_center, 290 // 2, (0, 255, 255), 1)
        cv2.ellipse(image0, image_center, (125, 135), 0, 0, 360, (0, 255, 255), 1)
        cv2.circle(image0, image_center, 22, (0, 255, 255), 1)

        # red arrow
        cv2.circle(image0, image_center, 88, (0, 255, 255), 1)
        cv2.circle(image0, image_center, 20, (0, 255, 255), 1)

        mask = np.zeros(image.shape, dtype=image.dtype)
        cv2.circle(mask, image_center, 220, (255, 255, 255), -1)
        image = cv2.bitwise_and(image, mask)

        if theta1_l and theta2_l:
            print(len(theta2_l))
            theta1m = np.mean(theta1_l)
            theta2m = np.mean(theta2_l)
            print(theta1, theta2, theta1m, theta2m)
            theta1 = theta1m
            theta2 = theta2m

            if True:
                ll = 300
                a = math.cos(theta1)
                b = math.sin(theta1)
                x0, y0 = image_center
                # pt1 = (int(x0 + ll * (-b)), int(y0 + ll * (a)))
                pt2 = (int(x0 - ll * (-b)), int(y0 - ll * (a)))
                cv2.line(image, image_center, pt2, (0, 255, 255), 1, cv2.LINE_AA)
            if True:
                ll = 200
                a = math.cos(theta2)
                b = math.sin(theta2)
                x0, y0 = image_center
                # pt1 = (int(x0 + ll * (-b)), int(y0 + ll * (a)))
                pt2 = (int(x0 - ll * (-b)), int(y0 - ll * (a)))
                cv2.line(image, image_center, pt2, (0, 255, 255), 1, cv2.LINE_AA)

            draw_labels(image, theta1, theta2)
            draw_fps(image)

        img_all0 = np.vstack([cimg, image0, image])
        img_all1 = np.vstack([val01, dst1, image1])
        img_all2 = np.vstack([val02, dst2, image2])

        img_all = np.hstack([img_all0, img_all1, img_all2])
        # img_all = img_all1
        s = 1.
        img_all = cv2.resize(img_all, None, fx=s, fy=s)

        cv2.imshow("Live", img_all)
        if cv2.waitKey(5) >= 0:
            break

    print()
    print()
    list_camera_properties(video_capture)


main()
