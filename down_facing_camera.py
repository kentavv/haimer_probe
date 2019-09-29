#!/usr/bin/env python

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
# 1) How should non-circular holes be supported?

import math
import os
import sys
import time

import cv2
import numpy as np

c_initial_image_rot = (-3.04) / 180. * math.pi

c_final_image_scale_factor = 1

c_label_font = cv2.FONT_HERSHEY_SIMPLEX
c_label_color = (63, 255, 63)
c_label_s = .8

c_line_color = (0, 200, 0)
c_line_s = 2

c_center_offset = [0, 0]
c_image_center = lambda w, h: (w // 2 + c_center_offset[0], h // 2 + c_center_offset[1])

c_crop_rect = [(0, 0), (1280, 720)]


class QuitException(Exception):
    pass


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


def line_angle(pt1, pt2):
    delta_x = pt2[0] - pt1[0]
    delta_y = pt2[1] - pt1[1]
    return math.atan2(delta_y, delta_x)


def summarize_lines(lines, image_center):
    aa = []

    for lst in lines:
        inc, (x1, y1, x2, y2) = lst[0], lst[1][0]

        if inc:
            pt1 = (x1, y1)
            pt2 = (x2, y2)

            pt0 = image_center

            aa += [line_angle(pt0, pt1), line_angle(pt0, pt2)]

    theta = None
    if aa:
        theta = mean_angles(aa)

    return theta


def black_arrow_mask(image):
    mask = np.zeros(image.shape, dtype=image.dtype)

    cv2.rectangle(mask, c_crop_rect[0], c_crop_rect[1], (255, 255, 255), -1)

    return mask


def seg_func(image):
    mask = black_arrow_mask(image)
    image = cv2.bitwise_and(image, mask)

    if False:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        sat = hsv[:, :, 1] < 80
        val = hsv[:, :, 2] < 180
        seg = sat * val * mask[:, :, 0]
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # m = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        # rv, thres = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        # rv, thres = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        # rv, thres = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE + cv2.THRESH_BINARY_INV)
        # print('threshold_value', rv)

        # thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 5)
        # thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 5)

        gray = cv2.bitwise_not(gray)
        tt = 170
        gray[gray > tt] = 255
        gray[gray <= tt] = 0
        seg = cv2.bitwise_and(gray, mask[:, :, 0])
    #        seg = thres

    # seg = cv2.bitwise_and(gray, thres)

    return image, seg


all1 = dict([])
all2 = []

f_perform_filter = True
seg__ = None


def find_holes(image):  # , image_center, seg_func, hough_threshold, hough_min_line_length, hough_max_line_gap, ll):
    image, seg0 = seg_func(image)

    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 250
    params.maxArea = 25000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.7

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(cv2.bitwise_not(seg0))
    image = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    t = time.time()
    global all1
    for kp in keypoints:
        ipt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
        isz = int(round(kp.size/2))

        tup = (ipt[0], ipt[1], isz)
        if tup not in all1:
            all1[tup] = [0, t, 0.]

        mask = np.zeros_like(seg0)
        cv2.circle(mask, ipt, isz, 255, -1)
        seg_ = cv2.bitwise_and(seg0, mask)
        cc = cv2.countNonZero(seg_)
        cov = cc / (isz ** 2 * math.pi)

        all1[tup] = [all1[tup][0] + 1, t, cov]

        # remove_old_circles()
        for k in list(all1.keys()):
            v = all1[k]
            if t - v[1] > .5:
                del all1[k]
            # elif v[2] < .80:
            #     del all1[k]

    def draw_circles(img, keypoints):
        if keypoints:
            for kp in keypoints:
                pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
                sz = int(round(kp.size/2))
                cv2.circle(img, pt, sz, (0, 255, 0), 2)
                cv2.circle(img, pt, 2, (0, 0, 255), 3)

                x = pt[0] + sz
                y = pt[1] + sz
                #cv2.putText(img, '{:d} {:.1f} {:.2f}'.format(v[0], t - v[1], v[2]), (x, y + 30 * 1), c_label_font, c_label_s, c_label_color)
                cv2.putText(img, '{:.2f}'.format(kp.size), (x, y + 30 * 1), c_label_font, c_label_s, c_label_color)

    def draw_circles2(img, keypoints):
            for pt_sz,v in keypoints.items():
                pt, sz = pt_sz[:2], pt_sz[-1]
                cnt, t0, cov = v
                cv2.circle(img, pt, int(round(sz)), (0, 255, 0), 2)
                cv2.circle(img, pt, 2, (0, 0, 255), 3)

                x = pt[0] + int(round(sz))
                y = pt[1] + int(round(sz))
                print(x, y)
                cv2.putText(img, '{:d} {:.1f} {:.2f} {:.2f}'.format(cnt, t - t0, cov, sz), (x, y + 30 * 1), c_label_font, c_label_s, c_label_color)
                #cv2.putText(img, '{:.2f}'.format(kp.size), (x, y + 30 * 1), c_label_font, c_label_s, c_label_color)

    #draw_circles(seg0, keypoints)
    seg0 = cv2.cvtColor(seg0, cv2.cv2.COLOR_GRAY2BGR)
    draw_circles2(seg0, all1)

    return None, image, seg0, None


def find_holes(image):  # , image_center, seg_func, hough_threshold, hough_min_line_length, hough_max_line_gap, ll):
    image, seg0 = seg_func(image)

    circles = cv2.HoughCircles(seg0, cv2.HOUGH_GRADIENT, .5, 40,
                               param1=40, param2=10, minRadius=2, maxRadius=60)

    t = time.time()

    global all1, seg__
    if circles is not None:
        # combine similar circles

        # One ideas was to paint the circles to a separate grayscale accumulation mask. 
        # With each iteration lower the intensity of all pixels and add the pixels from 
        # the b/w segmented image, masked with the detected circle and scaled down. Roughly,
        # this creates a map of the most recently detected areas decaying as time advances.
        # if seg__ is None:
        #     seg__ = np.zeros(seg0.shape, dtype=np.int32)

        # seg__ -= 2
        # np.clip(seg__, 0, 255, out=seg__)

        # mask = np.zeros_like(seg0)

        # for i in circles[0, :]:
        #     cv2.circle(mask, (i[0], i[1]), i[2], 255, -1)
        # cv2.imwrite('b.pgm', mask)
        # seg_ = cv2.bitwise_and(seg0, mask)
        # print((seg_ / 64).astype(np.int32).max())
        # seg__ += (seg_ / 64).astype(np.int32)

        # np.clip(seg__, 0, 255, out=seg__)
        # cv2.imwrite('a.pgm', seg__.astype(np.uint8))

        # lst2 = []
        # print(all2b.shape, len(all2))


        # global all2
        # for i, c1 in enumerate(circles[0, :]):
        #     min_dd = 100000000
        #     min_j = -1
        #     for j, c2 in enumerate(all2):
        #         dx = c1[0] - c2[0]
        #         dy = c1[1] - c2[1]
        #         dr = c1[2] - c2[2]
        #         dd = dx ** 2 + dy ** 2
        #         if min_dd > dd:
        #             min_dd = dd
        #             min_j = j
        #     if math.sqrt(min_dd) < 5:
        #         print(i, min_j, math.sqrt(min_dd), dr)
        #     else:
        #         all2 += [c1]

        # print('aa', len(circles[0]), len(all2))

        # global all2
        # all2 += [circles]
        # all2 = all2[-5:]
        # all2b = np.vstack([x[0] for x in all2])

        # One way to combine a timeseries of circles is to paint them to a new image
        # as filled circles and then repeat HoughCircles. The circles could be filtered
        # before painting to remove those that are too small or have poor ratio of white-black
        # pixels contained by their area. Filter is probably required because otherwise
        # a circle with a low ratio will be replaced with one that's solid and with
        # 100% coverage. This method is nice because it's fairly linear.

        # Another way to combine circles is to cluster the centers, but how does this
        # consider the similarity of radii? Which clustering algorithm ot use?
        # E.g., if k-means, how is k chosen?

        # d = np.zeros((all2b.shape[0], all2b.shape[0]), np.float)
        # for i in range(all2b.shape[0]-1):
        #     for j in range(i+1, all2b.shape[0]):
        #         # print(all2b[i], all2b[j])
        #         dd = (all2b[i][0] - all2b[j][0]) ** 2 + (all2b[i][1] - all2b[j][1]) ** 2
        #         d[i][j] = dd
        #         d[j][i] = dd
        #
        # print(d)

        # Another method to combine similar circles is to perform connected component analysis.
        # This is possible because we know the points are associated with circles, the radius
        # of the circles, and can calculate overlap of circles. The overlap testing function
        # would, for each pair of circles, consider distance between centers and the relative
        # radii to calculate coverage. But maybe a very small circle contained in a large circle
        # would be considered separate.

        # To compute the distance between all pairs of points, this seems O(n^2), but we can
        # potentially improve this by first sorting the the points, first by x, then by y.
        # Then scan the points using a sliding window the length of one of dimensions of the
        # the image, and width equal to a cutoff distance. Pairs of points that are more
        # distance in the sliding dimension than the cutoff are ignored. Within the sliding
        # window there may be pairs of points that are greater than the cutoff. As the sliding
        # window progresses, only the values that are entered the sliding window need to be
        # tested against values remaining in the sliding window, and with the points sorted by
        # first then second dimension, the the list can be further reduce by approximating Eucledian
        # distance with Manhatten distance. This still has
        # O(n^2) worst case complexity, but generally will perform much better. This idea could be
        # extended to scan in the second dimension within the larger sliding window.

        # d = np.zeros((all2b.shape[0], all2b.shape[0]), np.float)
        # ll = circles[0].tolist()
        # ll = sorted(ll)
        # ww = 20
        # ll2 = []
        # nl = [(ii, x) for ii, x in enumerate(ll) if 0 <= x[0] < ww]
        # for ss in range(seg0.shape[1] - ww):
        #     if ss > 0:
        #         ll2 = [x for x in ll2 if ss < x[1][0]]
        #         nl = [(ii, x) for ii, x in enumerate(ll) if ss + ww - 1 - 1 <= x[0] < ss + ww - 1]
        #
        #     # print(ss, ss + ww - 1, len(ll2), len(ll), len(nl))
        #
        #     for ii in range(len(nl)):
        #         for jj in range(ii + 1, len(nl)):
        #             i, iii = nl[ii]
        #             j, jjj = nl[jj]
        #             # print(all2b[i], all2b[j])
        #             dd = (all2b[i][0] - all2b[j][0]) ** 2 + (all2b[i][1] - all2b[j][1]) ** 2
        #             d[i][j] = dd
        #             d[j][i] = dd
        #
        #     for ii in range(len(nl)):
        #         for jj in range(len(ll2)):
        #             i, iii = nl[ii]
        #             j, jjj = ll2[jj]
        #             # print(all2b[i], all2b[j])
        #             dd = (all2b[i][0] - all2b[j][0]) ** 2 + (all2b[i][1] - all2b[j][1]) ** 2
        #             d[i][j] = dd
        #             d[j][i] = dd
        #
        #     ll2 += nl
        #
        # print(d)

        # mask = np.zeros_like(seg0)
        # for i in all2b:
        #     cv2.circle(mask, (i[0], i[1]), i[2], 255, -1)
        # cv2.imwrite('aa.png', mask)

        # lst2 = []
        # print(all2b.shape, len(all2))

        # summarize_circles()
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            tup = tuple(i)
            if tup not in all1:
                all1[tup] = [0, t]

            mask = np.zeros_like(seg0)
            cv2.circle(mask, (i[0], i[1]), i[2], 255, -1)
            seg_ = cv2.bitwise_and(seg0, mask)
            # cv2.imwrite('a.png', seg0)
            # cv2.imwrite('b.png', mask)
            # cv2.imwrite('c.png', seg_)
            # sys.exit(1)
            cc = cv2.countNonZero(seg_)
            # cc2 = seg_[seg_ > 0]
            r = i[2]
            cov = cc / (r ** 2 * math.pi)
            # print(cc, cc2.shape, cov)

            all1[tup] = [all1[tup][0] + 1, t, cov]

        # remove_old_circles()
        for k in list(all1.keys()):
            v = all1[k]
            if t - v[1] > .5:
                del all1[k]
            # elif v[2] < .80:
            #     del all1[k]

    # print(len(all1))

    seg0 = cv2.cvtColor(seg0, cv2.COLOR_GRAY2BGR)

    def draw_circles(img, circles):
        if circles is not None and len(circles) > 0:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    def draw_circles2(img, circles, t):
        if circles is not None and len(circles) > 0:
            for k, v in circles[0].items():
                i = np.around(k)
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

                x = i[0] + i[2]
                y = i[1] + i[2]
                cv2.putText(img, '{:d} {:.1f} {:.2f}'.format(v[0], t - v[1], v[2]), (x, y + 30 * 1), c_label_font, c_label_s, c_label_color)

    # filter_circles
    global f_perform_filter
    # print(f_perform_filter)
    if f_perform_filter:
        circles = [{k: v for k, v in all1.items() if v[2] > .9}]
        # print(circles)

        draw_circles2(seg0, circles, t)
        draw_circles2(image, circles, t)
    else:
        draw_circles(seg0, circles)
        draw_circles(image, circles)

    return None, image, seg0, None


def draw_labels(image, image_b, image_r, theta_b, theta_r, mm_b, mm_r, mm_final):
    # cv2.putText(image_b, f'{theta_b:5.2f} rad {mm_b:6.3f} mm', (20, 30 * 1), c_label_font, c_label_s, c_label_color)
    # cv2.putText(image_r, f'{theta_r:5.2f} rad {mm_r:6.3f} mm', (20, 30 * 1), c_label_font, c_label_s, c_label_color)
    # cv2.putText(image, f'{mm_final:6.3f} mm', (20, 30 * 1), c_label_font, c_label_s, c_label_color)
    cv2.putText(image_b, '{:5.2f} rad {:6.3f} mm'.format(theta_b, mm_b), (20, 30 * 1), c_label_font, c_label_s, c_label_color)
    cv2.putText(image_r, '{:5.2f} rad {:6.3f} mm'.format(theta_b, mm_b), (20, 30 * 1), c_label_font, c_label_s, c_label_color)
    cv2.putText(image, '{:6.3f} mm'.format(mm_final), (20, 30 * 1), c_label_font, c_label_s, c_label_color)


@static_vars(fps_lst=[], fps_t1=None)
def draw_fps(image):
    if draw_fps.fps_t1 is None:
        draw_fps.fps_t1 = time.time()
        return
    t2 = time.time()
    append_v(draw_fps.fps_lst, 1. / (t2 - draw_fps.fps_t1), 90)
    draw_fps.fps_t1 = t2

    fps = np.mean(draw_fps.fps_lst)

    # cv2.putText(image, f'{fps:.2f} fps', (20, 30 * 2), c_label_font, c_label_s, c_label_color)
    cv2.putText(image, '{:.2f} fps'.format(fps), (20, 30 * 2), c_label_font, c_label_s, c_label_color)


error_str = None


def display_error(s):
    global error_str
    error_str = s


@static_vars(ind=0)
def next_frame(video_capture, debug=True):
    if not debug:
        retval, image0 = video_capture.read()
    else:
        for _ in range(2):
            fn = 'tests/downward_facing_camera/1280x720/mov_raw_{:06d}.ppm'.format(next_frame.ind)
            if os.path.exists(fn):
                break
            next_frame.ind = 0

        retval, image0 = 1, cv2.imread(fn, -1)
        next_frame.ind += 1

    if not retval:
        print('rv is false')
        sys.exit(1)
    if image0.size == 0:
        print('image0 is empty')
        sys.exit(1)

    image0 = cv2.rotate(image0, cv2.ROTATE_180)

    return image0


c_view = 4
image_center = None


@static_vars(theta_b_l=[], theta_r_l=[], pause_updates=False, record=False, record_ind=0, mouse_op='alignment')
def get_measurement(video_capture):
    image0 = next_frame(video_capture)

    h, w = image0.shape[:2]
    global image_center
    if image_center is None:
        image_center = (w // 2, h // 2)

    global c_initial_image_rot
    m = cv2.getRotationMatrix2D(image_center, c_initial_image_rot / math.pi * 180., 1.0)
    image1 = cv2.warpAffine(image0, m, (w, h))
    image2 = image1.copy()

    theta_b, image_b, seg_b, skel_b = find_holes(image1)
    skel_b = seg_b

    global c_crop_rect
    cv2.rectangle(image1, c_crop_rect[0], c_crop_rect[1], c_line_color, c_line_s)

    # Draw final marked up image
    mask = np.zeros(image2.shape, dtype=image2.dtype)
    # cv2.circle(mask, image_center, c_dial_outer_mask_r, (255, 255, 255), -1)
    image2 = cv2.bitwise_and(image2, mask)

    # Draw calculated red and black arrows
    # if get_measurement.theta_b_l and get_measurement.theta_r_l:
    #     plot_lines(None, theta_b, c_black_drawn_line_length, image2, image_center)
    #        plot_lines(None, theta_r, c_red_drawn_line_length, image2, image_center)

    #        draw_labels(image2, image_b, image_r, theta_b, theta_r, mm_b, mm_r, mm_final)

    global c_view

    # Build and display composite image
    final_img = None
    if c_view == 0:
        img_all0 = np.vstack([image0, image1, image2])
        img_all1 = np.vstack([seg_b, skel_b, image_b])
        img_all = np.hstack([img_all0, img_all1])
        if error_str:
            print(error_str)
            c_label_font_error = cv2.FONT_HERSHEY_SIMPLEX
            c_label_color_error = (0, 0, 255)
            c_label_s_error = 1.5
            cv2.putText(img_all, 'WARNING: ' + error_str, (200, img_all.shape[0] // 2 - 20), c_label_font_error, c_label_s_error, c_label_color_error, 3)
        img_all_resized = cv2.resize(img_all, None, fx=c_final_image_scale_factor, fy=c_final_image_scale_factor)
        final_img = img_all_resized

        if get_measurement.record:
            fn1 = 'mov_raw_{:06}.ppm'.format(get_measurement.record_ind)
            cv2.imwrite(fn1, image0)
            fn2 = 'mov_all_{:06}.ppm'.format(get_measurement.record_ind)
            cv2.imwrite(fn2, img_all)
            fn3 = 'mov_fin_{:06}.ppm'.format(get_measurement.record_ind)
            cv2.imwrite(fn3, image2)
            get_measurement.record_ind += 1
            print('Recorded {} {}'.format(fn1, fn2))
    elif c_view == 1:
        final_img = image0
    elif c_view == 2:
        final_img = image1
    elif c_view == 3:
        final_img = skel_b
    elif c_view == 4:
        final_img = image_b

    if not get_measurement.pause_updates:
        global mouse_pts, mouse_moving
        if len(mouse_pts) == 2 and mouse_pts[1] is not None:
            pt1 = tuple([int(round(x / c_final_image_scale_factor)) for x in mouse_pts[0]])
            pt2 = tuple([int(round(x / c_final_image_scale_factor)) for x in mouse_pts[1]])

            if get_measurement.mouse_op == 'alignment' and c_view == 1:
                cv2.line(final_img, pt1, pt2, (255, 0, 255), thickness=3)
                if not mouse_moving:
                    c_initial_image_rot = line_angle(*mouse_pts)
                    image_center = (mouse_pts[0][0], mouse_pts[0][1])
                    mouse_pts = []
                    c_view = 2
            elif get_measurement.mouse_op == 'crop' and c_view == 2:
                cv2.rectangle(final_img, pt1, pt2, (255, 0, 255), thickness=3)
                if not mouse_moving:
                    c_crop_rect = [pt1, pt2]
                    mouse_pts = []

        if c_view not in [1, 2]:
            mouse_pts = []
            mouse_moving = False

    draw_fps(final_img)

    if not get_measurement.pause_updates:
        cv2.imshow("Live", final_img)
    key = cv2.waitKey(5) & 0xff
    if key == ord('p'):
        get_measurement.pause_updates = not get_measurement.pause_updates
    elif key == ord('r'):
        get_measurement.record = not get_measurement.record
    elif key == ord('s'):
        for i in range(100):
            # fn1 = f'raw_{i:03}.png'
            fn1 = 'raw_{:03}.png'.format(i)
            if not os.path.exists(fn1):
                cv2.imwrite(fn1, image0)
                # fn2 = f'all_{i:03}.png'
                fn2 = 'all_{:03}.png'.format(i)
                cv2.imwrite(fn2, img_all)
                # print(f'Wrote images {fn1} and {fn2}')
                print('Wrote images {} and {}'.format(fn1, fn2))
                break
    elif key == ord('a'):
        get_measurement.mouse_op = 'alignment'
        c_view = 1
    elif key == ord('c'):
        get_measurement.mouse_op = 'crop'
        c_view = 2
    elif key == ord('f'):
        global f_perform_filter
        f_perform_filter = not f_perform_filter
    elif key == ord('0'):
        c_view = 0
    elif key == ord('1'):
        c_view = 1
    elif key == ord('2'):
        c_view = 2
    elif key == ord('3'):
        c_view = 3
    elif key == ord('4'):
        c_view = 4
    elif key == ord('q'):
        raise QuitException
    elif key >= 0:
        # print(key)
        pass

    return None, key


mouse_pts = []
mouse_moving = False


def click_and_crop(event, x, y, flags, param):
    global mouse_pts, mouse_moving

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pts = [(x, y), None]

    elif event == cv2.EVENT_MOUSEMOVE:
        if len(mouse_pts) == 2:
            mouse_pts[1] = (x, y)
            mouse_moving = True

    elif event == cv2.EVENT_LBUTTONUP:
        if len(mouse_pts) == 2:
            mouse_pts[1] = (x, y)
            mouse_moving = False


def gauge_vision_setup():
    np.set_printoptions(precision=2)

    video_capture = cv2.VideoCapture(1)
    if not video_capture.isOpened():
        print('camera is not open')
        sys.exit(1)

    set_camera_properties(video_capture)
    # list_camera_properties(video_capture)

    return video_capture


def main():
    # video_capture = gauge_vision_setup()
    video_capture = None

    cv2.namedWindow("Live")
    cv2.setMouseCallback("Live", click_and_crop)

    while True:
        try:
            mm_final, key = get_measurement(video_capture)
            # print('mm_final:', mm_final)
        except QuitException:
            break


if __name__ == "__main__":
    main()
