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
# 1) How should non-circular holes be supported? Including rectangular,
#    ellipse, and slots.

import math
import os
import sys
import time

import cv2
import numpy as np

c_camera_name = 'Z-Camera'

c_final_image_scale_factor = 1

c_label_font = cv2.FONT_HERSHEY_SIMPLEX
c_label_color = (63, 255, 63)
c_label_s = .8

c_line_color = (0, 200, 0)
c_path_color = (200, 200, 64)
c_line_s = 2

c_crop_rect = None
c_machine_rect = [[0.0, 0.0], [4.266, 3.0]]

c_demo_mode = False


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
    capture_properties = [('cv2.CAP_PROP_FRAME_WIDTH', 1280),
                          ('cv2.CAP_PROP_FRAME_HEIGHT', 720)
                          ]

    # capture_properties = [('cv2.CAP_PROP_FRAME_WIDTH', 640),
    #                       ('cv2.CAP_PROP_FRAME_HEIGHT', 480)
    #                       ]

    for nm, v in capture_properties:
        if not video_cap.set(eval(nm), v):
            print('Unable to set', nm, v)


def min_path(lst, start_pt=None, end_pt=None):
    from itertools import permutations

    m_d = float('inf')
    m_lst = []
    for l in permutations(lst):
        l2 = list(l)

        if start_pt is not None:
            l2 = [start_pt] + l2

        if end_pt is not None:
            l2 = l2 + [end_pt]

        d = 0.
        for pt1, pt2 in zip(l2, l2[1:]):
            d += (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2

        if m_d > d:
            m_d = d
            m_lst = l2

#        print(l2)
#        print(zip(l2, l2[1:]))
#        print(d)

#    print(m_d, m_lst)

    return math.sqrt(m_d), m_lst


def organize_circles(circles, start_pt, end_pt):
    try:
        lst = [tuple(list(x[1][0]) + [i]) for i, x in enumerate(circles)]
    except IndexError:
        pass
    else:
        d, lst = min_path(lst, tuple(list(start_pt) + [-1]), tuple(list(end_pt) + [-1]))
        circles = [circles[x[2]] for x in lst if x[2] >= 0]

    return circles


def line_length(pt1, pt2):
    delta_x = pt2[0] - pt1[0]
    delta_y = pt2[1] - pt1[1]
    return math.sqrt(delta_x ** 2 + delta_y ** 2)


def plate_mask(image):
    if c_crop_rect is None:
        mask = np.ones(image.shape, dtype=image.dtype) * 255
    else:
        off = 10
        pt1 = round_pt(add_pts(c_crop_rect[0], (-off, -off)))
        pt2 = round_pt(add_pts(c_crop_rect[1], (off, off)))
        mask = np.zeros(image.shape, dtype=image.dtype)
        cv2.rectangle(mask, pt1, pt2, (255, 255, 255), -1)

    return mask


def find_holes(image):
    mask = plate_mask(image)
    gray = cv2.cvtColor(cv2.bitwise_and(mask, image), cv2.COLOR_BGR2GRAY)
    image /= 2
    image[cv2.bitwise_not(mask) == 255] /= 2

    params = cv2.SimpleBlobDetector_Params()

    # For b/w image
    # params.filterByColor = True
    # params.blobColor = 255

    # Change thresholds
    params.minThreshold = 100
    params.maxThreshold = 181
    # print(params.thresholdStep)
    # params.thresholdStep = 10.

    # Filter by Area.
    params.filterByArea = True
    # print(params.minArea)
    # params.minArea = 25.0
    # print(params.maxArea)
    # params.maxArea = 5000.0
    params.minArea = 250
    params.maxArea = 25000

    # Filter by Circularity
    params.filterByCircularity = True
    # print(params.minCircularity)
    # params.minCircularity = 0.80
    params.minCircularity = 0.80
    # print(params.maxCircularity)

    # Filter by Convexity
    params.filterByConvexity = False
    # print(params.minConvexity)
    # params.minConvexity = 0.95
    # print(params.maxConvexity)

    # Filter by Inertia
    params.filterByInertia = False
    # print(params.minInertiaRatio)
    # params.minInertiaRatio = 0.10
    # print(params.maxInertiaRatio)

    # print(params.minDistBetweenBlobs)
    # params.minDistBetweenBlobs = 10.
    # print(params.minRepeatability)
    # params.minRepeatability = 2

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(gray)
    # image = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def keypoints_to_circles(keypoints):
        lst = [((kp.pt[0], kp.pt[1]), kp.size) for kp in keypoints]
        return lst

    def circles_to_machine_circles(circles):
        lst = []

        global c_crop_rect, c_machine_rect
        if circles and c_crop_rect and c_machine_rect:
            pt0 = c_crop_rect[0]
            mpt0 = c_machine_rect[0]

            w = c_crop_rect[1][0] - c_crop_rect[0][0]
            h = c_crop_rect[1][1] - c_crop_rect[0][1]
            mw = c_machine_rect[1][0] - c_machine_rect[0][0]
            mh = c_machine_rect[1][1] - c_machine_rect[0][1]

            for cir in circles:
                pt, diam = cir

                mx = mpt0[0] + ((pt[0] - pt0[0]) / float(w) * mw)
                my = mpt0[1] - ((pt[1] - pt0[1]) / float(h) * mh)
                mdiam = diam / float(w) * mw

                lst += [((mx, my), mdiam)]
        else:
            lst = [[] for cir in circles]

        return lst

    img_circles = keypoints_to_circles(keypoints)
    m_circles = circles_to_machine_circles(img_circles)
    circles = list(zip(img_circles, m_circles))

    return circles


def draw_circles(img, circles):
    for i, cir in enumerate(circles):
        ((x, y), diam), m_cir = cir

        pt = round_pt((x, y))
        sz = int(round(diam / 2))
        cv2.circle(img, pt, sz, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.circle(img, pt, 2, (0, 0, 255), 3, lineType=cv2.LINE_AA)

        label = chr(ord('A') + i)
        tpt = add_pts(pt, (sz + 1, sz + 1))

        cv2.putText(img, '{}'.format(label), tpt, c_label_font, c_label_s, c_label_color)


def draw_table(img, circles):
    def h(i):
        return 20, 130 + i * 25

    cv2.putText(img, '{} {:6s} {:6s} {:6s}'.format('ID', '  X', '  Y', 'diam'), h(-1), c_label_font, c_label_s, c_label_color)

    for i, cir in enumerate(circles):
        ((x, y), diam), m_cir = cir

        label = chr(ord('A') + i)

        if m_cir:
            (mx, my), mdiam = m_cir
            cv2.putText(img, '{} {:.3f} {:.3f} {:.3f}'.format(label, mx, my, mdiam), h(i), c_label_font, c_label_s, c_label_color)
        else:
            # a = (diam / 2) ** 2 * math.pi
            # cv2.putText(img, '{} {:.1f} {:.1f}'.format(label, diam, a), h(i), c_label_font, c_label_s, c_label_color)
            cv2.putText(img, '{}'.format(label), h(i), c_label_font, c_label_s, c_label_color)


def round_pt(pt):
    return tuple([int(round(x)) for x in pt])


def add_pts(pt1, pt2):
    return tuple([x + y for x,y in zip(pt1, pt2)])
 

def draw_path(img, circles, start_pt, end_pt):
    global c_crop_rect, c_machine_rect
    if not (circles and c_crop_rect and c_machine_rect):
        return

    pt0 = c_crop_rect[0]
    mpt0 = c_machine_rect[0]

    w = c_crop_rect[1][0] - c_crop_rect[0][0]
    h = c_crop_rect[1][1] - c_crop_rect[0][1]
    mw = c_machine_rect[1][0] - c_machine_rect[0][0]
    mh = c_machine_rect[1][1] - c_machine_rect[0][1]

    def mpt_to_pt(mpt):
        x = (mpt[0] - mpt0[0]) / mw * w + pt0[0]
        y = (mpt[1] - mpt0[1]) / mh * h + pt0[1]

        return round_pt((x, y))

    try:
        lst = [round_pt(x[0][0]) for x in circles]
        lst = [mpt_to_pt(start_pt)] + lst + [mpt_to_pt(end_pt)]
    except IndexError:
        pass
    else:
        for i in range(len(lst)-1):
            pt1 = lst[i]
            pt2 = lst[i+1]
            cv2.line(img, pt1, pt2, c_path_color, thickness=c_line_s, lineType=cv2.LINE_AA)
        

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
    cv2.putText(image, '{:.2f} fps'.format(fps), (20, image.shape[0] - 30), c_label_font, c_label_s, c_label_color)


_error_str = None


def display_error(s):
    global _error_str
    _error_str = s


@static_vars(ind=0)
def next_frame(video_capture):
    if not c_demo_mode:
        retval, image0 = video_capture.read()
    else:
        for _ in range(2):
            # fn = 'tests/z_camera/1280x720/mov_raw_{:06d}.ppm'.format(next_frame.ind)
            # fn = 'tests/z_camera/1280x720/holes.png'
            fn = 'tests/z_camera/1280x720/holes_and_slots.png'
            if os.path.exists(fn):
                break
            next_frame.ind = 0

        retval, image0 = 1, cv2.imread(fn, -1)
        next_frame.ind += 1

        image0 = cv2.rotate(image0, cv2.ROTATE_180)

    if not retval:
        print('rv is false')
        sys.exit(1)
    if image0.size == 0:
        print('image0 is empty')
        sys.exit(1)

    return image0


def draw_selected_points(img, pts, c=(255, 64, 32), t=3):
    # for i in range(len(pts)):
    #     pt1, pt2 = pts[i], pts[(i + 1) % len(pts)]
    #     cv2.line(img, pt1, pt2, c, thickness=t, lineType=cv2.LINE_AA)

    c = (0, 0, 255)
    t = 1
    off = 10

    for pt in pts[:-1]:
        cv2.line(img, (pt[0] - off, pt[1]), (pt[0] + off, pt[1]), c, thickness=t, lineType=cv2.LINE_AA)
        cv2.line(img, (pt[0], pt[1] - off), (pt[0], pt[1] + off), c, thickness=t, lineType=cv2.LINE_AA)

    if pts:
        pt = pts[-1]
        x, y = pt[0], pt[1]
        off2 = 30
        if off2 * 2 < x < img.shape[1] - off2 * 2 and off2 * 2 < y < img.shape[0] - off2 * 2:
            sub = img[y - off2 // 2:y + off2 // 2, x - off2 // 2:x + off2 // 2, :]
            enlarged = cv2.resize(sub, (off2 * 4, off2 * 4))
            img[y - off2 * 2:y + off2 * 2, x - off2 * 2:x + off2 * 2, :] = enlarged

        cv2.line(img, (pt[0] - off, pt[1]), (pt[0] + off, pt[1]), c, thickness=t, lineType=cv2.LINE_AA)
        cv2.line(img, (pt[0], pt[1] - off), (pt[0], pt[1] + off), c, thickness=t, lineType=cv2.LINE_AA)


@static_vars(pause_updates=False, save=False, record=False, record_ind=0, mouse_op='', c_view=3, warp_m=None, start_mpt=(0, 0), end_mpt=(0, 0))
def get_measurement(video_capture):
    image0 = next_frame(video_capture)

    if get_measurement.warp_m is not None:
        h, w = image0.shape[:2]
        warped = cv2.warpPerspective(image0, get_measurement.warp_m, (w, h))
        image1 = warped
    else:
        image1 = image0.copy()

    circles = find_holes(image1)
    circles = organize_circles(circles, get_measurement.start_mpt, get_measurement.end_mpt)

    image_b = image1.copy()
    draw_table(image_b, circles)
    draw_circles(image_b, circles)
    draw_path(image_b, circles, get_measurement.start_mpt, get_measurement.end_mpt) 

    global c_crop_rect
    if c_crop_rect:
        cv2.rectangle(image_b, round_pt(c_crop_rect[0]), round_pt(c_crop_rect[1]), c_line_color, c_line_s)

    global c_view

    # Build and display composite image
    final_img = None
    if get_measurement.c_view == 0:
        img_all = np.hstack([image0, image1, image_b])
        img_all_resized = cv2.resize(img_all, None, fx=c_final_image_scale_factor, fy=c_final_image_scale_factor)
        final_img = img_all_resized
    elif get_measurement.c_view == 1:
        final_img = image0
    elif get_measurement.c_view == 2:
        final_img = image1
    elif get_measurement.c_view == 3:
        final_img = image_b

    global mouse_sqr_pts_done, mouse_sqr_pts
    if not mouse_sqr_pts_done and get_measurement.mouse_op == 'alignment':
        draw_selected_points(final_img, mouse_sqr_pts)

    if _error_str:
        s = 'WARNING: ' + _error_str

        c_label_font_error = cv2.FONT_HERSHEY_SIMPLEX
        c_label_color_error = (0, 0, 255)
        c_label_s_error = 1.5

        thickness = 3
        text_size, baseline = cv2.getTextSize(s, c_label_font_error, c_label_s_error, thickness)

        text_pos = ((final_img.shape[1] - text_size[0]) // 2, (final_img.shape[0] + text_size[1]) // 2)
        cv2.putText(final_img, s, text_pos, c_label_font_error, c_label_s_error, c_label_color_error, thickness)

    if get_measurement.record:
        fn1 = 'mov_raw_z_{:06}.ppm'.format(get_measurement.record_ind)
        cv2.imwrite(fn1, image0)
        # fn2 = 'mov_all_z_{:06}.ppm'.format(get_measurement.record_ind)
        # cv2.imwrite(fn2, img_all)
        fn3 = 'mov_fin_z_{:06}.ppm'.format(get_measurement.record_ind)
        cv2.imwrite(fn3, final_img)
        get_measurement.record_ind += 1
        print('Recorded {} {}'.format(fn1, fn3))

    if get_measurement.save:
        get_measurement.save = False

        for i in range(100):
            # fn1 = f'raw_z_{i:03}.png'
            fn1 = 'raw_z_{:03}.png'.format(i)
            if not os.path.exists(fn1):
                cv2.imwrite(fn1, image0)
                # fn2 = f'all_z_{i:03}.png'
                fn2 = 'all_z_{:03}.png'.format(i)
                cv2.imwrite(fn2, final_img)
                # print(f'Wrote images {fn1} and {fn2}')
                print('Wrote images {} and {}'.format(fn1, fn2))
                break

    global in_alignment

    if not get_measurement.pause_updates:
        if get_measurement.mouse_op == 'alignment' and get_measurement.c_view == 1:
            if mouse_sqr_pts_done:
                # draw_selected_points(final_img, mouse_sqr_pts)

                rct = np.array(mouse_sqr_pts, dtype=np.float32)
                w1 = line_length(mouse_sqr_pts[0], mouse_sqr_pts[1])
                w2 = line_length(mouse_sqr_pts[2], mouse_sqr_pts[3])
                h1 = line_length(mouse_sqr_pts[0], mouse_sqr_pts[3])
                h2 = line_length(mouse_sqr_pts[1], mouse_sqr_pts[2])
                w = max(w1, w2)
                h = max(h1, h2)

                pt1 = mouse_sqr_pts[0]
                dst0 = [pt1, [pt1[0] + w, pt1[1]], [pt1[0] + w, pt1[1] + h], [pt1[0], pt1[1] + h]]
                dst = np.array(dst0, dtype=np.float32)

                get_measurement.warp_m = cv2.getPerspectiveTransform(rct, dst)

                pt1, pt2 = dst0[0], dst0[2]
                c_crop_rect = [pt1, pt2]

                mouse_sqr_pts = []
                mouse_sqr_pts_done = False

                get_measurement.c_view = 3
                get_measurement.mouse_op = ''

                in_alignment = True

        if get_measurement.c_view not in [1, 2]:
            mouse_pts = []
            mouse_moving = False

    if in_alignment:
        global ss
        ss2 = 'Enter plate dimensions (W,H): ' + process_key.plate_size_str
        cv2.putText(final_img, ss2, (20, 30), c_label_font, c_label_s, c_label_color)
    else:
        cv2.putText(final_img, 'Size (WxH): {:.3f} x {:.3f}'.format(*c_machine_rect[1]), (20, 30), c_label_font, c_label_s, c_label_color)

    draw_fps(final_img)

    if not get_measurement.pause_updates:
        cv2.imshow(c_camera_name, final_img)

    return circles


in_alignment = False


@static_vars(plate_size_str='')
def process_key(key):
    global in_alignment, c_machine_rect
    global mouse_sqr_pts, mouse_sqr_pts_done

    if in_alignment:
        if key == ord('a'):
            get_measurement.mouse_op = 'alignment'
            get_measurement.c_view = 1
            mouse_sqr_pts = []
            mouse_sqr_pts_done = False
            in_alignment = False
            process_key.plate_size_str = ''
        elif key in [ord(x) for x in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ',']]:
            process_key.plate_size_str += chr(key)
        elif key == ord('\r'):
            print('Parsing:', process_key.plate_size_str)
            in_alignment = False
            try:
                process_key.plate_size = [float(x) for x in process_key.plate_size_str.split(',')]
            except ValueError:
                pass
            else:
                if len(process_key.plate_size) == 2:
                    c_machine_rect[1] = process_key.plate_size
                    print(c_machine_rect)

            process_key.plate_size_str = ''
        elif key == 8:  # backspace
            process_key.plate_size_str = process_key.plate_size_str[:-1]
        elif key == ord('q'):
            raise QuitException
        elif key == 255:
            pass
        else:
            return False

        return True

    if key == ord('p'):
        get_measurement.pause_updates = not get_measurement.pause_updates
    elif key == ord('l'):
        get_measurement.pause_updates = not get_measurement.pause_updates
    elif key == ord('r'):
        get_measurement.record = not get_measurement.record
    elif key == ord('s'):
        get_measurement.save = True
    elif key == ord('a'):
        get_measurement.mouse_op = 'alignment'
        get_measurement.c_view = 1
        mouse_sqr_pts = []
        mouse_sqr_pts_done = False
    # elif key == ord('f'):
    #     find_holes.f_perform_filter = not find_holes.f_perform_filter
    elif key == ord('0'):
        get_measurement.c_view = 0
    elif key == ord('1'):
        get_measurement.c_view = 1
    elif key == ord('2'):
        get_measurement.c_view = 2
    elif key == ord('3'):
        get_measurement.c_view = 3
    elif key == ord('q'):
        raise QuitException
    elif key != 255:
        print(key)
        return False

    return True


mouse_pts = []
mouse_moving = False
mouse_sqr_pts = []
mouse_sqr_pts_done = False


def click_and_crop(event, x, y, flags, param):
    global mouse_pts, mouse_moving
    global mouse_sqr_pts
    global mouse_sqr_pts_done

    # x -= 5
    # y -= 5

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pts = [(x, y), None]
        mouse_sqr_pts += [(x, y)]
        if len(mouse_sqr_pts) == 1:
            mouse_sqr_pts += [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if len(mouse_sqr_pts) == 0:
            mouse_sqr_pts += [(x, y)]
        elif len(mouse_sqr_pts) == 1:
            mouse_sqr_pts[0] = (x, y)
        elif len(mouse_pts) == 2:
            mouse_pts[1] = (x, y)
            mouse_moving = True
            if mouse_sqr_pts:
                mouse_sqr_pts[-1] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if len(mouse_pts) == 2:
            mouse_pts[1] = (x, y)
            mouse_moving = False

        if not mouse_sqr_pts_done:
            mouse_sqr_pts[-1] = (x, y)

        if len(mouse_sqr_pts) > 4:
            mouse_sqr_pts = mouse_sqr_pts[:4]
            mouse_sqr_pts_done = True


def gauge_vision_setup():
    cv2.namedWindow(c_camera_name)
    cv2.setMouseCallback(c_camera_name, click_and_crop)

    if c_demo_mode:
        return None

    video_capture = cv2.VideoCapture(1)
    if not video_capture.isOpened():
        print('camera is not open')
        sys.exit(1)

    set_camera_properties(video_capture)
    # list_camera_properties(video_capture)

    return video_capture


def main():
    np.set_printoptions(precision=2)

    video_capture = gauge_vision_setup()

    while True:
        try:
            circles = get_measurement(video_capture)
            key = cv2.waitKey(5) & 0xff
            process_key(key)
            if key == ord('l'):
                for c in circles:
                    print(c)
                print
        except QuitException:
            break


if __name__ == "__main__":
    main()
