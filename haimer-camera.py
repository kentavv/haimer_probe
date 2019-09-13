#!/usr/bin/env python3

# Cpyright Kent A. Vander Velden, 2019
# kent.vandervelden@gmail.com

import math
import sys
import time

import cv2
import numpy as np

c_dial_outer_mask_r = 220

c_red_angle_start = 1.8780948507158541
c_red_angle_end = 4.387011637081153
c_initial_image_rot = -0.062469504755442426

c_rho_resolution = 1 / 2  # 1/2 pixel
c_theta_resolution = np.pi / 180 / 4  # 1/4 degree

c_black_outer_mask_r = 130
c_black_outer_mask_e = (125, 135)
c_inner_mask_r = 20
c_red_outer_mask_r = 88

# minLinLength for black arrow needs to be larger than the height of the HAIMER label
c_black_hough_threshold = 5
c_black_hough_min_line_length = 22
c_black_hough_max_line_gap = 2
c_black_ll = 300
c_red_hough_threshold = 5
c_red_hough_min_line_length = 10
c_red_hough_max_line_gap = 2
c_red_ll = 200


def c_image_center(w, h):
    return (w // 2 + 25 - 3 + 1 - 5 - 4, h // 2 + 2 - 1 - 5)


# Interesting challenges
# 1) The long black pointer passes over the top of the short red pointer.
# 2) The blue dot created by the LED on the camera changes the hue of the pointer tha
#    passes under it and then some amount of the pointer is lost.

def mean_angles(lst):
    # Because the list of angles can contain both 0 and 2pi,
    # however, 0 and pi are also contained and will average to pi/2,
    # this is thus probably not the best way to do this.
    # https://en.wikipedia.org/wiki/Mean_of_circular_quantities
    return math.atan2(np.mean(np.sin(lst)), np.mean(np.cos(lst)))


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

    for nm, v in capture_properties:
        print(nm, video_cap.set(eval(nm), v))


def find_skeleton3_(img):
    from skimage.morphology import skeletonize
    return (skeletonize(img // 255) * 255).astype(np.uint8), 1
    # return (medial_axis(img//255) * 255).astype(np.uint8), 1
    # img = skeletonize(img, method='zhang')
    # img = skeletonize(img, method='lee')


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


def filter_lines2(lines, image_center, cutoff=5):
    lines2 = []
    for lst in lines:
        x1, y1, x2, y2 = lst[0]
        x0, y0 = image_center

        # Distance between a point (image center) and a line defined by two
        # points, as returned from HoughLinesP
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        d = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        if d < cutoff:
            lines2 += [lst]

    return lines2


def plot_lines2(lines, theta, ll, image, image_center):
    if lines is not None:
        for i in range(0, len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            pt1 = (x1, y1)
            pt2 = (x2, y2)
            pt0 = image_center

            cv2.line(image, pt0, pt1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.line(image, pt0, pt2, (255, 0, 0), 1, cv2.LINE_AA)

            cv2.line(image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    if theta is not None:
        # if 0 < theta < .5:
        #     abort()

        a = math.cos(theta)
        b = math.sin(theta)
        x0, y0 = image_center
        # pt1 = (int(x0 + ll * (-b)), int(y0 + ll * (a)))
        pt2 = (int(x0 - ll * (-b)), int(y0 - ll * (a)))
        cv2.line(image, image_center, pt2, (0, 255, 255), 1, cv2.LINE_AA)


def summarize2(lines, image_center):
    aa = []
    for lst in lines:
        x1, y1, x2, y2 = lst[0]
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

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    val = hsv[:, :, 2] < 180
    seg = val * mask[:, :, 0]

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

    # Instead of cv2.HoughLines use cv2.HoughLinesP, which may be faster and has
    # options for minimal line length.
    lines = cv2.HoughLinesP(skel, c_rho_resolution, c_theta_resolution, hough_threshold,
                            minLineLength=hough_min_line_length, maxLineGap=hough_max_line_gap)

    theta = None
    if lines is not None:
        lines = filter_lines2(lines, image_center)
        theta = summarize2(lines, image_center)
        plot_lines2(lines, theta, ll, image, image_center)

    return theta, image, seg0, skel


def black_arrow(image, image_center):
    return arrow_common(image, image_center, black_arrow_segment, c_black_hough_threshold, c_black_hough_min_line_length, c_black_hough_max_line_gap, c_black_ll)


def red_arrow(image, image_center):
    return arrow_common(image, image_center, red_arrow_segment, c_red_hough_threshold, c_red_hough_min_line_length, c_red_hough_max_line_gap, c_red_ll)


def draw_labels(image, theta1, theta2):
    font = cv2.FONT_HERSHEY_DUPLEX
    bb = theta1 / (math.pi * 2) * 1
    rr = (theta2 - c_red_angle_start) / (c_red_angle_end - c_red_angle_start) * 4 - 2

    cc = rr - bb
    if rr < 0:
        cc = 1 - bb
    else:
        cc = bb
    ee = math.modf(abs(rr))[0] - cc

    yy = [abs(math.modf(bb)[0]) * math.pi * 2,
          abs(math.modf(rr)[0]) * math.pi * 2]
    theta_yy = mean_angles(yy)
    yy = theta_yy / (math.pi * 2)

    ccc = math.modf(rr)[1] + math.copysign(yy, rr)

    ttt = abs(math.modf(rr)[0])
    td1 = abs(ttt - bb)
    td2 = abs((1 - bb) + ttt)
    print(f'aa {bb:8.4f} {rr:8.4f} {cc:8.4f} {ee:8.4f} {yy:8.4f} {ccc:8.4f} {td1:8.4f} {td2:8.4f}')

    # if ee > .50:
    #     bb = abs(bb - 1)
    #
    #     cc = rr - bb
    #     if rr < 0:
    #         cc = 1 - bb
    #     else:
    #         cc = bb
    #     ee = math.modf(abs(rr))[0] - cc
    #
    # print(f'bb {bb:8.4f} {rr:8.4f} {cc:8.4f} {ee:8.4f}')

    cv2.putText(image, f'b {theta1:.2f} {bb:.2f}', (20, 30), font, 1, (255, 255, 255))
    cv2.putText(image, f'r {theta2:.2f} {rr:.2f}', (20, 60), font, 1, (255, 255, 255))


fps_lst = []
fps_t1 = None


def draw_fps(image):
    global fps_t1, fps_lst

    if fps_t1 is None:
        fps_t1 = time.time()
        return
    t2 = time.time()
    if len(fps_lst) > 90:
        fps_lst = fps_lst[1:]
    fps_lst = [1. / (t2 - fps_t1)] + fps_lst
    fps_t1 = t2

    fps = np.mean(fps_lst)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, f'fps {fps:.2f}', (20, 90), font, 1, (255, 255, 255))


def main():
    np.set_printoptions(precision=2)

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print('camera is not open')
        sys.exit(1)

    # list_camera_properties(video_capture)
    # set_camera_properties(video_capture)

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

        image = cimg

        h, w = image.shape[:2]
        image_center = c_image_center(w, h)

        m = cv2.getRotationMatrix2D(image_center, c_initial_image_rot / math.pi * 180., 1.0)
        image = cv2.warpAffine(image, m, (w, h))

        # image = cv2.CLAHE.apply(image)

        theta1, image1, seg1, skel1 = black_arrow(image, image_center)
        seg1 = cv2.cvtColor(seg1, cv2.COLOR_GRAY2BGR)
        skel1 = cv2.cvtColor(skel1, cv2.COLOR_GRAY2BGR)

        theta2, image2, seg2, skel2 = red_arrow(image, image_center)
        seg2 = cv2.cvtColor(seg2, cv2.COLOR_GRAY2BGR)
        skel2 = cv2.cvtColor(skel2, cv2.COLOR_GRAY2BGR)

        # Maintain a list of thetas to average to reduce noise and to fill in
        # during times where not measurements are available such as when the
        # black hand passes over the red hand
        # print(theta1, theta2)
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
        cv2.circle(image0, (image_center), c_inner_mask_r // 2, (0, 0, 255), 1)
        cv2.line(image0,
                 (image_center[0] - c_inner_mask_r, image_center[1] - c_inner_mask_r),
                 (image_center[0] + c_inner_mask_r, image_center[1] + c_inner_mask_r),
                 (0, 0, 255), 1)
        cv2.line(image0,
                 (image_center[0] - c_inner_mask_r, image_center[1] + c_inner_mask_r),
                 (image_center[0] + c_inner_mask_r, image_center[1] - c_inner_mask_r),
                 (0, 0, 255), 1)

        # black arrow
        cv2.circle(image0, image_center, c_black_outer_mask_r, (0, 255, 255), 1)
        cv2.ellipse(image0, image_center, c_black_outer_mask_e, 0, 0, 360, (0, 255, 255), 1)
        cv2.circle(image0, image_center, c_inner_mask_r, (0, 255, 255), 1)

        # red arrow
        cv2.circle(image0, image_center, c_red_outer_mask_r, (0, 255, 255), 1)
        cv2.circle(image0, image_center, c_inner_mask_r, (0, 255, 255), 1)

        mask = np.zeros(image.shape, dtype=image.dtype)
        cv2.circle(mask, image_center, c_dial_outer_mask_r, (255, 255, 255), -1)
        image = cv2.bitwise_and(image, mask)

        if theta1_l and theta2_l:
            theta1m = mean_angles(theta1_l)
            if theta1m < 0:
                theta1m += math.pi * 2
            theta1 = theta1m

            theta2m = mean_angles(theta2_l)
            if theta2m < 0:
                theta2m += math.pi * 2
            theta2 = theta2m

            plot_lines2(None, theta1, c_red_ll, image, image_center)
            plot_lines2(None, theta2, c_black_ll, image, image_center)

            draw_labels(image, theta1, theta2)
            draw_fps(image)

        img_all0 = np.vstack([cimg, image0, image])
        img_all1 = np.vstack([seg1, skel1, image1])
        img_all2 = np.vstack([seg2, skel2, image2])

        img_all = np.hstack([img_all0, img_all1, img_all2])
        # img_all = img_all1
        s = 1.
        img_all = cv2.resize(img_all, None, fx=s, fy=s)

        cv2.imshow("Live", img_all)
        if cv2.waitKey(5) >= 0:
            break


if __name__ == "__main__":
    main()
