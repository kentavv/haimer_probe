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


import math

import cv2
import numpy as np


def euc_dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def main():
    frame_dim = (1080, 1920, 3)
    r0 = 10
    frame_center = (1920 // 2, 1080 // 2)

    pt0 = (frame_center[0], frame_center[1] + 100)  # this the point where the probe lowers
    pt1 = pt2 = pt3 = pt4 = pt0

    c_unknown_color = (0, 0, 127)
    c_safe_color = (0, 200, 0)
    c_pt_color = (0, 255, 0)
    c_unassigned_color = (0, 0, 95)

    # line_type = cv2.LINE_AA
    line_type = 0
    scan_range = [-3000, 3000]

    step_size = 10

    for move in ['right', 'left', 'down', 'up']:
        u = 0
        if move == 'right':
            u = 100
        elif move == 'left':
            u = 100
        elif move == 'down':
            pt3 = (md_pt[0], md_pt[1])
            u = 50
        elif move == 'up':
            pt4 = (md_pt[0], md_pt[1])
            u = 600

        for ii in range(0, u, step_size):
            img = np.zeros(frame_dim, dtype=np.uint8)
            img[:] = c_safe_color

            md_pt = (pt2[0] + (pt1[0] - pt2[0]) / 2, pt2[1])
            mi, mr = -1, float('inf')
            cb = False

            if move == 'right':
                pt1 = (pt1[0] + step_size, pt1[1])
            elif move == 'left':
                pt2 = (pt2[0] - step_size, pt1[1])
            elif move == 'down':
                pt3 = (pt3[0], pt3[1] + step_size)
            elif move == 'up':
                pt4 = (pt4[0], pt4[1] - step_size)

            for pt in [pt1, pt2, pt3, pt4]:
                cv2.circle(img, pt, r0, c_pt_color, -1, lineType=line_type)

            for i in range(*scan_range):
                tpt = (md_pt[0], md_pt[1] + i)

                r2 = euc_dist(pt2, tpt)
                if move in ['right', 'left']:
                    cv2.circle(img, tpt, int(round(r2 + r0)), c_unknown_color, 2, lineType=line_type)
                elif move == 'down':
                    r3 = euc_dist(pt3, tpt)
                    if abs(r2 - r3) < 2.:
                        print(ii, r2, r3)
                        cv2.circle(img, tpt, int(round(r2 + r0)), c_unknown_color, 2, lineType=line_type)
                elif move == 'up':
                    cv2.circle(img, pt4, r0, c_pt_color, -1, lineType=line_type)

                    r3 = euc_dist(pt3, tpt)
                    r4 = euc_dist(pt4, tpt)
                    # if abs(r2 - r3) < 2 and abs(r2 - r4) < 2.:
                    if abs(r2 - r3) < 0.1:  # and abs(r2 - r4) < 2.:
                        if mr > abs(r2 - r4):
                            mr = abs(r2 - r4)
                            mi = i
                        print(ii, r2, r3, r4)
                        cv2.circle(img, tpt, int(round(r2 + r0)), c_unknown_color, 2, lineType=line_type)
                        cb = cb or r2 - (r4 + r0) < 0.

                # cv2.imshow('test', img)
                # cv2.waitKey(5)

            cv2.floodFill(img, None, (0, 1080 // 2), c_unassigned_color)
            cv2.floodFill(img, None, (1920 - 1, 1080 // 2), c_unassigned_color)

            cv2.imshow('test', img)
            cv2.waitKey(5)

            if move == 'up':
                print(ii, mi, mr)
                if cb:
                    break

    cv2.imshow('test', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
