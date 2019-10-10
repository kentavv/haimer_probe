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

# TODO: A nicer interface would be to specify the center and radius of the circle being probed and the initial probe position.
#  Current interface is the initial probe position and the limits of the probing in each direction.
#  This was done to expedite the needed example but make a less useful simulation.

from __future__ import print_function

import math

import cv2
import numpy as np


def euc_dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


inc = 0
record = True


def update_screen(img, delay=0):
    global inc, record
    if record:
        # Can convert to PNGs later with
        # find . -iname '*.ppm' -print0 | xargs -0 -n 1 -P 16 optipng -o9
        fn = 'mov_{0:06d}.ppm'.format(inc)
        inc += 1
        cv2.imwrite(fn, img)
        print('Wrote', fn)

    cv2.imshow('simulation', img)
    key = cv2.waitKey(delay) & 0xff
    if key == ord('q'):
        exit(0)


def main():
    frame_dim = (1080, 1920, 3)
    frame_center = (1920 // 2, 1080 // 2)

    scale = 2
    r0 = 10 * scale  # Radius of the probe tip
    pts = [(frame_center[0], frame_center[1])] * 6  # this the point where the probe lowers

    c_unknown_color = (0, 0, 128)
    c_safe_color = (0, 160, 0)
    c_pt_color = (0, 255, 0)
    c_unassigned_color = (0, 0, 96)

    # Anti-aliased lines leave unexpected bits when used with painter's algorithm
    # line_type = cv2.LINE_AA
    line_type = 0
    scan_range = [-3000, 3000]

    step_size = 10

    img = None

    def draw_points():
        # for i in range(4, -1, -1):
        for i in range(len(pts)):
            cv2.circle(img, pts[i], r0, c_pt_color, -1, lineType=line_type)

            c_label_font = cv2.FONT_HERSHEY_SIMPLEX
            c_label_color = (0, 0, 0)
            c_label_s = .4 * scale
            thickness = 2
            s = str(i)
            text_size, baseline = cv2.getTextSize(s, c_label_font, c_label_s, thickness)
            cv2.putText(img, s, (pts[i][0] - text_size[0] // 2, pts[i][1] + text_size[1] // 2), c_label_font, c_label_s, c_label_color, thickness)

    img = np.zeros(frame_dim, dtype=np.uint8)
    img[:] = c_unknown_color
    draw_points()
    update_screen(img, 5)

    for move in ['right', 'left', 'down', 'up']:
        u = 0
        if move == 'right':
            u = 200 * scale
        elif move == 'left':
            u = 100 * scale
        elif move == 'down':
            u = 90 * scale
        elif move == 'up':
            u = 600 * scale

        for ii in range(0, u, step_size):
            img = np.zeros(frame_dim, dtype=np.uint8)
            img[:] = c_safe_color

            if move == 'right':
                pts[1] = (pts[1][0] + step_size, pts[1][1])
            elif move == 'left':
                pts[2] = (pts[2][0] - step_size, pts[1][1])
            elif move == 'down':
                pts[3] = (pts[3][0], pts[3][1] + step_size)
            elif move == 'up':
                pts[4] = (pts[4][0], pts[4][1] - step_size)

            md_pt = (pts[2][0] + (pts[1][0] - pts[2][0]) / 2, pts[2][1])
            if move in ['right', 'left']:
                pts[3] = pts[4] = pts[5] = md_pt

            # There's some error that causes some of the drawn outer points to be drawn over, especially points 1 and 2
            # while searching left and right. This is likely because the later circles use integer dimensions. Adding a
            # small fudge factor the later circle radii (r0/4 be enough) helps cosmetically, but should not be needed.
            fudge = r0 / 4 - 1
            draw_points()

            mi, mr = -1, float('inf')
            for i in range(*scan_range):
                tpt = (md_pt[0], md_pt[1] + i)

                r1 = euc_dist(pts[1], tpt)
                # assert(r1 == (r2 = euc_dist(pts[2], tpt)))
                r3 = euc_dist(pts[3], tpt)
                # r4 = euc_dist(pts[4], tpt)

                # The following is not necessary if for every loop of ii, the point being moved is moved by step_size.
                # The second check is probably not right in all cases. It's purpose is to catch when the
                # downward moving point is within the region suggested by pts[1, 2].
                # if move in ['right', 'left'] or (abs(pts[1][1] - pts[3][1]) <= r0):

                if move in ['right', 'left']:
                    # Use painters algorithm to fill in unsafe areas on a canvas that's initially all safe.
                    # A thickness valud > 1 is likely needed to fill in gaps between circles.
                    cv2.circle(img, tpt, int(round(r1 + r0 + fudge)), c_unknown_color, 2, lineType=line_type)
                elif move in ['down', 'up']:
                    dr = abs(r1 - r3)
                    if mr > dr:
                        mr = dr
                        mi = i

            top_pt_found = False
            if move in ['down', 'up']:
                tpt = (md_pt[0], md_pt[1] + mi)

                if move == 'down':
                    pts[5] = tpt

                r1 = euc_dist(pts[1], tpt)
                r4 = euc_dist(pts[4], tpt)

                # top_pt_found = r1 - (r4 + r0) < 0.
                top_pt_found = r1 - r4 <= 0.
                if top_pt_found:
                    err = r1 - r4
                    if abs(err) > 0.:
                        print('Final overshoot:', err)

                cv2.circle(img, tpt, int(round(r1 + r0)), c_unknown_color, 2, lineType=line_type)

            c = c_unassigned_color if move in ['right', 'left'] else c_unknown_color
            cv2.floodFill(img, None, (0, pts[0][1]), c)
            cv2.floodFill(img, None, (img.shape[1] - 1, pts[0][1]), c)

            update_screen(img, 5)
            # if move in ['right', 'left']:
            #     update_screen(5)
            # else:
            #     update_screen(0)

            if move == 'up':
                # print(ii, mi, mr)
                if top_pt_found:
                    break

    update_screen(img, 0)


if __name__ == "__main__":
    main()
