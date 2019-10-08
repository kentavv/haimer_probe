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

def main():
    frame_dim = (1080, 1920, 3)
    r0 = 10
    cpt = (1920//2, 1080//2) # this the point where the probe lowers
    pt0 = (cpt[0], cpt[1]+100)

    c_unknown_color = (0, 0, 127)
    c_safe_color = (0, 200, 0)
    c_pt_color = (0, 255, 0)
    c_unassigned_color = (0, 0, 63)

    #line_type = cv2.LINE_AA
    line_type = 0 
    scan_range = [-1000, 1000]

    if True:
     # First move to the right
     for ii in range(0, 100, 10):
        img = np.zeros(frame_dim, dtype=np.uint8)
        img[:] = c_safe_color

        pt1 = pt2 = pt0
        pt2 = (pt0[0] + ii, pt0[1])
        cv2.circle(img, pt1, r0, c_pt_color, -1, lineType=line_type)
        cv2.circle(img, pt2, r0, c_pt_color, -1, lineType=line_type)
 
        md_pt = (pt1[0]+(pt2[0]-pt1[0])/2, pt1[1])
        for i in range(*scan_range):
           tpt = (md_pt[0], md_pt[1] + i)
 
           r2 = math.sqrt((pt1[0] - tpt[0])**2 + (pt1[1] - tpt[1])**2)
           cv2.circle(img, tpt, int(round(r2+r0)), c_unknown_color, 2, lineType=line_type)

           #cv2.imshow('test', img)
           #cv2.waitKey(5)

        cv2.floodFill(img, None, (0, 1080//2), c_unassigned_color)
        cv2.floodFill(img, None, (1920-1, 1080//2), c_unassigned_color)

        cv2.imshow('test', img)
        cv2.waitKey(5) 

    if True:
     # Next move to the left
     for ii in range(0, 100, 10):
        img = np.zeros(frame_dim, dtype=np.uint8)
        img[:] = c_safe_color

        pt1 = pt0
        pt1 = (pt0[0] - ii, pt0[1])
        cv2.circle(img, pt1, r0, c_pt_color, -1, lineType=line_type)
        cv2.circle(img, pt2, r0, c_pt_color, -1, lineType=line_type)

        md_pt = (pt1[0]+(pt2[0]-pt1[0])/2, pt1[1])
        for i in range(*scan_range):
           tpt = (md_pt[0], md_pt[1] + i)

           r2 = math.sqrt((pt1[0] - tpt[0])**2 + (pt1[1] - tpt[1])**2)
           cv2.circle(img, tpt, int(round(r2+r0)), c_unknown_color, 2, lineType=line_type)

           #cv2.imshow('test', img)
           #cv2.waitKey(5)

        cv2.floodFill(img, None, (0, 1080//2), c_unassigned_color)
        cv2.floodFill(img, None, (1920-1, 1080//2), c_unassigned_color)

        cv2.imshow('test', img)
        cv2.waitKey(5)

    #pt1 = (pt0[0]-200, pt0[1]+200)
   # pt2 = (pt0[0]+200, pt0[1]+200)

    if True:
     # Next move down
     for ii in range(0, 50, 10):
        img = np.zeros(frame_dim, dtype=np.uint8)
        img[:] = c_safe_color

        md_pt = (pt1[0]+(pt2[0]-pt1[0])/2, pt1[1])
        pt3 = (md_pt[0], md_pt[1]+ii)

        cv2.circle(img, pt1, r0, c_pt_color, -1, lineType=line_type)
        cv2.circle(img, pt2, r0, c_pt_color, -1, lineType=line_type)

        for i in range(*scan_range):
           tpt = (md_pt[0], md_pt[1] + i)

           cv2.circle(img, pt3, r0, c_pt_color, -1, lineType=line_type)

           r2 = math.sqrt((pt1[0] - tpt[0])**2 + (pt1[1] - tpt[1])**2)
           r3 = math.sqrt((pt3[0] - tpt[0])**2 + (pt3[1] - tpt[1])**2)
           if abs(r2 - r3) < 2:
             print(ii, r2, r3)
             cv2.circle(img, tpt, int(round(r2+r0)), c_unknown_color, 2, lineType=line_type)

           #cv2.imshow('test', img)
           #cv2.waitKey(5)

        cv2.floodFill(img, None, (0, 1080//2), c_unassigned_color)
        cv2.floodFill(img, None, (1920-1, 1080//2), c_unassigned_color)

        cv2.imshow('test', img)
        cv2.waitKey(5)

    #md_pt = (pt1[0]+(pt2[0]-pt1[0])/2, pt1[1])
    #pt3 = (md_pt[0], md_pt[1]+100)

    if True:
      # Finally, move up
     for ii in range(0, 600, 10):
        img = np.zeros(frame_dim, dtype=np.uint8)
        img[:] = c_safe_color

        md_pt = (pt1[0]+(pt2[0]-pt1[0])/2, pt1[1])
        pt4 = (md_pt[0], md_pt[1]-ii)

        cv2.circle(img, pt1, r0, c_pt_color, -1, lineType=line_type)
        cv2.circle(img, pt2, r0, c_pt_color, -1, lineType=line_type)
        cv2.circle(img, pt3, r0, c_pt_color, -1, lineType=line_type)

        mi, mr = -1, 100000
        cb = False
        for i in range(*scan_range):
           tpt = (md_pt[0], md_pt[1] + i)

           cv2.circle(img, pt4, r0, c_pt_color, -1, lineType=line_type)

           r2 = math.sqrt((pt1[0] - tpt[0])**2 + (pt1[1] - tpt[1])**2)
           r3 = math.sqrt((pt3[0] - tpt[0])**2 + (pt3[1] - tpt[1])**2)
           r4 = math.sqrt((pt4[0] - tpt[0])**2 + (pt4[1] - tpt[1])**2)
           # if abs(r2 - r3) < 2 and abs(r2 - r4) < 2:
           if abs(r2 - r3) < 0.1: # and abs(r2 - r4) < 2:
             if mr > abs(r2 - r4):
               mr = abs(r2 - r4)
               mi = i
             print(ii, r2, r3, r4)
             cv2.circle(img, tpt, int(round(r2+r0)), c_unknown_color, 2, lineType=line_type)
             cb = cb or r2 - (r4 + r0) < 0

           #cv2.imshow('test', img)
           #cv2.waitKey(5)

        cv2.floodFill(img, None, (0, 1080//2), c_unassigned_color)
        cv2.floodFill(img, None, (1920-1, 1080//2), c_unassigned_color)

        print(ii, mi, mr)
        if cb:
           break

        cv2.imshow('test', img)
        cv2.waitKey(5)

    cv2.imshow('test', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

