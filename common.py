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


from __future__ import print_function

import os
import time

import cv2
import numpy as np

c_label_font = cv2.FONT_HERSHEY_SIMPLEX
c_label_color = (63, 255, 63)
c_label_s = .8

c_line_color = (0, 200, 0)
c_path_color = (200, 200, 64)
c_line_s = 2


class InvalidImage(Exception):
    pass


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


@static_vars(fps_lst=[], fps_t1=None)
def draw_fps(image, append_t=True):
    if draw_fps.fps_t1 is None:
        draw_fps.fps_t1 = time.time()
        return
    if append_t:
        t2 = time.time()
        append_v(draw_fps.fps_lst, 1. / (t2 - draw_fps.fps_t1), 90)
        draw_fps.fps_t1 = t2

    fps = np.mean(draw_fps.fps_lst)

    # cv2.putText(image, f'{fps:.2f} fps', (20, 30 * 2), c_label_font, c_label_s, c_label_color)
    cv2.putText(image, '{:.2f} fps'.format(fps), (20, image.shape[0] - 30), c_label_font, c_label_s, c_label_color)


@static_vars(error_str='')
def display_error(s):
    display_error.error_str = s


def draw_error(img, s_=None):
    if s_ is None:
        s = display_error.error_str
    else:
        s = s_

    if s:
        s = 'WARNING: ' + s

        c_label_font_error = cv2.FONT_HERSHEY_SIMPLEX
        c_label_color_error = (0, 0, 255)
        c_label_s_error = 1.5

        thickness = 3
        text_size, baseline = cv2.getTextSize(s, c_label_font_error, c_label_s_error, thickness)

        text_pos = ((img.shape[1] - text_size[0]) // 2, (img.shape[0] + text_size[1]) // 2)
        cv2.putText(img, s, text_pos, c_label_font_error, c_label_s_error, c_label_color_error, thickness)


@static_vars(ind=0)
def next_frame(video_capture, fn='', fn_pattern=''):
    if not fn and not fn_pattern:
        retval, image0 = video_capture.read()
    else:
        if not fn and fn_pattern:
            for _ in range(2):
                # 'tests/haimer_camera/640x480/mov_raw_{:06d}.ppm'
                # 'tests/z_camera/1280x720/mov_raw_{:06d}.ppm'
                fn = fn_pattern.format(next_frame.ind)
                if os.path.exists(fn):
                    break
                next_frame.ind = 0

        retval, image0 = 1, cv2.imread(fn, -1)
        next_frame.ind += 1

    if not retval:
        print('rv is false')
        raise InvalidImage
    if image0.size == 0:
        print('image0 is empty')
        raise InvalidImage

    return image0
