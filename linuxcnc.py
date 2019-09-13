#!/usr/bin/env python3

# Goal: Use machine vision to read a Haimer 3d Taster and control LinuxCNC
# find the center of a hole.

# Precondition: Before running this program, configure LinuxCNC, home axes,
# configure desired units, move probe tip in to plane of the hole, etc.
# Failure to do this could lead to damage.

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


# Documentation for external libraries
# http://linuxcnc.org/docs/2.6/html/common/python-interface.html
# http://linuxcnc.org/docs/2.7/html/config/python-interface.html


import datetime
import sys
import time

import haimer_camera
import linuxcnc

cnc_s = None
cnc_c = None


def move_to(x, y, z):
    cmd = 'G1 G54 X{0:f} Y{1:f} Z{2:f} f5'.format(x, y, z)
    print('Command,' + cmd)

    cnc_c.mdi(cmd)
    rv = cnc_c.wait_complete(60)
    if rv != 1:
        print('MDI command timed out')
        sys.exit(1)


# generate grid points in a zig-zag rectilinear pattern, helper function
# s: vector of start positions for each axis
# d: vector of step size for each axis
# n: vector of number of steps for each axis
# o: vector of 1,-1 controlling direction axis is traversed
# l: recursion depth
# p: vector containing current position
# rv: vector containing accumulated positions
def gen_grid_(s, d, n, o, l, p, rv):
    if l == len(p):
        pp = [x for x in p]
        for i in range(len(pp)):
            pp[i] = s[i] + d[i] * pp[i]
        rv += [pp]
        return

    for i in range(n[l]):
        gen_grid_(s, d, n, o, l + 1, p, rv)
        p[l] += o[l]

    if o[l] == 1:
        o[l] = -1
        p[l] = n[l] - 1
    elif o[l] == -1:
        o[l] = 1
        p[l] = 0


# generate grid points in a zig-zag rectalinear pattern
# s: vector of start positions for each axis
# e: vector of end positions for each axis
# d: vector of step size for each axis
# returns: vector containing ordered grid positions
def gen_grid(s, e, d):
    n = [1] * len(s)
    for i in range(len(n)):
        n[i] = int((e[i] - s[i]) / d[i] + 1.5)

    grid_pts = []
    gen_grid_(s[::-1], d[::-1], n[::-1], [1] * len(s), 0, [0] * len(s), grid_pts)
    return grid_pts


def main():
    if len(sys.argv) != 1 + 3 * 3:
        print(f'usage: {sys.argv[0]} <xs> <xe> <xd>  <ys> <ye> <yd>  <zs> <ze> <zd>')
        sys.exit(1)

    args = map(float, sys.argv[1:])

    global cnc_s
    global cnc_c

    cnc_s = linuxcnc.stat()
    cnc_c = linuxcnc.command()

    cnc_c.mode(linuxcnc.MODE_MDI)
    cnc_c.wait_complete()

    s = args[0::3]
    e = args[1::3]
    d = args[2::3]

    video_capture = haimer_camera.gauge_vision_setup()
    # Warm up
    for i in range(30):
        mm_final = haimer_camera.get_measurement(video_capture)

    grid_pts = gen_grid(s, e, d)
    for pp in grid_pts:
        z, y, x = pp
        move_to(x, y, z)
        time.sleep(.1)  # settling time

        mm_final = haimer_camera.get_measurement(video_capture)
        print('mm_final:', mm_final)

        dt = str(datetime.datetime.now())
        print('Result,' + ','.join(map(str, [dt, x, y, z, mm_final])), flush=True)


if __name__ == "__main__":
    main()
