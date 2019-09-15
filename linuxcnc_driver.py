#!/usr/bin/env python

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
# http://linuxcnc.org/docs/2.7/html/config/python-interface.html


import math
import sys
import time

import linuxcnc

import haimer_camera

cnc_s = None
cnc_c = None


def move_to(x, y, z):
    # cmd = 'G1 G54 X{0:f} Y{1:f} Z{2:f} f5'.format(x, y, z)
    cmd = 'G1 G53 X{0:f} Y{1:f} Z{2:f} f2'.format(x, y, z)
    print('Command,' + cmd)

    cnc_c.mdi(cmd)


#    rv = cnc_c.wait_complete(60)
#    if rv != 1:
#        print('MDI command timed out')
#        sys.exit(1)


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


# From
# http://linuxcnc.org/docs/2.7/html/config/python-interface.html
def ok_for_mdi(s):
    s.poll()
    return not s.estop and s.enabled and (s.homed.count(1) == s.axes) and (s.interp_state == linuxcnc.INTERP_IDLE)


def find_ul_corner(video_capture):
    return


def find_left_edge(video_capture):
    success = False

    state = 1
    d1 = None
    d2 = None

    c_slow_dwell = 3.
    c_fast_dwell = .5

    vv_t = None

    while True:
        s = linuxcnc.stat()
        s.poll()
        print(s.axis[0]['input'], s.axis[0]['output'], s.axis[0]['inpos'], s.axis[0]['homed'], s.axis[0]['velocity'], s.axis[0]['enabled'])

        x = s.axis[0]['inpos']
        y = s.axis[1]['inpos']
        z = s.axis[2]['inpos']
        inpos = x == 1 and y == 1 and z == 1

        x = s.axis[0]['input']
        y = s.axis[1]['input']
        z = s.axis[2]['input']

        mm_final, _ = haimer_camera.get_measurement(video_capture)
        if mm_final is None:
            print('mm_final is None')
            continue

        if mm_final > 1.0:
            print('Dangerous overshoot! mm_final=', mm_final)
            cnc_c.abort()
            success = False
            break

        in_final = mm_final / 25.4

        xe = in_final
        ye = 0
        ze = 0

        #    print('mm_final:', mm_final, 'in_final:', in_final, 'cur_x:', x, 'tar_x:', tar_x, 'xe:', xe, 'cmd_x:', cmd_x, inpos, ok_for_mdi(s))

        if ok_for_mdi(s) and inpos:
            #print('state', state, 'total_e:', total_e, 'in_final:', in_final, 'mm_final:', mm_final)
            print('state', state, 'in_final:', in_final, 'mm_final:', mm_final)

            cmd_x = x
            cmd_y = y
            cmd_z = z

            if state == 1:
                cmd_x = x - xe
                move_to(cmd_x, cmd_y, cmd_z)
            elif state == 2:
                if d1 is None:
                    d1 = time.time()
                else:
                    dt = time.time() - d1
                    dwell_time = c_fast_dwell if 1.5 < abs(mm_final) else c_slow_dwell

                    if dt < dwell_time:
                        print('In settling period', dt, '<', dwell_time)
                    else:
                        d1 = None
                        total_e = abs(xe) + abs(ye) + abs(ze)
                        if dwell_time == c_fast_dwell:
                            state = 1
                        elif total_e > .0005:
                            print('total_e', total_e)
                            state = 3
                        else:
                            state = 4
                continue
            elif state == 3:
                if abs(mm_final) > .05:
                    cmd_x = x - xe * .95
                else:
                    print('part2')
                    cmd_x = x - xe / 2
                move_to(cmd_x, cmd_y, cmd_z)
                state = 2
            elif state == 4:
                print('Done, edge found at:', x, y, z, 'with error', total_e)
                success = True
                break
        else:
            print('Waiting for last move to complete.')

            if state == 1:
                print('mm_final', mm_final)
                if abs(mm_final) < 1.5:
                    print('mm_final0', mm_final)
                    cnc_c.abort()
                    state = 2
                    continue

        sys.stdout.flush()

    return success, x, y, z


def find_center_of_hole(video_capture):
    state = 1
    d1 = None

    while True:
        s = linuxcnc.stat()
        s.poll()
        print(s.axis[0]['input'], s.axis[0]['output'], s.axis[0]['inpos'], s.axis[0]['homed'], s.axis[0]['velocity'], s.axis[0]['enabled'])

        x = s.axis[0]['inpos']
        y = s.axis[1]['inpos']
        z = s.axis[2]['inpos']
        inpos = x == 1 and y == 1 and z == 1

        x = s.axis[0]['input']
        y = s.axis[1]['input']
        z = s.axis[2]['input']

        mm_final, _ = haimer_camera.get_measurement(video_capture)
        if mm_final is None:
            print('mm_final is None')
            continue

        if mm_final > 1.0:
            print('Danger!')
            cnc_c.abort()
            state = -1
            continue

        in_final = mm_final / 25.4

        # tar_x = -1.20
        # xe = x - tar_x

        xe = 0
        ye = 0
        ze = 0

        cmd_x = x
        cmd_y = y
        cmd_z = z

        if state == -1:
            continue
        if state == 1:
            tar_x = 0
            xe = in_final - tar_x

            xe *= -1
            cmd_x = x - xe / 2
        elif state == 2:
            tar_x = 0
            xe = in_final - tar_x

            cmd_x = x - xe / 2
        elif state == 3:
            xe = x - tar_x
            if abs(xe) > .1:
                xe = math.copysign(.1, xe)

            cmd_x = x - xe / 2
        elif state == 4:
            tar_y = 0
            ye = in_final - tar_y

            ye *= -1
            cmd_y = y - ye / 2
        elif state == 5:
            tar_y = 0
            ye = in_final - tar_y

            cmd_y = y - ye / 2
        elif state == 6:
            ye = y - tar_y
            if abs(ye) > .1:
                ye = math.copysign(.1, ye)

            cmd_y = y - ye / 2
            print('state 6', aft_y, forward_y, tar_y, y, ye, cmd_y)

        total_e = abs(xe) + abs(ye) + abs(ze)
        #    print('mm_final:', mm_final, 'in_final:', in_final, 'cur_x:', x, 'tar_x:', tar_x, 'xe:', xe, 'cmd_x:', cmd_x, inpos, ok_for_mdi(s))

        if ok_for_mdi(s) and inpos:
            if total_e > .0005:
                if d1 is not None and time.time() - d1 < .5:
                    print('In settling period')
                    continue
                move_to(cmd_x, cmd_y, cmd_z)
                d1 = time.time()
            else:
                print('No move is needed')
                if state == 1:
                    left_x = x
                    cmd_x += .1
                    move_to(cmd_x, y, z)
                    state = 2
                elif state == 2:
                    right_x = x
                    tar_x = left_x + (right_x - left_x) / 2.
                    state = 3
                elif state == 3:
                    state = 4
                elif state == 4:
                    aft_y = y
                    cmd_y += .1
                    move_to(x, cmd_y, z)
                    state = 5
                elif state == 5:
                    forward_y = y
                    tar_y = forward_y + (aft_y - forward_y) / 2.
                    state = 6
                elif state == 6:
                    print('Done, center found at:', cmd_x, cmd_y, cmd_z)
                    break
        else:
            print('Waiting for last move to complete.')

        sys.stdout.flush()


def main():
    #    if len(sys.argv) != 1 + 3 * 3:
    #        print(f'usage: {sys.argv[0]} <xs> <xe> <xd>  <ys> <ye> <yd>  <zs> <ze> <zd>')
    #        sys.exit(1)

    args = map(float, sys.argv[1:])

    global cnc_s
    global cnc_c

    cnc_s = linuxcnc.stat()
    cnc_c = linuxcnc.command()

    cnc_c.mode(linuxcnc.MODE_MDI)
    cnc_c.wait_complete()

    #    s = args[0::3]
    #    e = args[1::3]
    #    d = args[2::3]

    video_capture = haimer_camera.gauge_vision_setup()
    # Warm up
    for i in range(30):
        _ = haimer_camera.get_measurement(video_capture)

    while True:
        mm_final, key = haimer_camera.get_measurement(video_capture)
        if key == ord('0'):
            find_center_of_hole(video_capture)
        elif key == ord('6'):
            cnc_c.mode(linuxcnc.MODE_MDI)
            cnc_c.wait_complete()
            find_left_edge(video_capture)
        elif key == ord('7'):
            find_ul_corner(video_capture)


# grid_pts = gen_grid(s, e, d)
# for pp in grid_pts:
#    z, y, x = pp
#    move_to(x, y, z)
#    time.sleep(.1)  # settling time
#
#
#        dt = str(datetime.datetime.now())
#        #print('Result,' + ','.join(map(str, [dt, x, y, z, mm_final])), flush=True)
#        print('Result,' + ','.join(map(str, [dt, x, y, z, mm_final])))
#        sys.stdout.flush()


if __name__ == "__main__":
    main()
