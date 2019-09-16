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

c_slow_dwell = 2.
c_fast_dwell = .5

cnc_s = None
cnc_c = None


def move_to(x, y, z):
    # cmd = 'G1 G54 X{0:f} Y{1:f} Z{2:f} f5'.format(x, y, z)
    cmd = 'G1 G53 X{0:f} Y{1:f} Z{2:f} f5'.format(x, y, z)
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


def is_moving(s):
    return any([abs(s.axis[x]['velocity']) > 0. for x in range(3)])


def monitored_move_to(video_capture, cmd_x, cmd_y, cmd_z):
    cnc_c.mode(linuxcnc.MODE_MDI)
    cnc_c.wait_complete()

    state = 0
    d1 = None
    d2 = None

    vv_t = None

    start_x = None
    start_y = None
    start_z = None

    while True:
        s = linuxcnc.stat()
        s.poll()
        print(s.axis[0]['input'], s.axis[0]['output'], s.axis[0]['homed'], s.axis[0]['velocity'], s.axis[0]['enabled'])

        moving = is_moving(s)

        x = s.axis[0]['input']
        y = s.axis[1]['input']
        z = s.axis[2]['input']

        mm_final, _ = haimer_camera.get_measurement(video_capture)
        if mm_final is None:
            print('mm_final is None')
            continue

        if mm_final > 1.0:
            s = 'Dangerous overshoot! mm_final={}'.format(mm_final)
            print(s)
            cnc_c.abort()
            raise Exception(s)

        if ok_for_mdi(s) and not moving:
            print('state', state, 'mm_final:', mm_final)

            if state == 0:
                start_x = x
                start_y = y
                start_z = z
                state = 1
                continue
            elif state == 1:
                move_to(cmd_x, cmd_y, cmd_z)
            elif state == 2:
                dx = x - cmd_x
                dy = y - cmd_y
                dz = z - cmd_z
                total_e = abs(dx) + abs(dy) + abs(dz)
                print('Done, move to', x, y, z, 'completed with error', total_e)
                break
        else:
            print('Waiting for move to complete.')

            if state == 1:
                print('mm_final', mm_final)
                state = 2

        sys.stdout.flush()

    return (x, y, z), (x - start_x, y - start_y, z - start_z)


def find_edge(video_capture, direction):
    cnc_c.mode(linuxcnc.MODE_MDI)
    cnc_c.wait_complete()

    state = 0
    d1 = None
    d2 = None

    vv_t = None

    start_x = None
    start_y = None
    start_z = None

    while True:
        s = linuxcnc.stat()
        s.poll()
        print(s.axis[0]['input'], s.axis[0]['output'], s.axis[0]['homed'], s.axis[0]['velocity'], s.axis[0]['enabled'])

        moving = is_moving(s)

        x = s.axis[0]['input']
        y = s.axis[1]['input']
        z = s.axis[2]['input']

        print(s.axis[0])

        mm_final, _ = haimer_camera.get_measurement(video_capture)
        if mm_final is None:
            print('mm_final is None')
            continue

        if mm_final > 1.0:
            s = 'Dangerous overshoot! mm_final={}'.format(mm_final)
            print(s)
            cnc_c.abort()
            raise Exception(s)

        in_final = mm_final / 25.4

        xe = in_final * direction[0]
        ye = in_final * direction[1]
        ze = in_final * direction[2]

        #    print('mm_final:', mm_final, 'in_final:', in_final, 'cur_x:', x, 'tar_x:', tar_x, 'xe:', xe, 'cmd_x:', cmd_x, ok_for_mdi(s))

        if ok_for_mdi(s) and not moving:
            #print('state', state, 'total_e:', total_e, 'in_final:', in_final, 'mm_final:', mm_final)
            print('state', state, 'in_final:', in_final, 'mm_final:', mm_final)

            cmd_x = x
            cmd_y = y
            cmd_z = z

            if state == 0:
                start_x = x
                start_y = y
                start_z = z
                state = 1
                continue
            elif state == 1:
                cmd_x = x - xe
                cmd_y = y - ye
                cmd_z = z - ze
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
                    cmd_y = y - ye * .95
                    cmd_z = z - ze * .95
                else:
                    print('part2')
                    cmd_x = x - xe / 2
                    cmd_y = y - ye / 2
                    cmd_z = z - ze / 2
                move_to(cmd_x, cmd_y, cmd_z)
                state = 2
            elif state == 4:
                print('Done, edge found at:', x, y, z, 'with error', total_e)
                break
        else:
            print('Waiting for move to complete.')

            if state == 1:
                print('mm_final', mm_final)
                if abs(mm_final) < 1.5:
                    print('mm_final0', mm_final)
                    cnc_c.abort()
                    state = 2
                    continue

        sys.stdout.flush()

    return (x, y, z), (x - start_x, y - start_y, z - start_z)


def touch_off(axis, v):
    if axis == 'x':
        cmd = 'G10 L2 P1 X{}'.format(v)
    elif axis == 'y':
        cmd = 'G10 L2 P1 Y{}'.format(v)
    elif axis == 'z':
        cmd = 'G10 L2 P1 Z[{} - #5403]'.format(v)
    else:
        cmd = None

    if cmd is not None:
        cnc_c.mdi(cmd)
        rv = cnc_c.wait_complete(60)
        print('touch off cmd', cmd, rv)
    else:
        print('unable to touch off')


def find_left_edge(video_capture):
    lst, dlst = find_edge(video_capture, [1, 0, 0])
    touch_off('x', lst[0])
    return lst, dlst


def find_right_edge(video_capture):
    lst, dlst = find_edge(video_capture, [-1, 0, 0])
    touch_off('x', lst[0])
    return lst, dlst


def find_aft_edge(video_capture):
    lst, dlst = find_edge(video_capture, [0, -1, 0])
    touch_off('y', lst[1])
    return lst, dlst


def find_forward_edge(video_capture):
    lst, dlst = find_edge(video_capture, [0, 1, 0])
    touch_off('y', lst[1])
    return lst, dlst


def find_top_edge(video_capture):
    lst, dlst = find_edge(video_capture, [0, 0, -1])
    touch_off('z', lst[2])
    return lst, dlst


def find_corner(video_capture, direction):
    s = linuxcnc.stat()
    s.poll()

    moving = is_moving(s)

    start_x = s.axis[0]['input']
    start_y = s.axis[1]['input']
    start_z = s.axis[2]['input']

    ball_diam = 4. / 25.4

    if direction == 'ul':
        edge_x, _ = find_left_edge(video_capture)
        _, _ = monitored_move_to(video_capture, edge_x[0] - ball_diam, start_y, start_z)
        _, _ = monitored_move_to(video_capture, edge_x[0] - ball_diam, start_y + 0.5, start_z)
        _, _ = monitored_move_to(video_capture, edge_x[0] + ball_diam, start_y + 0.5, start_z)
        edge_y, _ = find_aft_edge(video_capture)
    elif direction == 'ur':
        edge_x, _ = find_right_edge(video_capture)
        _, _ = monitored_move_to(video_capture, edge_x[0] + ball_diam, start_y, start_z)
        _, _ = monitored_move_to(video_capture, edge_x[0] + ball_diam, start_y + 0.5, start_z)
        _, _ = monitored_move_to(video_capture, edge_x[0] - ball_diam, start_y + 0.5, start_z)
        edge_y, _ = find_aft_edge(video_capture)
    elif direction == 'll':
        edge_x, _ = find_left_edge(video_capture)
        _, _ = monitored_move_to(video_capture, edge_x[0] - ball_diam, start_y, start_z)
        _, _ = monitored_move_to(video_capture, edge_x[0] - ball_diam, start_y - 0.5, start_z)
        _, _ = monitored_move_to(video_capture, edge_x[0] + ball_diam, start_y - 0.5, start_z)
        edge_y, _ = find_forward_edge(video_capture)
    elif direction == 'lr':
        edge_x, _ = find_right_edge(video_capture)
        _, _ = monitored_move_to(video_capture, edge_x[0] + ball_diam, start_y, start_z)
        _, _ = monitored_move_to(video_capture, edge_x[0] + ball_diam, start_y - 0.5, start_z)
        _, _ = monitored_move_to(video_capture, edge_x[0] - ball_diam, start_y - 0.5, start_z)
        edge_y, _ = find_forward_edge(video_capture)

    s.poll()

    moving = is_moving(s)
    print('moving:', moving)

    x = edge_x[0]
    y = edge_y[1]
    z = s.axis[2]['input']

    print('Touch off at', x, y)

    touch_off('x', x)
    touch_off('y', y)

    return (x, y, z), (x - start_x, y - start_y, z - start_z)


def find_ul_corner(video_capture):
    return find_corner(video_capture, 'ul')


def find_ur_corner(video_capture):
    return find_corner(video_capture, 'ur')


def find_ll_corner(video_capture):
    return find_corner(video_capture, 'll')


def find_lr_corner(video_capture):
    return find_corner(video_capture, 'lr')


def find_center_of_hole(video_capture):
    s = linuxcnc.stat()
    s.poll()

    moving = is_moving(s)

    start_x = s.axis[0]['input']
    start_y = s.axis[1]['input']
    start_z = s.axis[2]['input']

    left, _ = find_left_edge(video_capture)
    _, _ = monitored_move_to(video_capture, start_x, start_y, start_z)

    right, _ = find_right_edge(video_capture)

    dx = right[0] - left[0]
    cmd_x = left[0] + dx / 2.
    _, _ = monitored_move_to(video_capture, cmd_x, start_y, start_z)

    aft, _ = find_aft_edge(video_capture)
    _, _ = monitored_move_to(video_capture, cmd_x, start_y, start_z)

    forward, _ = find_forward_edge(video_capture)

    dy = forward[1] - aft[1]
    cmd_y = aft[1] + dy / 2.
    _, _ = monitored_move_to(video_capture, cmd_x, cmd_y, start_z)

    dz = 0

    s.poll()

    moving = is_moving(s)
    print('moving:', moving)

    x = s.axis[0]['input']
    y = s.axis[1]['input']
    z = s.axis[2]['input']

    print('Centered in circle at', x, y)

    touch_off('x', x)
    touch_off('y', y)

    return (x, y, z), (x - start_x, y - start_y, z - start_z)


def main():
    #    if len(sys.argv) != 1 + 3 * 3:
    #        print(f'usage: {sys.argv[0]} <xs> <xe> <xd>  <ys> <ye> <yd>  <zs> <ze> <zd>')
    #        sys.exit(1)

    args = map(float, sys.argv[1:])

    global cnc_s
    global cnc_c

    cnc_s = linuxcnc.stat()
    cnc_c = linuxcnc.command()

    #cnc_c.mode(linuxcnc.MODE_MDI)
    #cnc_c.wait_complete()

    #    s = args[0::3]
    #    e = args[1::3]
    #    d = args[2::3]

    video_capture = haimer_camera.gauge_vision_setup()
    cmds = {ord('0'): find_center_of_hole,
            ord('4'): find_right_edge,
            ord('6'): find_left_edge,
            ord('8'): find_forward_edge,
            ord('2'): find_aft_edge,
            ord('5'): find_top_edge,
            ord('1'): find_ur_corner,
            ord('3'): find_ul_corner,
            ord('7'): find_lr_corner,
            ord('9'): find_ll_corner
            }

    try:
        # Warm up
        # for i in range(30):
        #     _ = haimer_camera.get_measurement(video_capture)

        while True:
            mm_final, key = haimer_camera.get_measurement(video_capture)
            try:
                res = cmds[key](video_capture)
                print(res)
            except KeyError:
                pass
    except haimer_camera.QuitException:
        s = linuxcnc.stat()
        s.poll()

        moving = is_moving(s)

        cnc_c.abort()
        print('Quit requested', moving)

        if moving:
            sys.exit(1)


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
