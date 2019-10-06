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

# Ideas and observations:
# 1) Instead of blindly moving the probe by a fixed amount when moving
#    from the first to the second edge, the probe could be dragged 
#    along the first edge until past the second edge by half
#    the probe diameter.
# 2) The find_edge(...) may be able to be simplified using monitored_move_to(...)
# 3) The probe3d(...) method is meant as a starting point and has not been tested.
# 4) External cylindrical boss center finding would be useful. Likely to be
#    very different than find_center_of_hole(...)


from __future__ import print_function

import os
import sys
import time

import cv2
import numpy as np
import linuxcnc

import haimer_camera
import z_camera

c_slow_dwell = 2.
c_fast_dwell = .5
c_feedrate = 5
c_feedrate_fast = 20

cnc_s = None
cnc_c = None

c_label_font = cv2.FONT_HERSHEY_SIMPLEX
c_label_color = (0, 255, 0)
c_label_s = .8

video_capture2 = None

class QuitException(Exception):
    pass


class OvershootException(Exception):
    def __init__(self, mm_final):
        self.mm_final = mm_final

    def __str__(self):
        s = 'Dangerous overshoot! mm_final={}'.format(self.mm_final)
        return s


class NotReady(Exception):
    def __init__(self):
        pass

    def __str__(self):
        s = 'Not ready for MDI, not homed?'
        return s


# Decorator for static variables, from

# Decorator for static variables, from
# https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def move_to(x, y, z, feedrate=None):
    if feedrate is None:
        feedrate = c_feedrate

    # cmd = 'G1 G54 X{0:f} Y{1:f} Z{2:f} f5'.format(x, y, z)
    cmd = 'G1 G53 X{0:f} Y{1:f} Z{2:f} f{3:d}'.format(x, y, z, feedrate)
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


def monitored_move_to(video_capture, cmd_x, cmd_y, cmd_z, max_mm=1.0, local=False, feedrate=None):
    if feedrate is None:
        feedrate = c_feedrate

    cnc_c.mode(linuxcnc.MODE_MDI)
    cnc_c.wait_complete()

    if local:
        print('Converting:', [cmd_x, cmd_y, cmd_z])
        cmd_x, cmd_y, cmd_z = part_to_machine_cs([cmd_x, cmd_y, cmd_z])
        print('Converted:', [cmd_x, cmd_y, cmd_z])

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

        mm_final, circles = update_view(video_capture, video_capture2)
        if mm_final is None:
            print('mm_final is None')
            continue

        if mm_final > max_mm:
            cnc_c.abort()
            raise OvershootException(mm_final)

        if not ok_for_mdi(s) and not moving:
            raise NotReady()
        elif ok_for_mdi(s) and not moving:
            print('state', state, 'mm_final:', mm_final)

            if state == 0:
                start_x = x
                start_y = y
                start_z = z
                state = 1
                continue
            elif state == 1:
                move_to(cmd_x, cmd_y, cmd_z, feedrate=feedrate)
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

    in_state_3 = 0

    while True:
        s = linuxcnc.stat()
        s.poll()
        print(s.axis[0]['input'], s.axis[0]['output'], s.axis[0]['homed'], s.axis[0]['velocity'], s.axis[0]['enabled'])

        moving = is_moving(s)

        x = s.axis[0]['input']
        y = s.axis[1]['input']
        z = s.axis[2]['input']

        print(s.axis[0])

        mm_final, circles = update_view(video_capture, video_capture2)
        if mm_final is None:
            print('mm_final is None')
            continue

        if mm_final > 1.0:
            cnc_c.abort()
            raise OvershootException(mm_final)

        in_final = mm_final / 25.4

        xe = in_final * direction[0]
        ye = in_final * direction[1]
        ze = in_final * direction[2]

        #    print('mm_final:', mm_final, 'in_final:', in_final, 'cur_x:', x, 'tar_x:', tar_x, 'xe:', xe, 'cmd_x:', cmd_x, ok_for_mdi(s))

        div_f = 2

        if not ok_for_mdi(s) and not moving:
            raise NotReady()
        elif ok_for_mdi(s) and not moving:
            # print('state', state, 'total_e:', total_e, 'in_final:', in_final, 'mm_final:', mm_final)
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
                in_state_3 += 1
                if in_state_3 > 5:
                    print('Oscillation may be occuring. Reducing div_f')
                    # Could alternatively increase the finishing error tolerance
                    div_f = 4
                if abs(mm_final) > .05:
                    cmd_x = x - xe * .95
                    cmd_y = y - ye * .95
                    cmd_z = z - ze * .95
                else:
                    print('part2')
                    cmd_x = x - xe / div_f
                    cmd_y = y - ye / div_f
                    cmd_z = z - ze / div_f
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
        # Offset the position v by the tool length along the z axis.
        # http://linuxcnc.org/docs/2.7/html/gcode/overview.html#sub:numbered-parameters
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
    return find_edge(video_capture, [1, 0, 0])


def find_right_edge(video_capture):
    return find_edge(video_capture, [-1, 0, 0])


def find_aft_edge(video_capture):
    return find_edge(video_capture, [0, -1, 0])


def find_forward_edge(video_capture):
    return find_edge(video_capture, [0, 1, 0])


def find_top_edge(video_capture):
    return find_edge(video_capture, [0, 0, -1])


def touch_off_left_edge(video_capture):
    lst, dlst = find_left_edge(video_capture)
    touch_off('x', lst[0])
    return lst, dlst


def touch_off_right_edge(video_capture):
    lst, dlst = find_right_edge(video_capture)
    touch_off('x', lst[0])
    return lst, dlst


def touch_off_aft_edge(video_capture):
    lst, dlst = find_aft_edge(video_capture)
    touch_off('y', lst[1])
    return lst, dlst


def touch_off_forward_edge(video_capture):
    lst, dlst = find_forward_edge(video_capture)
    touch_off('y', lst[1])
    return lst, dlst


def touch_off_top_edge(video_capture):
    lst, dlst = find_top_edge(video_capture)
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

    return (x, y, z), (x - start_x, y - start_y, z - start_z)


def touch_off_corner(video_capture, direction):
    lst, dlst = find_corner(video_capture, direction)
    x, y = lst[:2]
    print('Touch off at', x, y)
    touch_off('x', x)
    touch_off('y', y)
    return lst, dlst


def touch_off_ul_corner(video_capture):
    return touch_off_corner(video_capture, 'ul')


def touch_off_ur_corner(video_capture):
    return touch_off_corner(video_capture, 'ur')


def touch_off_ll_corner(video_capture):
    return touch_off_corner(video_capture, 'll')


def touch_off_lr_corner(video_capture):
    return touch_off_corner(video_capture, 'lr')


def find_center_of_hole(video_capture):
    s = linuxcnc.stat()
    s.poll()

    moving = is_moving(s)

    start_x = s.axis[0]['input']
    start_y = s.axis[1]['input']
    start_z = s.axis[2]['input']

    left, _ = find_left_edge(video_capture)
    _, _ = monitored_move_to(video_capture, start_x, start_y, start_z, feedrate=c_feedrate_fast)

    right, _ = find_right_edge(video_capture)

    dx = right[0] - left[0]
    cmd_x = left[0] + dx / 2.
    _, _ = monitored_move_to(video_capture, cmd_x, start_y, start_z, feedrate=c_feedrate_fast)

    aft, _ = find_aft_edge(video_capture)
    _, _ = monitored_move_to(video_capture, cmd_x, start_y, start_z, feedrate=c_feedrate_fast)

    forward, _ = find_forward_edge(video_capture)

    dy = forward[1] - aft[1]
    cmd_y = aft[1] + dy / 2.
    _, _ = monitored_move_to(video_capture, cmd_x, cmd_y, start_z, feedrate=c_feedrate_fast)

    dz = 0

    s.poll()

    moving = is_moving(s)
    print('moving:', moving)

    x = s.axis[0]['input']
    y = s.axis[1]['input']
    z = s.axis[2]['input']

    diam = dy

    return (x, y, z), (x - start_x, y - start_y, z - start_z), (dx, dy, dz), diam


def touch_off_center_of_hole(video_capture):
    lst, dlst, _, diam = find_center_of_hole(video_capture)
    x, y = lst[:2]
    print('Centered in circle at', x, y, 'with diameter', diam)
    touch_off('x', x)
    touch_off('y', y)
    return lst, dlst


def probe3d(video_capture):
    s = [0., 0., 1.]
    e = [1., 1., 1.]
    d = [.1, .1, 1.]

    final_pts = []

    grid_pts = gen_grid(s, e, d)
    for pp in grid_pts:
        z, y, x = pp

        cmd_x = x
        cmd_y = y
        cmd_z = z
        monitored_move_to(video_capture, cmd_x, cmd_y, cmd_z)

        lst, dlst = find_edge(video_capture, [0, 0, -1])
        final_z = lst[2]

        monitored_move_to(video_capture, cmd_x, cmd_y, cmd_z)

        final_pts += [[x, y, final_z]]
        print(x, y, final_z)
        sys.stdout.flush()

    return final_pts


def part_to_machine_cs(pt):
    global cnc_s
    cnc_s.poll()

    tool_off = cnc_s.tool_offset[:3]
    g92_off = cnc_s.g92_offset[:3]
    g5x_off = cnc_s.g5x_offset[:3]
    machine_pos = cnc_s.position[:3]

# position + tool_off + g5x_off + g92_off = machine_pos
# position = machine_pos - g92_off - g5x_off - tool_off
    nm = [pt[i] + tool_off[i] + g5x_off[i] + g92_off[i] for i in range(3)]
    return nm


def machine_to_part_cs(machine_pos=None):
    global cnc_s
    cnc_s.poll()

    tool_off = cnc_s.tool_offset[:3]
    g92_off = cnc_s.g92_offset[:3]
    g5x_off = cnc_s.g5x_offset[:3]
    if machine_pos is None:
        machine_pos = cnc_s.position[:3]

# position + tool_off + g5x_off + g92_off = machine_pos
# position = machine_pos - g92_off - g5x_off - tool_off
    nm = [machine_pos[i] - g92_off[i] - g5x_off[i] - tool_off[i] for i in range(3)]
    return nm


# def go():
#     global cnc_s
#     global cnc_c
# 
#     cnc_s.poll()
#     print(dir(cnc_s))
#     print(dir(cnc_c))
#     print(cnc_s.g5x_index)
#     print(cnc_s.tool_in_spindle)
#     print(cnc_s.linear_units)
# 
#     tool_off = cnc_s.tool_offset[:3]
#     g92_off = cnc_s.g92_offset[:3]
#     g5x_off = cnc_s.g5x_offset[:3]
#     machine_pos = cnc_s.position[:3]
# 
#     print('tool_off', tool_off)
#     print('g92_off', g92_off)
#     print('g5x_off', g5x_off)
#     print('machine_pos (relative to home position)', machine_pos)
#     print('axis_input', [cnc_s.axis[i]['input'] for i in range(3)])
#     print('axis_output', [cnc_s.axis[i]['output'] for i in range(3)])
# 
#     print('part_to_machine_cs([0,0,0])', part_to_machine_cs([0, 0, 0]))
#     print('machine_to_part_cs()', machine_to_part_cs())
# 
#     return machine_to_part_cs()


# Machine Coordinates (G53)
# Nine Coordinate System Offsets (G54-G59.3)
# Global Offsets (G92)
# Is a good use case would be G92 to the corner of a fixture plate and G54-59.3 to each part location relative to the fixture plate location?
# The G54-G59.3 values could then be saved and only G92 would need to be set when the fixture plate is located.
# machine_origin is the home position of the machine, it is set by homing the machine
# machine_pos is the offset from the machine origin (machine coordinate system)
# g92_off contains offsets of the origin, they are added to commanded position to produce a machine position (global coordinate system) (persistent)
# g5x_off contains offsets of the origin, they are added to commanded position to produce a machine position (local or part coordinate system) (persistent)

#1
#53
#0.0393700787402
#('tool_off', (0.0, 0.0, 3.652))
#('g92_off', (0.0, 0.0, 0.0))
#('g5x_off', (1.3769656307399998, -2.31421642429, -8.739318121759998))
#('machine_pos', (2.951150911092579, -0.3856613977556656, 0.017924044147022797))
# 1.5742 1.9285 5.1051

#1
#100
#0.0393700787402
#('tool_off', (0.0, 0.0, 4.776))
#('g92_off', (0.0, 0.0, 0.0))
#('g5x_off', (1.3769656307399998, -2.31421642429, -8.739318121759998))
#('machine_pos (relative to home position)', (2.951150911092579, -0.3856613977556656, 0.017924044147022797))
# 1.5742 1.9285 3.9811

#1
#100
#0.0393700787402
#('tool_off', (0.0, 0.0, 4.776))
#('g92_off', (0.0, 0.0, 0.0))
#('g5x_off', (1.3769656307399998, -2.31421642429, -8.739318121759998))
#('machine_pos (relative to home position)', (2.951150911092579, -0.3856613977556656, -0.8220035369112901))
# 1.5742 1.9285 3.1413


def re_holes(video_capture, circles):
    part_z = 5.4 / 25.4
    # With a tight max_mm, it's likely to see false errors after the find_center_of_hole
    # Having a dwell after find_center_of_hole so the needles can settle after the fast move to center might be enough.
    # The best solution may be to read a few frames from the camera, clear out any old frames and get a dwell time
    max_mm = -1.95

    # The probe tip should travel half the tip diameter above the part surface
    # The probe tip should drop into a hole, half the tip diameter plus 1mm
    # The offsets may be a little different than expected because the probe compresses
    # before reading 0. 
    zh = part_z + haimer_camera.c_haimer_ball_diam / 25.4
    zl = part_z - 1 / 25.4

    lst = machine_to_part_cs()
    sx, sy, sz = lst
    start_pt = (sx, sy)
    end_pt = (0, 0)
    lst = []
    for c in circles:
        print(c)
        if c[1]:
            m_pt, diam = c[1]
            lst += [c[1]]

    results = []
    if lst:
        t0 = time.time()
        (x, y), z = start_pt, zh
        monitored_move_to(video_capture, x, y, z, max_mm=max_mm, local=True, feedrate=c_feedrate_fast)
        for i, c in enumerate(lst):
            lbl = chr(ord('A') + i)
            (x, y), diam = c

            monitored_move_to(video_capture, x, y, zh, max_mm=max_mm, local=True, feedrate=c_feedrate_fast)
            monitored_move_to(video_capture, x, y, zl, max_mm=max_mm, local=True, feedrate=c_feedrate)
            cpt, _, delta, diam2 = find_center_of_hole(video_capture)
            ll = machine_to_part_cs(cpt)
            ll2 = machine_to_part_cs()
            results += [(lbl, time.time()-t0, x, y, ll[0], ll[1], ll2[0], ll2[1], diam, diam2, abs(delta[0]), abs(delta[1]))]
            x, y = ll[:2]
            # time.sleep(.25)
            for i in range(10):
                mm_final, circles = update_view(video_capture, video_capture2)
                print(mm_final)
            monitored_move_to(video_capture, x, y, zh, max_mm=max_mm, local=True, feedrate=c_feedrate_fast)

        (x, y), z = end_pt, zh
        monitored_move_to(video_capture, x, y, z, max_mm=max_mm, local=True, feedrate=c_feedrate_fast)

    return results


def click_and_crop(event, x0, y0, flags, param):
    c_final_image_scale_factor = 1280/1920.
    x = int(round(x0 / c_final_image_scale_factor))
    y = int(round(y0 / c_final_image_scale_factor))

    z_camera.click_and_crop(event, x, y, flags, param)


_error_str = None


def display_error(s):
    global _error_str
    _error_str = s


c_camera_name = 'linuxcnc_driver'


@static_vars(save=False, record=False, record_ind=0, do_touchoff=False)
def update_view(video_capture, video_capture2):
    cmds = {ord('0'): touch_off_center_of_hole,
            ord('4'): touch_off_right_edge,
            ord('6'): touch_off_left_edge,
            ord('8'): touch_off_forward_edge,
            ord('2'): touch_off_aft_edge,
            ord('5'): touch_off_top_edge,
            ord('1'): touch_off_ur_corner,
            ord('3'): touch_off_ul_corner,
            ord('7'): touch_off_lr_corner,
            ord('9'): touch_off_ll_corner
            }

    lst = machine_to_part_cs()
    z_camera.get_measurement.start_mpt = lst
    z_camera.get_measurement.cur_mpt = lst

    mm_final, haimer_img = haimer_camera.get_measurement(video_capture)
    circles, z_img = z_camera.get_measurement(video_capture2)
    if haimer_img.shape == (1008, 1344, 3):
        haimer_img = cv2.resize(haimer_img, (960, 720))
    final_img = np.hstack([z_img, haimer_img])

    if _error_str:
        s = 'WARNING: ' + _error_str

        c_label_font_error = cv2.FONT_HERSHEY_SIMPLEX
        c_label_color_error = (0, 0, 255)
        c_label_s_error = 1.5

        thickness = 3
        text_size, baseline = cv2.getTextSize(s, c_label_font_error, c_label_s_error, thickness)

        text_pos = ((final_img.shape[1] - text_size[0]) // 2, (final_img.shape[0] + text_size[1]) // 2)
        cv2.putText(final_img, s, text_pos, c_label_font_error, c_label_s_error, c_label_color_error, thickness)

    if update_view.do_touchoff:
        s = 'Touch off using numeric keypad'
        thickness = 1
        text_size, baseline = cv2.getTextSize(s, c_label_font, c_label_s, thickness)
        text_pos = ((final_img.shape[1] - text_size[0]) // 2, (final_img.shape[0] + text_size[1]) // 2)
        cv2.putText(final_img, s, text_pos, c_label_font, c_label_s, c_label_color, thickness)
        
    c_final_image_scale_factor = 1280/1920.
    final_img_resized = cv2.resize(final_img, None, fx=c_final_image_scale_factor, fy=c_final_image_scale_factor)

    cv2.imshow(c_camera_name, final_img_resized)
    key = cv2.waitKey(5)

    if update_view.record:
        fn1 = 'mov_all_l_{:06}.ppm'.format(update_view.record_ind)
        cv2.imwrite(fn1, final_img)
        update_view.record_ind += 1
        print('Recorded {}'.format(fn1))

    if update_view.save:
        update_view.save = False

        for i in range(100):
            fn1 = 'all_l_{:03}.png'.format(i)
            if not os.path.exists(fn1):
                cv2.imwrite(fn1, final_img)
                print('Wrote image {}'.format(fn1))
                break

    accepted = False
    if update_view.do_touchoff:
        try:
            res = cmds[key](video_capture)
            update_view.do_touchoff = False
            print(res)
            accepted = True
        except KeyError:
            pass

    accepted = True
    if key == ord('r'):
        update_view.record = not update_view.record
    elif key == ord('s'):
        update_view.save = not update_view.save
    elif key == ord('t'):
        update_view.do_touchoff = not update_view.do_touchoff
    elif key == ord('g'):
        results = re_holes(video_capture, circles)

        print

        print('results += [(lbl, time.time()-t0, x, y, ll[0], ll[1], ll2[0], ll2[1], diam, diam2, abs(delta[0]), abs(delta[1]))]')
        for res in results:
            print(res)
       
        print
    elif key in [27, ord('q')]: # Escape or q
        raise QuitException
    else:
        accepted = False

    if not accepted or key in [ord('r'), ord('s')]:
        accepted = haimer_camera.process_key(key)
    if not accepted or key in [ord('r'), ord('s')]:
        accepted = z_camera.process_key(key)

    return mm_final, circles


def main():
    cv2.namedWindow(c_camera_name)
    cv2.setMouseCallback(c_camera_name, click_and_crop)

    global cnc_s
    global cnc_c

    cnc_s = linuxcnc.stat()
    cnc_c = linuxcnc.command()

    video_capture = haimer_camera.gauge_vision_setup()
    global video_capture2
    video_capture2 = z_camera.gauge_vision_setup()

    while True:
        try:
            mm_final, circles = update_view(video_capture, video_capture2)
        except OvershootException as e:
            cnc_c.abort()
            display_error(str(e))
        except NotReady as e:
            display_error(str(e))
        except (haimer_camera.QuitException, z_camera.QuitException, QuitException):
            s = linuxcnc.stat()
            s.poll()

            moving = is_moving(s)

            cnc_c.abort()
            print('Quit requested', moving)

            if moving:
                sys.exit(1)
            else:
                break


if __name__ == "__main__":
    main()
