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


MODE_MDI = 0
INTERP_IDLE = 0


class Stat:
    def poll(self):
        return

    tool_offset = [0, 0, 0]
    g92_offset = [0, 0, 0]
    g5x_offset = [0, 0, 0]
    position = [0, 0, 0]
    estop = False
    enabled = False
    interp_state = INTERP_IDLE
    homed = [1, 1, 1]

    axis = [{'homed': True, 'input': 0, 'output': 0, 'velocity': 0., 'enabled': True} for _ in range(3)]


class Command:
    def mode(self, _):
        return

    def wait_complete(self):
        return True


def stat():
    return Stat()


def command():
    return Command()
