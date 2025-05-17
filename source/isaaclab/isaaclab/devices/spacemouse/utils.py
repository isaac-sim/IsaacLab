# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper functions for SpaceMouse."""

# MIT License
#
# Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def convert_buffer(b1, b2):
    """Converts raw SpaceMouse readings to commands.

    Args:
        b1: 8-bit byte
        b2: 8-bit byte

    Returns:
        Scaled value from Space-mouse message
    """
    return _scale_to_control(_to_int16(b1, b2))


"""
Private methods.
"""


def _to_int16(y1, y2):
    """Convert two 8 bit bytes to a signed 16 bit integer.

    Args:
        y1: 8-bit byte
        y2: 8-bit byte

    Returns:
        16-bit integer
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def _scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """Normalize raw HID readings to target range.

    Args:
        x: Raw reading from HID
        axis_scale: (Inverted) scaling factor for mapping raw input value
        min_v: Minimum limit after scaling
        max_v: Maximum limit after scaling

    Returns:
        Clipped, scaled input from HID
    """
    x = x / axis_scale
    return min(max(x, min_v), max_v)
