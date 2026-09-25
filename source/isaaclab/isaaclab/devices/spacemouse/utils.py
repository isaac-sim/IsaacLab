# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper functions for SpaceMouse."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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


# USB identifiers of the directly attached 3Dconnexion devices, mapped to the product name that
# selects the HID report layout used by the device listener threads. Identifiers are taken from the
# community-maintained USB ID repository at https://www.linux-usb.org/usb.ids.
# Wireless receivers are deliberately left out: their report layout differs from the cabled devices
# (see the Universal Receiver branch in the listener threads) and is unverified against hardware, so
# they are matched by product string only.
SPACEMOUSE_USB_IDS: dict[tuple[int, int], str] = {
    (0x046D, 0xC626): "SpaceNavigator",
    (0x256F, 0xC62E): "SpaceMouse Wireless",
    (0x256F, 0xC635): "SpaceMouse Compact",
}
"""Mapping from the ``(vendor_id, product_id)`` of a supported SpaceMouse to its product name."""


def resolve_device_name(device: dict[str, Any], supported_names: Sequence[str]) -> str | None:
    """Resolve the product name of an enumerated HID device against the supported SpaceMouse models.

    USB identifiers are matched first, because the ``hidapi`` wheels bundle a backend that reaches the
    device through ``libusb`` and reports empty product strings unless the process is allowed to open
    the USB node. Product strings are only used as a fallback, and are matched both verbatim and with
    the ``"3Dconnexion "`` prefix that some HID backends prepend stripped off.

    Args:
        device: An entry returned by :func:`hid.enumerate`.
        supported_names: Product names accepted by the caller.

    Returns:
        The matched product name, or None if the device is not a supported SpaceMouse.
    """
    name = SPACEMOUSE_USB_IDS.get((device["vendor_id"], device["product_id"]))
    if name in supported_names:
        return name
    product_string = (device.get("product_string") or "").strip()
    for candidate in (product_string, product_string.removeprefix("3Dconnexion ")):
        if candidate in supported_names:
            return candidate
    return None


_PERMISSION_HINT = (
    " Note that on Linux the bundled HID backend reaches the device through libusb, so the user needs"
    " read and write access to the USB node under '/dev/bus/usb'; granting access to '/dev/hidraw*'"
    " alone is not sufficient. See the teleoperation documentation for the required udev rule."
)
"""Guidance appended to the discovery errors, which are almost always caused by USB permissions."""


def device_not_found_message(
    supported_names: Sequence[str],
    enumerated_devices: Sequence[dict[str, Any]],
    open_failures: Sequence[str] = (),
) -> str:
    """Compose the error raised when no supported SpaceMouse could be opened.

    Args:
        supported_names: Product names accepted by the caller.
        enumerated_devices: The HID devices reported by :func:`hid.enumerate` during the search.
        open_failures: Descriptions of the supported devices that were found but could not be opened.

    Returns:
        An error message naming either the devices that could not be opened, or, when none matched,
        the supported models and the HID devices that were seen.
    """
    if open_failures:
        return (
            "Found a supported SpaceMouse but could not open it: " + "; ".join(open_failures) + "." + _PERMISSION_HINT
        )
    seen = ", ".join(
        f"{device['vendor_id']:#06x}:{device['product_id']:#06x} ({device.get('product_string') or 'unnamed'})"
        for device in enumerated_devices
    )
    return (
        "No device found by SpaceMouse. Is the device connected?"
        f" Supported models: {', '.join(supported_names)}."
        f" Enumerated HID devices: {seen or 'none'}."
    ) + _PERMISSION_HINT


def describe_open_failure(device_name: str, vendor_id: int, product_id: int, error: OSError) -> str:
    """Describe a supported device that was detected but could not be opened.

    Args:
        device_name: Product name of the detected device.
        vendor_id: USB vendor identifier of the detected device.
        product_id: USB product identifier of the detected device.
        error: The error raised while opening the device.

    Returns:
        A short description naming the device and the underlying error.
    """
    return f"'{device_name}' ({vendor_id:#06x}:{product_id:#06x}): {error}"


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
