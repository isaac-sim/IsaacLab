# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import ctypes

from isaaclab_newton.physics._cubric import CubricBindings


def _verify_version(major: int, minor: int) -> bool:
    """Run the cubric ABI verifier against an in-memory plugin descriptor."""
    interface_name = ctypes.create_string_buffer(b"omni::cubric::IAdapter\0")
    interface = (ctypes.c_uint64 * 2)()
    interface[0] = ctypes.addressof(interface_name)
    ctypes.c_uint32.from_address(ctypes.addressof(interface) + 8).value = major
    ctypes.c_uint32.from_address(ctypes.addressof(interface) + 12).value = minor

    plugin_descriptor = (ctypes.c_ubyte * 56)()
    ctypes.c_uint64.from_address(ctypes.addressof(plugin_descriptor) + 40).value = ctypes.addressof(interface)
    ctypes.c_uint64.from_address(ctypes.addressof(plugin_descriptor) + 48).value = 1

    get_descriptor_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)
    get_descriptor = get_descriptor_type(lambda _: ctypes.addressof(plugin_descriptor))
    framework = (ctypes.c_uint64 * 13)()
    framework[12] = ctypes.cast(get_descriptor, ctypes.c_void_p).value

    return CubricBindings._verify_iadapter_version(ctypes.addressof(framework), 1)


def test_cubric_adapter_rejects_unvalidated_minor_version():
    """A newer minor ABI must use the safe CPU transform-hierarchy fallback."""
    assert _verify_version(0, 1)
    assert not _verify_version(0, 2)
