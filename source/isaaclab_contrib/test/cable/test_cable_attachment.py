# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for cable-endpoint ↔ rigid-body attachments."""

from __future__ import annotations


def test_cable_attachment_cfg_defaults_and_types():
    """CableAttachmentCfg accepts head/tail anchors and exposes the documented defaults."""
    from isaaclab_contrib.cable import CableAttachmentCfg

    cfg = CableAttachmentCfg(target_prim_path="/World/Plug001", cable_anchor="tail")
    assert cfg.target_prim_path == "/World/Plug001"
    assert cfg.cable_anchor == "tail"
    assert cfg.local_pos == (0.0, 0.0, 0.0)
    assert cfg.local_quat == (1.0, 0.0, 0.0, 0.0)

    cfg2 = CableAttachmentCfg(
        target_prim_path="/Foo",
        cable_anchor="head",
        local_pos=(1.0, 2.0, 3.0),
        local_quat=(0.5, 0.5, 0.5, 0.5),
    )
    assert cfg2.cable_anchor == "head"
    assert cfg2.local_pos == (1.0, 2.0, 3.0)
    assert cfg2.local_quat == (0.5, 0.5, 0.5, 0.5)
