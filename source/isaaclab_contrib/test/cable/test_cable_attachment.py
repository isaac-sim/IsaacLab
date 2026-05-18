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


def test_cable_object_cfg_attachments_field_default_empty():
    """CableObjectCfg exposes an `attachments` list field that defaults to empty."""
    from isaaclab_contrib.cable import CableAttachmentCfg
    from isaaclab_contrib.cable.cable_object_cfg import CableObjectCfg

    cfg = CableObjectCfg(prim_path="/World/Cable001")
    assert hasattr(cfg, "attachments"), "CableObjectCfg must expose an `attachments` field"
    assert cfg.attachments == []

    cfg2 = CableObjectCfg(
        prim_path="/World/Cable001",
        attachments=[CableAttachmentCfg(target_prim_path="/World/Plug001", cable_anchor="tail")],
    )
    assert len(cfg2.attachments) == 1
    assert cfg2.attachments[0].target_prim_path == "/World/Plug001"
    assert cfg2.attachments[0].cable_anchor == "tail"
