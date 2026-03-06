# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX simulation backend tests: PhysxCfg yields PhysxManager."""

from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg


def test_physx_physics_cfg_yields_physx_manager():
    """PhysxCfg.class_type refers to PhysxManager (simulation backend)."""
    cfg = PhysxCfg()
    ct = cfg.class_type
    assert "PhysxManager" in (ct if isinstance(ct, str) else ct.__name__)


def test_physx_physics_cfg_backend_identity():
    """PhysxCfg used as sim backend config yields PhysxManager."""
    cfg = PhysxCfg()
    manager_cls = cfg.class_type
    if isinstance(manager_cls, str):
        assert "PhysxManager" in manager_cls
    else:
        assert manager_cls.__name__ == "PhysxManager"
