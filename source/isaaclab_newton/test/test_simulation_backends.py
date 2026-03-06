# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton simulation backend tests: NewtonCfg yields NewtonManager."""

from isaaclab_newton.physics import NewtonCfg, NewtonManager


def test_newton_physics_cfg_yields_newton_manager():
    """NewtonCfg.class_type refers to NewtonManager (simulation backend)."""
    cfg = NewtonCfg()
    ct = cfg.class_type
    assert ct is NewtonManager or (isinstance(ct, str) and "NewtonManager" in ct)


def test_newton_physics_cfg_backend_identity():
    """NewtonCfg used as sim backend config yields NewtonManager."""
    cfg = NewtonCfg()
    manager_cls = cfg.class_type
    if isinstance(manager_cls, str):
        assert "NewtonManager" in manager_cls
    else:
        assert manager_cls.__name__ == "NewtonManager"
