# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hook for applying global Newton model parameters after builder finalization."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply_model_cfg() -> None:
    """Apply global model parameters from :class:`NewtonModelCfg` to the finalized model.

    Sets ``soft_contact_ke/kd/mu`` and optionally overrides per-shape
    ``shape_material_ke/kd/mu`` on the Newton model. This hook is always
    executed (not gated behind contact attributes) to ensure contact
    parameters are consistently applied.
    """
    from isaaclab_newton.physics import NewtonManager

    from isaaclab.physics import PhysicsManager

    cfg = PhysicsManager._cfg
    if cfg is None or not hasattr(cfg, "model_cfg") or cfg.model_cfg is None:
        return

    model = NewtonManager._model
    if model is None:
        return

    model_cfg = cfg.model_cfg
    model.soft_contact_ke = float(model_cfg.soft_contact_ke)
    model.soft_contact_kd = float(model_cfg.soft_contact_kd)
    model.soft_contact_mu = float(model_cfg.soft_contact_mu)

    if model_cfg.shape_material_ke is not None:
        model.shape_material_ke.fill_(float(model_cfg.shape_material_ke))
    if model_cfg.shape_material_kd is not None:
        model.shape_material_kd.fill_(float(model_cfg.shape_material_kd))
    if model_cfg.shape_material_mu is not None:
        model.shape_material_mu.fill_(float(model_cfg.shape_material_mu))

    logger.info(
        "Applied NewtonModelCfg: soft_contact_ke=%.1f, soft_contact_kd=%.4f, soft_contact_mu=%.2f",
        model_cfg.soft_contact_ke,
        model_cfg.soft_contact_kd,
        model_cfg.soft_contact_mu,
    )
