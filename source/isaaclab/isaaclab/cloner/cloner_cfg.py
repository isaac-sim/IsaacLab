# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from .cloner_strategies import random


@configclass
class TemplateCloneCfg:
    """Configuration for template-based cloning.

    This configuration is consumed by :func:`~isaaclab.scene.cloner.clone_from_template` to
    replicate one or more "prototype" prims authored under a template root into multiple
    per-environment destinations. It supports both USD-spec replication and PhysX replication
    and allows choosing between random or round-robin prototype assignment across environments.

    The cloning flow is:

    1. Discover prototypes under :attr:`template_root` whose base name starts with
        :attr:`template_prototype_identifier` (for example, ``proto_asset_0``, ``proto_asset_1``).
    2. Build a per-prototype mapping to environments according to
        :attr:`random_heterogeneous_cloning` (random) or modulo assignment (deterministic).
    3. Stamp the selected prototypes to destinations derived from :attr:`clone_regex`.
    4. Optionally perform PhysX replication for the same mapping.

    Example
    -------

    .. code-block:: python

        from isaaclab.cloner import TemplateCloneCfg, clone_from_template
        from isaacsim.core.utils.stage import get_current_stage

        stage = get_current_stage()
        cfg = TemplateCloneCfg(
            num_clones=128,
            template_root="/World/template",
            template_prototype_identifier="proto_asset",
            clone_regex="/World/envs/env_.*",
            clone_usd=True,
            clone_physics=True,
            random_heterogeneous_cloning=False,  # use round-robin mapping
            device="cpu",
        )

        clone_from_template(stage, num_clones=cfg.num_clones, template_clone_cfg=cfg)
    """

    template_root: str = "/World/template"
    """Root path under which template prototypes are authored."""

    template_prototype_identifier: str = "proto_asset"
    """Name prefix used to identify prototype prims under :attr:`template_root`."""

    clone_regex: str = "/World/envs/env_.*"
    """Destination template for per-environment paths.

    The substring ``".*"`` is replaced with ``"{}"`` internally and formatted with the
    environment index (e.g., ``/World/envs/env_0``, ``/World/envs/env_1``).
    """

    clone_usd: bool = True
    """Enable USD-spec replication to author cloned prims and optional transforms."""

    clone_physics: bool = True
    """Enable PhysX replication for the same mapping to speed up physics setup."""

    physics_clone_fn: callable | None = None
    """Function used to perform physics replication."""

    clone_strategy: callable = random
    """Function used to build prototype-to-environment mapping. Default is :func:`random`."""

    device: str = "cpu"
    """Torch device on which mapping buffers are allocated."""

    clone_in_fabric: bool = False
    """Enable/disable cloning in fabric for PhysX replication. Default is False."""
