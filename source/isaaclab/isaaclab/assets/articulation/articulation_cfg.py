# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from ...actuators import ActuatorBaseCfg
from ...utils import configclass
from ..asset_base_cfg import AssetBaseCfg
from .ordering import ArticulationOrderingConvention

if TYPE_CHECKING:
    from .articulation import Articulation


@configclass
class ArticulationCfg(AssetBaseCfg):
    """Configuration parameters for an articulation."""

    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
        """Initial state of the articulation."""

        # root velocity
        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Linear velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Angular velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

        # joint state
        joint_pos: dict[str, float] = {".*": 0.0}
        """Joint positions of the joints. Defaults to 0.0 for all joints."""
        joint_vel: dict[str, float] = {".*": 0.0}
        """Joint velocities of the joints. Defaults to 0.0 for all joints."""

    ##
    # Initialize configurations.
    ##

    class_type: type[Articulation] | str = "{DIR}.articulation:Articulation"

    articulation_root_prim_path: str | None = None
    """Path to the articulation root prim under the :attr:`prim_path`. Defaults to None, in which case the class
    will search for a prim with the USD ArticulationRootAPI on it.

    This path should be relative to the :attr:`prim_path` of the asset. If the asset is loaded from a USD file,
    this path should be relative to the root of the USD stage. For instance, if the loaded USD file at :attr:`prim_path`
    contains two articulations, one at `/robot1` and another at `/robot2`, and you want to use `robot2`,
    then you should set this to `/robot2`.

    The path must start with a slash (`/`).
    """

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the articulated object. Defaults to identity pose with zero velocity and zero joint state."""

    enable_joint_wrench: bool = False
    """Request incoming joint reaction wrenches before simulation startup.

    Newton uses this to allocate extended state and enable the solver's wrench computation.
    It is unnecessary for PhysX and OVPhysX, which provide these values on demand.
    Read them through :attr:`ArticulationData.body_joint_wrench`.
    """

    soft_joint_pos_limit_factor: float = 1.0
    """Fraction specifying the range of joint position limits (parsed from the asset) to use. Defaults to 1.0.

    The soft joint position limits are scaled by this factor to specify a safety region within the simulated
    joint position limits. This isn't used by the simulation, but is useful for learning agents to prevent the joint
    positions from violating the limits, such as for termination conditions.

    The soft joint position limits are accessible through the :attr:`ArticulationData.soft_joint_pos_limits` attribute.
    """

    self_collision_filter_pairs: list[tuple[str | list[str], str | list[str]]] = []
    """Additional self-collision exclusions between pairs of body-name regex groups. Defaults to [].

    Each pair excludes collisions between every body matched by its first group and every body matched
    by its second group. A group is a regex string or a non-empty list of regex strings, matched against
    complete body names. Every expression must match at least one body in the selected articulation.
    Overlapping groups are allowed; a body is never paired with itself. Use the same group on both
    sides to exclude collisions within that group.

    For example, ``[("thumb_.*", ["palm_link", "arm_link_[5-7]"])]`` filters thumb-to-palm and
    thumb-to-arm contacts. Existing exclusions, including disabled articulation self-collisions and
    adjacent-link filters, still apply. These pairs cannot re-enable contacts.

    The exclusions are authored as USD ``physics:filteredPairs`` relationships when the asset is
    constructed, before physics import and cloning. This also applies when :attr:`spawn` is None.
    Set this field before constructing the asset; changing it after model initialization has no effect.
    Body prims that are USD instance proxies cannot be edited and raise :class:`ValueError`.
    """

    joint_ordering: list[str] | tuple[str, ...] | str | ArticulationOrderingConvention | None = None
    """Public joint-name ordering convention or complete explicit permutation.

    Accepts ``"physx"``, ``"mjwarp"``, and ``"robot_schema"`` aliases, the
    corresponding :class:`ArticulationOrderingConvention` members, or a list or
    tuple (normalized to a tuple at initialization) containing every backend joint name exactly once.

    ``None`` is the default: public joint order follows active backend solver-view
    order and no ordering map is installed. An order that resolves to backend
    order is normalized to ``None`` as well, so an installed map always denotes
    an actual permutation. Symbolic resolution and map construction occur during
    articulation initialization only, not each step.
    """

    body_ordering: list[str] | tuple[str, ...] | str | ArticulationOrderingConvention | None = None
    """Public body-name ordering convention or complete explicit permutation.

    Accepts ``"physx"``, ``"mjwarp"``, and ``"robot_schema"`` aliases, the
    corresponding :class:`ArticulationOrderingConvention` members, or a list or
    tuple (normalized to a tuple at initialization) containing every backend body name exactly once.

    ``None`` is the default: public body order follows active backend solver-view
    order and no ordering map is installed. An order that resolves to backend
    order is normalized to ``None`` as well, so an installed map always denotes
    an actual permutation. Symbolic resolution and map construction occur during
    articulation initialization only, not each step.

    For fixed-base articulations, the backend root body must remain at public index
    zero; all remaining bodies may be permuted. Floating-base orders may relocate
    the root body.
    """

    actuators: dict[str, ActuatorBaseCfg] = MISSING
    """Actuators for the robot with corresponding joint names.

    Each joint can belong to at most one actuator group.
    """

    actuator_value_resolution_debug_print = False
    """Print the resolution of actuator final value when input cfg is different from USD value, Defaults to False
    """

    def _post_spawn(self, stage: Any) -> None:
        """Author collision exclusions and, for spawned assets, Newton actuator prims.

        Invoked by :class:`~isaaclab.assets.Asset` once the articulation's prims
        exist on the stage. Actuator authoring delegates to
        :func:`~isaaclab.sim.schemas.define_actuator_properties`, which gates itself
        on ``sim_cfg.use_newton_actuators``. BAM requires native Newton execution;
        other actuator configurations no-op when native execution is disabled.
        """
        if self.self_collision_filter_pairs:
            from ...sim.schemas.schemas import _apply_articulation_collision_filters  # noqa: PLC0415

            _apply_articulation_collision_filters(
                self._resolve_self_collision_filter_paths(),
                self.self_collision_filter_pairs,
                articulation_root_prim_path=self.articulation_root_prim_path,
                stage=stage,
            )
        if self.spawn is None or self.actuators is MISSING:
            return
        from ...sim.schemas.schemas_actuators import define_actuator_properties  # noqa: PLC0415

        # In InteractiveScene, articulated assets are often spawned first under
        # a template path (for example ``/World/template/Robot``) and cloned
        # into ``{ENV_REGEX_NS}`` later. Author NewtonActuator prims on the
        # actual spawned source prim so clones inherit them.
        author_prim_path = (
            self.spawn.spawn_path if self.spawn is not None and self.spawn.spawn_path is not None else self.prim_path
        )
        define_actuator_properties(author_prim_path, self.actuators, stage=stage)

    def _resolve_self_collision_filter_paths(self) -> list[str]:
        """Resolve source prim-path expressions for this asset's collision exclusions."""
        from ...cloner import path as cloner_path  # noqa: PLC0415
        from ...sim import MultiAssetSpawnerCfg, MultiUsdFileCfg, SimulationContext  # noqa: PLC0415

        expressions = [self.prim_path]
        if self.spawn is not None:
            if self.spawn.spawn_path is not None:
                expressions.append(re.escape(self.spawn.spawn_path))
            if isinstance(self.spawn, (MultiAssetSpawnerCfg, MultiUsdFileCfg)) and self.spawn.spawn_paths is not None:
                expressions.extend(re.escape(path) for path in self.spawn.spawn_paths if path is not None)

        sim = SimulationContext.instance()
        plan = sim.get_clone_plan() if sim is not None else None
        if plan is not None:
            sources = cloner_path.get_asset_prototype_paths(plan)
            expressions.extend(
                re.escape(sources[index])
                for index in cloner_path.get_asset_prototypes(plan, self.prim_path)
                if sources[index] is not None
            )
            # A non-spawning articulation may be owned by a larger scene source.
            templates, _ = cloner_path.get_world_prototype_asset_templates(plan)
            for asset, template in zip(plan.topology.world_prototypes, templates, strict=True):
                source = sources[asset]
                if source is None or "{}" not in template:
                    continue
                matched = cloner_path.match(self.prim_path, template)
                if matched is not None:
                    expressions.append(re.escape(source) + matched.suffix)
        return expressions
