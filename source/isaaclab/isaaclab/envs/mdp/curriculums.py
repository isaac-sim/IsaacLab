# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)


class modify_env_param(ManagerTermBase):
    """Curriculum term for dynamically modifying a single environment parameter at runtime.

    This term compiles getter/setter accessors for a target attribute (specified by
    `cfg.params["address"]`) the first time it is called, then on each invocation
    reads the current value, applies a user-provided `modify_fn`, and writes back
    the result. Since None in this case can sometime be desirable value to write, we
    use token, NO_CHANGE, as non-modification signal to this class, see usage below.

    Usage:
        .. code-block:: python

            def resample_bucket_range(
                env, env_id, data, static_friction_range, dynamic_friction_range, restitution_range, num_steps
            ):
                if env.common_step_counter > num_steps:
                    range_list = [static_friction_range, dynamic_friction_range, restitution_range]
                    ranges = torch.tensor(range_list, device="cpu")
                    new_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(data), 3), device="cpu")
                    return new_buckets
                return mdp.modify_env_param.NO_CHANGE

            object_physics_material_curriculum = CurrTerm(
                func=mdp.modify_env_param,
                params={
                    "address": "event_manager.cfg.object_physics_material.func.material_buckets",
                    "modify_fn": resample_bucket_range,
                    "modify_params": {
                        "static_friction_range": [.5, 1.],
                        "dynamic_friction_range": [.3, 1.],
                        "restitution_range": [0.0, 0.5],
                        "num_step": 120000
                    }
                }
            )
    """

    NO_CHANGE = object()

    def __init__(self, cfg, env):
        """
        Initialize the ModifyEnvParam term.

        Args:
            cfg: A CurriculumTermCfg whose `params` dict must contain:
                - "address" (str): dotted path into the env where the parameter lives.
            env: The ManagerBasedRLEnv instance this term will act upon.
        """
        super().__init__(cfg, env)
        self._INDEX_RE = re.compile(r"^(\w+)\[(\d+)\]$")
        self.get_fn: callable = None
        self.set_fn: callable = None
        self.address: str = self.cfg.params.get("address")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        address: str,
        modify_fn: callable,
        modify_params: dict = {},
    ):
        """
        Apply one curriculum step to the target parameter.

        On the first call, compiles and caches the getter and setter accessors.
        Then, retrieves the current value, passes it through `modify_fn`, and
        writes back the new value.

        Args:
            env: The learning environment.
            env_ids: Sub-environment indices (unused by default).
            address: dotted path of the value retrieved from env.
            modify_fn: Function signature `fn(env, env_ids, old_value, **modify_params) -> new_value`.
            modify_params: Extra keyword arguments for `modify_fn`.
        """
        if not self.get_fn:
            self.get_fn, self.set_fn = self._compile_accessors(self._env, self.address)

        data = self.get_fn()
        new_val = modify_fn(self._env, env_ids, data, **modify_params)
        if new_val is not self.NO_CHANGE:  # if the modify_fn return NO_CHANGE signal, do not invoke self.set_fn
            self.set_fn(new_val)

    def _compile_accessors(self, root, path: str):
        """
        Build and return (getter, setter) functions for a dotted attribute path.

        Supports nested attributes, dict keys, and sequence indexing via "name[idx]".

        Args:
            root: Base object (usually `self._env`) from which to resolve `path`.
            path: Dotted path string, e.g. "foo.bar[2].baz".

        Returns:
            tuple:
              - getter: () -> current value
              - setter: (new_value) -> None (writes new_value back into the object)
        """
        # Turn "a.b[2].c" into ["a", ("b",2), "c"] and store in parts
        parts = []
        for part in path.split("."):
            m = self._INDEX_RE.match(part)
            if m:
                parts.append((m.group(1), int(m.group(2))))
            else:
                parts.append(part)

        cur = root
        for p in parts[:-1]:
            if isinstance(p, tuple):
                name, idx = p
                container = cur[name] if isinstance(cur, dict) else getattr(cur, name)
                cur = container[idx]
            else:
                cur = cur[p] if isinstance(cur, dict) else getattr(cur, p)

        self.container = cur
        self.last = parts[-1]
        # build the getter and setter
        if isinstance(self.container, tuple):
            getter = lambda: self.container[self.last]  # noqa: E731

            def setter(val):
                tuple_list = list(self.container)
                tuple_list[self.last] = val
                self.container = tuple(tuple_list)

        elif isinstance(self.container, (list, dict)):
            getter = lambda: self.container[self.last]  # noqa: E731

            def setter(val):
                self.container[self.last] = val

        elif isinstance(self.container, object):
            getter = lambda: getattr(self.container, self.last)  # noqa: E731

            def setter(val):
                setattr(self.container, self.last, val)

        else:
            raise TypeError(f"getter does not recognize the type {type(self.container)}")

        return getter, setter


class modify_term_cfg(modify_env_param):
    """Subclass of ModifyEnvParam that maps a simplified 's.'-style address
    to the full manager path. This is a more natural style for writing configurations

    Reads `cfg.params["address"]`, replaces only the first occurrence of "s."
    with "_manager.cfg.", and then behaves identically to ModifyEnvParam.
    for example: command_manager.cfg.object_pose.ranges.xpos -> commands.object_pose.ranges.xpos

    Usage:
        .. code-block:: python

            def override_value(env, env_ids, data, value, num_steps):
                if env.common_step_counter > num_steps:
                    return value
                return mdp.modify_term_cfg.NO_CHANGE

            command_object_pose_xrange_adr = CurrTerm(
                func=mdp.modify_term_cfg,
                params={
                    "address": "commands.object_pose.ranges.pos_x",   # note that `_manager.cfg` is omitted
                    "modify_fn": override_value,
                    "modify_params": {"value": (-.75, -.25), "num_steps": 12000}
                }
            )
    """

    def __init__(self, cfg, env):
        """
        Initialize the ModifyTermCfg term.

        Args:
            cfg: A CurriculumTermCfg whose `params["address"]` is a simplified
                 path using "s." as separator, e.g. instead of write "observation_manager.cfg", writes "observations".
            env: The ManagerBasedRLEnv instance this term will act upon.
        """
        super().__init__(cfg, env)
        input_address: str = self.cfg.params.get("address")
        self.address = input_address.replace("s.", "_manager.cfg.", 1)
