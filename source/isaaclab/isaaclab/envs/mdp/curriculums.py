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
from typing import TYPE_CHECKING, ClassVar

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class modify_reward_weight(ManagerTermBase):
    """Curriculum that modifies the reward weight based on a step-wise schedule."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # obtain term configuration
        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        weight: float,
        num_steps: int,
    ) -> float:
        # update term settings
        if env.common_step_counter > num_steps:
            self._term_cfg.weight = weight
            env.reward_manager.set_term_cfg(term_name, self._term_cfg)

        return self._term_cfg.weight


class modify_env_param(ManagerTermBase):
    """Curriculum term for modifying an environment parameter at runtime.

    This term helps modify an environment parameter (or attribute) at runtime.
    This parameter can be any attribute of the environment, such as the physics material properties,
    observation ranges, or any other configurable parameter that can be accessed via a dotted path.

    The term uses the ``address`` parameter to specify the target attribute as a dotted path string.
    For instance, "event_manager.cfg.object_physics_material.func.material_buckets" would
    refer to the attribute ``material_buckets`` in the event manager's event term "object_physics_material",
    which is a tensor of sampled physics material properties.

    The term uses the ``modify_fn`` parameter to specify the function that modifies the value of the target attribute.
    The function should have the signature:

    .. code-block:: python

        def modify_fn(env, env_ids, old_value, **modify_params) -> new_value | modify_env_param.NO_CHANGE:
            ...

    where ``env`` is the learning environment, ``env_ids`` are the sub-environment indices,
    ``old_value`` is the current value of the target attribute, and ``modify_params``
    are additional parameters that can be passed to the function. The function should return
    the new value to be set for the target attribute, or the special token ``modify_env_param.NO_CHANGE``
    to indicate that the value should not be changed.

    At the first call to the term after initialization, it compiles getter and setter functions
    for the target attribute specified by the ``address`` parameter. The getter retrieves the
    current value, and the setter writes a new value back to the attribute.

    This term processes getter/setter accessors for a target attribute in an(specified by
    as an "address" in the term configuration`cfg.params["address"]`) the first time it is called, then on each invocation
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

                # if the step counter is not reached, return NO_CHANGE to indicate no modification.
                # we do this instead of returning None, since None is a valid value to set.
                # additionally, returning the input data would not change the value but still lead
                # to the setter being called, which may add overhead.
                return mdp.modify_env_param.NO_CHANGE

            object_physics_material_curriculum = CurrTerm(
                func=mdp.modify_env_param,
                params={
                    "address": "event_manager.cfg.object_physics_material.func.material_buckets",
                    "modify_fn": resample_bucket_range,
                    "modify_params": {
                        "static_friction_range": [0.5, 1.0],
                        "dynamic_friction_range": [0.3, 1.0],
                        "restitution_range": [0.0, 0.5],
                        "num_step": 120000
                    }
                }
            )
    """

    NO_CHANGE: ClassVar = object()
    """Special token to indicate no change in the value to be set.

    This token is used to signal that the `modify_fn` did not produce a new value. It can
    be returned by the `modify_fn` to indicate that the current value should remain unchanged.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # resolve term configuration
        if "address" not in cfg.params:
            raise ValueError("The 'address' parameter must be specified in the curriculum term configuration.")

        # store current address
        self._address: str = cfg.params["address"]
        # store accessor functions
        self._get_fn: callable = None
        self._set_fn: callable = None

    def __del__(self):
        """Destructor to clean up the compiled functions."""
        # clear the getter and setter functions
        self._get_fn = None
        self._set_fn = None
        self._container = None
        self._last_path = None

    """
    Operations.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        address: str,
        modify_fn: callable,
        modify_params: dict | None = None,
    ):
        # fetch the getter and setter functions if not already compiled
        if not self._get_fn:
            self._get_fn, self._set_fn = self._process_accessors(self._env, self._address)

        # resolve none type
        modify_params = {} if modify_params is None else modify_params

        # get the current value of the target attribute
        data = self._get_fn()
        # modify the value using the provided function
        new_val = modify_fn(self._env, env_ids, data, **modify_params)
        # set the modified value back to the target attribute
        # note: if the modify_fn return NO_CHANGE signal, we do not invoke self.set_fn
        if new_val is not self.NO_CHANGE:
            self._set_fn(new_val)

    """
    Helper functions.
    """

    def _process_accessors(self, root: ManagerBasedRLEnv, path: str) -> tuple[callable, callable]:
        """Process and return the (getter, setter) functions for a dotted attribute path.

        This function resolves a dotted path string to an attribute in the given root object.
        The dotted path can include nested attributes, dictionary keys, and sequence indexing.

        For instance, the path "foo.bar[2].baz" would resolve to `root.foo.bar[2].baz`. This
        allows accessing attributes in a nested structure, such as a dictionary or a list.

        Args:
            root: The main object from which to resolve the attribute.
            path: Dotted path string to the attribute variable. For e.g., "foo.bar[2].baz".

        Returns:
            A tuple of two functions (getter, setter), where:
            the getter retrieves the current value of the attribute, and
            the setter writes a new value back to the attribute.
        """
        # Turn "a.b[2].c" into ["a", ("b", 2), "c"] and store in parts
        path_parts: list[str | tuple[str, int]] = []
        for part in path.split("."):
            m = re.compile(r"^(\w+)\[(\d+)\]$").match(part)
            if m:
                path_parts.append((m.group(1), int(m.group(2))))
            else:
                path_parts.append(part)

        # Traverse the parts to find the container
        container = root
        for container_path in path_parts[:-1]:
            if isinstance(container_path, tuple):
                # we are accessing a list element
                name, idx = container_path
                # find underlying attribute
                if isinstance(container_path, dict):
                    seq = container[name]  # type: ignore[assignment]
                else:
                    seq = getattr(container, name)
                # save the container for the next iteration
                container = seq[idx]
            else:
                # we are accessing a dictionary key or an attribute
                if isinstance(container, dict):
                    container = container[container_path]
                else:
                    container = getattr(container, container_path)

        # save the container and the last part of the path
        self._container = container
        self._last_path = path_parts[-1]  # for "a.b[2].c", this is "c", while for "a.b[2]" it is 2

        # build the getter and setter
        if isinstance(self._container, tuple):
            get_value = lambda: self._container[self._last_path]  # noqa: E731

            def set_value(val):
                tuple_list = list(self._container)
                tuple_list[self._last_path] = val
                self._container = tuple(tuple_list)

        elif isinstance(self._container, (list, dict)):
            get_value = lambda: self._container[self._last_path]  # noqa: E731

            def set_value(val):
                self._container[self._last_path] = val

        elif isinstance(self._container, object):
            get_value = lambda: getattr(self._container, self._last_path)  # noqa: E731
            set_value = lambda val: setattr(self._container, self._last_path, val)  # noqa: E731
        else:
            raise TypeError(
                f"Unable to build accessors for address '{path}'. Unknown type found for access variable:"
                f" '{type(self._container)}'. Expected a list, dict, or object with attributes."
            )

        return get_value, set_value


class modify_term_cfg(modify_env_param):
    """Curriculum for modifying a manager term configuration at runtime.

    This class inherits from :class:`modify_env_param` and is specifically designed to modify
    the configuration of a manager term in the environment. It mainly adds the convenience of
    using a simplified address style that uses "s." as a prefix to refer to the manager's configuration.

    For instance, instead of writing "event_manager.cfg.object_physics_material.func.material_buckets",
    you can write "events.object_physics_material.func.material_buckets" to refer to the same term configuration.
    The same applies to other managers, such as "observations", "commands", "rewards", and "terminations".

    Internally, it replaces the first occurrence of "s." in the address with "_manager.cfg.",
    thus transforming the simplified address into a full manager path.

    Usage:
        .. code-block:: python

            def override_value(env, env_ids, data, value, num_steps):
                if env.common_step_counter > num_steps:
                    return value
                return mdp.modify_term_cfg.NO_CHANGE

            command_object_pose_xrange_adr = CurrTerm(
                func=mdp.modify_term_cfg,
                params={
                    "address": "commands.object_pose.ranges.pos_x",   # note: `_manager.cfg` is omitted
                    "modify_fn": override_value,
                    "modify_params": {"value": (-.75, -.25), "num_steps": 12000}
                }
            )
    """

    def __init__(self, cfg, env):
        # initialize the parent
        super().__init__(cfg, env)
        # overwrite the simplified address with the full manager path
        self._address = self._address.replace("s.", "_manager.cfg.", 1)
