# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from leapp.utils.tensor_description import TensorSemantics

from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.managers import ManagerTermBase


def _lookup_annotating_getter(
    annotating_getters_by_type: dict[type, dict[str, Callable]], real_data: Any, name: str
) -> Callable | None:
    """Return the annotating getter for a property on the given data object, if any."""
    for data_cls in type(real_data).__mro__:
        getter = annotating_getters_by_type.get(data_cls, {}).get(name)
        if getter is not None:
            return getter
    return None


class _DataProxy:
    """Proxy around a real data object that intercepts tensor-returning property reads.

    The real data object may be any scene entity data class (``ArticulationData``,
    ``RigidObjectData``, sensor data classes, etc.).  The proxy intercepts all
    ``@property`` getters that were registered during scene introspection.  When
    the getter returns a ``torch.Tensor``, the result is annotated as a LEAPP
    input and cached for deduplication.  Non-tensor results are forwarded
    transparently.

    Properties with ``_leapp_semantics`` metadata produce rich annotations
    (kind, element_names).  Properties without it are still traced — with no
    semantic metadata — so that no tensor is silently baked as a constant.

    All other attribute access is forwarded transparently to the real object.
    """

    def __init__(
        self,
        real_data: Any,
        annotating_getters_by_type: dict[type, dict[str, Callable]],
        cache: dict,
        input_name_resolver: Callable,
    ):
        object.__setattr__(self, "_real_data", real_data)
        object.__setattr__(self, "_annotating_getters_by_type", annotating_getters_by_type)
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(self, "_input_name_resolver", input_name_resolver)

    def __getattr__(self, name):
        """Intercept registered property reads; forward everything else."""
        real_data = object.__getattribute__(self, "_real_data")
        getter = _lookup_annotating_getter(
            object.__getattribute__(self, "_annotating_getters_by_type"), real_data, name
        )
        if getter is not None:
            cache = object.__getattribute__(self, "_cache")
            cache_key = (id(real_data), name)
            if cache_key in cache:
                return cache[cache_key].clone()
            input_name = object.__getattribute__(self, "_input_name_resolver")(name)
            result = getter(real_data, input_name)
            if isinstance(result, torch.Tensor):
                cache[cache_key] = result
            return result
        return getattr(real_data, name)


class _EntityProxy:
    """Proxy around a real scene entity that returns a ``_DataProxy`` for ``.data``.

    All other attribute access is forwarded transparently to the real asset.
    """

    def __init__(self, real_entity: Any, data_proxy: _DataProxy):
        object.__setattr__(self, "_real_entity", real_entity)
        object.__setattr__(self, "_data_proxy", data_proxy)

    @property
    def data(self):
        """Return the annotating data proxy instead of the real data object."""
        return object.__getattribute__(self, "_data_proxy")

    def __getattr__(self, name):
        """Forward all non-data attribute access to the real scene entity."""
        return getattr(object.__getattribute__(self, "_real_entity"), name)


class _EntityMappingProxy:
    """Proxy around a mapping of scene entities that lazily wraps data-producing entries."""

    def __init__(self, real_mapping, annotating_getters_by_type: dict[type, dict[str, Callable]], cache: dict):
        object.__setattr__(self, "_real_mapping", real_mapping)
        object.__setattr__(self, "_annotating_getters_by_type", annotating_getters_by_type)
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(self, "_proxied", {})

    def __getitem__(self, key):
        """Return a proxied entity when it has a ``.data`` attribute."""
        proxied = object.__getattribute__(self, "_proxied")
        if key in proxied:
            return proxied[key]
        real_mapping = object.__getattribute__(self, "_real_mapping")
        entity = real_mapping[key]
        data = getattr(entity, "data", None)
        if data is None:
            return entity
        annotating_getters_by_type = object.__getattribute__(self, "_annotating_getters_by_type")
        data_proxy = _DataProxy(
            data,
            annotating_getters_by_type,
            object.__getattribute__(self, "_cache"),
            input_name_resolver=lambda prop_name: f"{key}_{prop_name}",
        )
        proxy = _EntityProxy(entity, data_proxy)
        proxied[key] = proxy
        return proxy

    def get(self, key, default=None):
        """Return a proxied entity when present, default otherwise."""
        real_mapping = object.__getattribute__(self, "_real_mapping")
        if key not in real_mapping:
            return default
        return self[key]

    def __iter__(self):
        return iter(object.__getattribute__(self, "_real_mapping"))

    def __len__(self):
        return len(object.__getattribute__(self, "_real_mapping"))

    def __getattr__(self, name):
        """Forward all other mapping access to the real mapping."""
        return getattr(object.__getattribute__(self, "_real_mapping"), name)


class _SceneProxy:
    """Proxy around the real InteractiveScene.

    When an observation term looks up a scene entity by name, this proxy lazily
    wraps any entity that has a ``.data`` attribute.  All tensor-returning
    properties on the data object are intercepted for LEAPP annotation.  This
    covers articulations, rigid objects, and sensors through both
    ``scene["name"]`` and ``scene.sensors["name"]`` access paths.
    """

    def __init__(self, real_scene, annotating_getters_by_type: dict[type, dict[str, Callable]], cache: dict):
        # use object.__setattr__ to avoid creating new attributes, only set the ones that are already defined
        object.__setattr__(self, "_real_scene", real_scene)
        object.__setattr__(self, "_annotating_getters_by_type", annotating_getters_by_type)
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(self, "_proxied", {})
        object.__setattr__(self, "_sensor_mapping_proxy", None)

    def _maybe_proxy_entity(self, key: str, entity: Any):
        """Return a proxy for any entity that has a ``.data`` attribute."""
        proxied = object.__getattribute__(self, "_proxied")
        if key in proxied:
            return proxied[key]

        data = getattr(entity, "data", None)
        if data is None:
            return entity

        annotating_getters_by_type = object.__getattribute__(self, "_annotating_getters_by_type")
        cache = object.__getattribute__(self, "_cache")
        data_proxy = _DataProxy(
            data,
            annotating_getters_by_type,
            cache,
            input_name_resolver=lambda prop_name, k=key: f"{k}_{prop_name}",
        )
        proxy = _EntityProxy(entity, data_proxy)
        proxied[key] = proxy
        return proxy

    def __getitem__(self, key):
        """Return a proxied entity when it exposes annotated data getters."""
        real_scene = object.__getattribute__(self, "_real_scene")
        entity = real_scene[key]
        return self._maybe_proxy_entity(key, entity)

    @property
    def sensors(self):
        """Return a mapping proxy for scene sensors."""
        sensor_mapping_proxy = object.__getattribute__(self, "_sensor_mapping_proxy")
        if sensor_mapping_proxy is None:
            real_scene = object.__getattribute__(self, "_real_scene")
            sensor_mapping_proxy = _EntityMappingProxy(
                real_scene.sensors,
                object.__getattribute__(self, "_annotating_getters_by_type"),
                object.__getattribute__(self, "_cache"),
            )
            object.__setattr__(self, "_sensor_mapping_proxy", sensor_mapping_proxy)
        return sensor_mapping_proxy

    def __getattr__(self, name):
        """Forward all other scene access to the real scene."""
        return getattr(object.__getattribute__(self, "_real_scene"), name)


class _EnvProxy:
    """Proxy around the real env that returns a _SceneProxy for ``.scene``.

    All other attribute access (``num_envs``, ``command_manager``, etc.)
    is forwarded transparently to the real env.
    """

    def __init__(self, real_env, scene_proxy: _SceneProxy):
        object.__setattr__(self, "_real_env", real_env)
        object.__setattr__(self, "_scene_proxy", scene_proxy)

    @property
    def scene(self):
        """Return the scene proxy instead of the real scene."""
        return object.__getattribute__(self, "_scene_proxy")

    def __getattr__(self, name):
        """Forward all non-scene attribute access to the real env."""
        return getattr(object.__getattribute__(self, "_real_env"), name)


def _build_scene_entity_lookup(real_scene) -> dict[int, tuple[str, str]]:
    """Map real scene entity object ids to their lookup path."""
    lookup: dict[int, tuple[str, str]] = {}
    for attr_name, attr_value in vars(real_scene).items():
        if not isinstance(attr_value, dict):
            continue
        container_kind = "sensors" if attr_name == "sensors" else "scene"
        for key, entity in attr_value.items():
            lookup[id(entity)] = (container_kind, key)
    return lookup


class _ManagerTermProxy(ManagerTermBase):
    """Proxy a class-based manager term while preserving its lifecycle methods.

    Observation manager terms can be stateful ``ManagerTermBase`` instances that
    expose ``reset()`` and ``serialize()`` in addition to being callable. This
    proxy preserves that interface while swapping the env argument passed into
    ``__call__`` for the observation-side proxy env.
    """

    def __init__(self, target: ManagerTermBase, proxy_env: _EnvProxy):
        super().__init__(target.cfg, target._env)
        self._target = target
        self._proxy_env = proxy_env
        self._entity_lookup = _build_scene_entity_lookup(target._env.scene)

    @property
    def __name__(self) -> str:
        """Expose the wrapped term name for compatibility and debugging."""
        return getattr(self._target, "__name__", self._target.__class__.__name__)

    def reset(self, env_ids=None) -> None:
        """Forward resets to the wrapped term instance."""
        self._target.reset(env_ids=env_ids)

    def serialize(self) -> dict:
        """Forward serialization to the wrapped term instance."""
        return self._target.serialize()

    def __call__(self, *args, **kwargs):
        """Call the wrapped term with the proxy env in place of the real env."""
        if args:
            args = (self._proxy_env, *args[1:])
        else:
            args = (self._proxy_env,)
        swapped_attrs: list[tuple[str, Any]] = []
        for attr_name, attr_value in vars(self._target).items():
            lookup = self._entity_lookup.get(id(attr_value))
            if lookup is None:
                continue

            container_kind, key = lookup
            proxy_entity = (
                self._proxy_env.scene.sensors[key] if container_kind == "sensors" else self._proxy_env.scene[key]
            )
            swapped_attrs.append((attr_name, attr_value))
            setattr(self._target, attr_name, proxy_entity)

        try:
            return self._target(*args, **kwargs)
        finally:
            for attr_name, attr_value in swapped_attrs:
                setattr(self._target, attr_name, attr_value)

    def __getattr__(self, name):
        """Forward all other attribute access to the wrapped term instance."""
        return getattr(self._target, name)


# ══════════════════════════════════════════════════════════════════
# Action-side proxy
# ══════════════════════════════════════════════════════════════════


class _ArticulationWriteProxy:
    """Proxy around a real Articulation for action terms.

    Intercepts ``_leapp_semantics``-decorated write methods **and** routes
    ``.data`` reads through a shared ``_DataProxy`` so that
    action-side state reads (e.g. ``self._asset.data.joint_pos`` inside
    ``RelativeJointPositionAction``) participate in LEAPP annotation and
    share the dedup cache with observation-side reads.

    All other attribute access is forwarded transparently to the real asset.
    """

    def __init__(
        self,
        real_asset: Articulation,
        term_name: str,
        output_cache: list[TensorSemantics],
        annotating_methods: dict[str, Callable],
        data_proxy: _DataProxy,
    ):
        object.__setattr__(self, "_real_asset", real_asset)
        object.__setattr__(self, "_term_name", term_name)
        object.__setattr__(self, "_output_cache", output_cache)
        object.__setattr__(self, "_annotating_methods", annotating_methods)
        object.__setattr__(self, "_data_proxy", data_proxy)

    @property
    def data(self):
        """Return the shared annotating data proxy."""
        return object.__getattribute__(self, "_data_proxy")

    def __getattr__(self, name):
        """Return an annotating wrapper for _leapp_semantics methods; forward everything else."""
        methods = object.__getattribute__(self, "_annotating_methods")
        if name in methods:
            real_asset = object.__getattribute__(self, "_real_asset")
            term_name = object.__getattribute__(self, "_term_name")
            output_cache = object.__getattribute__(self, "_output_cache")
            original_method = getattr(real_asset, name)
            return methods[name](real_asset, original_method, term_name, output_cache)
        return getattr(object.__getattribute__(self, "_real_asset"), name)
