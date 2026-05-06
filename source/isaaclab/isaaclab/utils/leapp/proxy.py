# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, cast

import torch
from leapp.utils.tensor_description import TensorSemantics

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.warp.proxy_array import ProxyArray

from .leapp_semantics import resolve_leapp_element_names
from .utils import TracedProxyArray, build_write_connection


def _resolve_annotated_property(
    property_resolution_cache: dict[tuple[type, str], tuple[Callable, Any] | None],
    real_data: Any,
    name: str,
) -> tuple[Callable, Any] | None:
    """Resolve a concrete property getter and inherited semantics metadata.

    The execution getter always comes from the concrete runtime class. Semantic
    metadata is resolved independently by walking the MRO until a property
    definition with ``_leapp_semantics`` is found. This mirrors the output-side
    export path, where semantics are authored on the base API while concrete
    backends provide the runtime implementation.
    """
    cache_key = (type(real_data), name)
    if cache_key in property_resolution_cache:
        return property_resolution_cache[cache_key]

    execution_prop = getattr(type(real_data), name, None)
    if not isinstance(execution_prop, property) or execution_prop.fget is None:
        property_resolution_cache[cache_key] = None
        return None

    semantics_meta = None
    for data_cls in type(real_data).__mro__:
        prop = data_cls.__dict__.get(name)
        if not isinstance(prop, property) or prop.fget is None:
            continue
        candidate = getattr(prop.fget, "_leapp_semantics", None)
        if candidate is None:
            continue
        if getattr(candidate, "const", False):
            property_resolution_cache[cache_key] = None
            return None
        semantics_meta = candidate
        break

    if semantics_meta is None:
        property_resolution_cache[cache_key] = None
        return None

    resolution = (execution_prop.fget, semantics_meta)
    property_resolution_cache[cache_key] = resolution
    return resolution


def _resolve_annotated_method(
    method_resolution_cache: dict[tuple[type, str], tuple[Callable, Any, inspect.Signature] | None],
    real_asset: Any,
    name: str,
) -> tuple[Callable, Any, inspect.Signature] | None:
    """Resolve a concrete bound method and inherited semantics metadata."""
    cache_key = (type(real_asset), name)
    if cache_key in method_resolution_cache:
        return method_resolution_cache[cache_key]

    original_method = getattr(real_asset, name, None)
    if not callable(original_method):
        method_resolution_cache[cache_key] = None
        return None

    for asset_cls in type(real_asset).__mro__:
        candidate = asset_cls.__dict__.get(name)
        if not callable(candidate):
            continue
        semantics_meta = getattr(candidate, "_leapp_semantics", None)
        if semantics_meta is None:
            continue
        resolution = (original_method, semantics_meta, inspect.signature(candidate))
        method_resolution_cache[cache_key] = resolution
        return resolution

    method_resolution_cache[cache_key] = None
    return None


class _WriteJointNameContext:
    """Resolve runtime joint-name subsets for lazy write interception."""

    __slots__ = ("joint_names", "_joint_ids")

    def __init__(self, joint_names: list[str], joint_ids):
        self.joint_names = joint_names
        self._joint_ids = joint_ids


def _unique_output_name(term_name: str, method_name: str, output_cache: list[TensorSemantics]) -> str:
    """Return a stable, unique output name for an action write entry."""
    existing = {t.name for t in output_cache}
    candidate = term_name
    if candidate in existing:
        candidate = f"{term_name}_{method_name}"
    suffix = 2
    while candidate in existing:
        candidate = f"{term_name}_{method_name}_{suffix}"
        suffix += 1
    return candidate


class _DataProxy:
    """Proxy around a real data object that intercepts tensor-returning property reads.

    The real data object may be any scene entity data class (``ArticulationData``,
    ``RigidObjectData``, sensor data classes, etc.). The proxy resolves property
    semantics lazily on first access by walking the runtime class MRO. This lets
    concrete backend overrides reuse semantic metadata authored on abstract base
    properties without copying decorators onto every implementation.

    When a semantic property returns a :class:`~isaaclab.utils.warp.ProxyArray`,
    the result is wrapped in a ``TracedProxyArray`` and cached for
    deduplication. Non-proxy results and ordinary attributes are forwarded
    transparently.

    All other attribute access is forwarded transparently to the real object.
    """

    def __init__(
        self,
        real_data: Any,
        entity_name: str,
        task_name: str,
        property_resolution_cache: dict[tuple[type, str], tuple[Callable, Any] | None],
        cache: dict,
        input_name_resolver: Callable,
    ):
        object.__setattr__(self, "_real_data", real_data)
        object.__setattr__(self, "_entity_name", entity_name)
        object.__setattr__(self, "_task_name", task_name)
        object.__setattr__(self, "_property_resolution_cache", property_resolution_cache)
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(self, "_input_name_resolver", input_name_resolver)

    def __getattr__(self, name):
        """Intercept semantic property reads; forward everything else."""
        real_data = object.__getattribute__(self, "_real_data")
        resolution = _resolve_annotated_property(
            object.__getattribute__(self, "_property_resolution_cache"), real_data, name
        )
        if resolution is None:
            return getattr(real_data, name)

        cache = object.__getattribute__(self, "_cache")
        cache_key = (id(real_data), name)
        if cache_key in cache:
            return cache[cache_key]

        execution_fget, semantics_meta = resolution
        result = execution_fget(real_data)
        if not isinstance(result, ProxyArray):
            return result

        input_name = object.__getattribute__(self, "_input_name_resolver")(name)
        traced = TracedProxyArray(
            result,
            input_name=input_name,
            semantics_meta=semantics_meta,
            real_data=real_data,
            entity_name=object.__getattribute__(self, "_entity_name"),
            property_name=name,
            task_name=object.__getattribute__(self, "_task_name"),
        )
        cache[cache_key] = traced
        return traced


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

    def __init__(
        self,
        real_mapping,
        task_name: str,
        property_resolution_cache: dict[tuple[type, str], tuple[Callable, Any] | None],
        cache: dict,
    ):
        object.__setattr__(self, "_real_mapping", real_mapping)
        object.__setattr__(self, "_task_name", task_name)
        object.__setattr__(self, "_property_resolution_cache", property_resolution_cache)
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
        data_proxy = _DataProxy(
            data,
            key,
            object.__getattribute__(self, "_task_name"),
            object.__getattribute__(self, "_property_resolution_cache"),
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

    def __init__(
        self,
        real_scene,
        task_name: str,
        property_resolution_cache: dict[tuple[type, str], tuple[Callable, Any] | None],
        cache: dict,
    ):
        object.__setattr__(self, "_real_scene", real_scene)
        object.__setattr__(self, "_task_name", task_name)
        object.__setattr__(self, "_property_resolution_cache", property_resolution_cache)
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

        cache = object.__getattribute__(self, "_cache")
        data_proxy = _DataProxy(
            data,
            key,
            object.__getattribute__(self, "_task_name"),
            object.__getattribute__(self, "_property_resolution_cache"),
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
                object.__getattribute__(self, "_task_name"),
                object.__getattribute__(self, "_property_resolution_cache"),
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

    def __init__(
        self,
        real_env,
        task_name: str,
        property_resolution_cache: dict[tuple[type, str], tuple[Callable, Any] | None],
        cache: dict,
    ):
        object.__setattr__(self, "_real_env", real_env)
        object.__setattr__(
            self,
            "_scene_proxy",
            _SceneProxy(real_env.scene, task_name, property_resolution_cache, cache),
        )

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
    """Proxy around a real articulation implementation for action terms.

    Intercepts ``_leapp_semantics``-decorated write methods **and** routes
    ``.data`` reads through a shared ``_DataProxy`` so that
    action-side state reads (e.g. ``self._asset.data.joint_pos`` inside
    ``RelativeJointPositionAction``) participate in LEAPP annotation and
    share the dedup cache with observation-side reads.

    All other attribute access is forwarded transparently to the real asset.
    """

    def __init__(
        self,
        real_asset: Any,
        entity_name: str,
        term_name: str,
        output_cache: list[TensorSemantics],
        method_resolution_cache: dict[tuple[type, str], tuple[Callable, Any, inspect.Signature] | None],
        captured_write_term_names: set[str],
        data_proxy: _DataProxy,
    ):
        object.__setattr__(self, "_real_asset", real_asset)
        object.__setattr__(self, "_entity_name", entity_name)
        object.__setattr__(self, "_term_name", term_name)
        object.__setattr__(self, "_output_cache", output_cache)
        object.__setattr__(self, "_method_resolution_cache", method_resolution_cache)
        object.__setattr__(self, "_captured_write_term_names", captured_write_term_names)
        object.__setattr__(self, "_data_proxy", data_proxy)

    @property
    def data(self):
        """Return the shared annotating data proxy."""
        return object.__getattribute__(self, "_data_proxy")

    def __getattr__(self, name):
        """Return an annotating wrapper for semantic write methods; forward everything else."""
        real_asset = object.__getattribute__(self, "_real_asset")
        resolution = _resolve_annotated_method(
            object.__getattribute__(self, "_method_resolution_cache"),
            real_asset,
            name,
        )
        if resolution is None:
            return getattr(real_asset, name)

        original_method, semantics_meta, signature = resolution
        term_name = object.__getattribute__(self, "_term_name")
        output_cache = object.__getattribute__(self, "_output_cache")
        captured_write_term_names = object.__getattribute__(self, "_captured_write_term_names")

        def interceptor(*args, **kwargs):
            result = original_method(*args, **kwargs)
            bound_args = signature.bind_partial(real_asset, *args, **kwargs)
            target = bound_args.arguments.get("target")

            if not isinstance(target, torch.Tensor):
                return result

            target_tensor = cast(torch.Tensor, target)
            joint_ids = bound_args.arguments.get("joint_ids")
            output_cache.append(
                TensorSemantics(
                    name=_unique_output_name(term_name, name, output_cache),
                    ref=target_tensor.clone(),
                    kind=semantics_meta.kind,
                    element_names=resolve_leapp_element_names(
                        semantics_meta,
                        _WriteJointNameContext(real_asset.joint_names, joint_ids),
                    ),
                    extra=build_write_connection(
                        object.__getattribute__(self, "_entity_name"),
                        name,
                    ),
                )
            )
            captured_write_term_names.add(term_name)

            return result

        return interceptor
