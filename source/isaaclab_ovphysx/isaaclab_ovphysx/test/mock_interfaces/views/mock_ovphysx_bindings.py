# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock implementations of ovphysx TensorBinding objects for unit testing."""

from __future__ import annotations

from typing import Literal

import numpy as np

from isaaclab_ovphysx import tensor_types as TT


class MockTensorBinding:
    """Mock of ovphysx.TensorBinding that stores data in numpy arrays.

    Mimics the real TensorBinding API: ``read(tensor)`` fills the tensor from
    the internal buffer, ``write(tensor, indices, mask)`` copies into it.
    """

    def __init__(
        self,
        tensor_type: int,
        shape: tuple[int, ...],
        count: int,
        dof_count: int = 0,
        body_count: int = 0,
        joint_count: int = 0,
        is_fixed_base: bool = False,
        dof_names: list[str] | None = None,
        body_names: list[str] | None = None,
        joint_names: list[str] | None = None,
        fixed_tendon_count: int = 0,
        spatial_tendon_count: int = 0,
        write_only: bool = False,
    ):
        self.tensor_type = tensor_type
        self._shape = shape
        self._count = count
        self._dof_count = dof_count
        self._body_count = body_count
        self._joint_count = joint_count
        self._is_fixed_base = is_fixed_base
        self._dof_names = dof_names or []
        self._body_names = body_names or []
        self._joint_names = joint_names or []
        self._fixed_tendon_count = fixed_tendon_count
        self._spatial_tendon_count = spatial_tendon_count
        self._write_only = write_only
        self._data = np.zeros(shape, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def count(self) -> int:
        return self._count

    @property
    def dof_count(self) -> int:
        return self._dof_count

    @property
    def body_count(self) -> int:
        return self._body_count

    @property
    def joint_count(self) -> int:
        return self._joint_count

    @property
    def is_fixed_base(self) -> bool:
        return self._is_fixed_base

    @property
    def dof_names(self) -> list[str]:
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        return self._body_names

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def fixed_tendon_count(self) -> int:
        return self._fixed_tendon_count

    @property
    def spatial_tendon_count(self) -> int:
        return self._spatial_tendon_count

    def read(self, tensor) -> None:
        """Copy internal data into the provided array (numpy or warp)."""
        if self._write_only:
            raise RuntimeError("write-only tensor binding does not support read()")
        try:
            import warp as wp

            if isinstance(tensor, wp.array):
                tmp = wp.from_numpy(self._data, dtype=wp.float32, device=tensor.device)
                wp.copy(tensor, tmp)
                return
        except ImportError:
            pass
        np_dst = np.asarray(tensor)
        np.copyto(np_dst, self._data.reshape(np_dst.shape))

    @staticmethod
    def _to_numpy(arr) -> np.ndarray:
        """Convert warp/torch/numpy array to numpy, handling GPU arrays."""
        try:
            import warp as wp

            if isinstance(arr, wp.array):
                return arr.numpy()
        except ImportError:
            pass
        try:
            import torch

            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy()
        except ImportError:
            pass
        return np.asarray(arr)

    def write(self, tensor, indices=None, mask=None) -> None:
        """Copy from the provided array (numpy or warp) into internal data."""
        np_src = self._to_numpy(tensor).astype(np.float32)
        if indices is not None:
            idx = self._to_numpy(indices)
            if np_src.shape[0] == self._data.shape[0]:
                self._data[idx] = np_src[idx]
            else:
                self._data[idx] = np_src.reshape(len(idx), *self._data.shape[1:])
        elif mask is not None:
            np_mask = self._to_numpy(mask).astype(bool)
            self._data[np_mask] = np_src[np_mask]
        else:
            np.copyto(self._data, np_src.reshape(self._data.shape))

    def destroy(self) -> None:
        pass

    def set_random_data(self, low: float = -1.0, high: float = 1.0) -> None:
        """Fill internal buffer with random data."""
        self._data = np.random.uniform(low, high, self._shape).astype(np.float32)


class MockOvPhysxView:
    """Mock of :class:`~isaaclab_ovphysx.sim.views.OvPhysxView` over a dict of
    :class:`MockTensorBinding`.

    Lets a unit test inject a working ``_root_view`` into an OVPhysX asset/data class
    without standing up the real view, a real ``PhysX``, or a USD stage. It mirrors the
    consumed surface of ``OvPhysxView`` -- ``binding_for`` / ``try_binding_for``,
    ``get_attribute`` / ``read_into`` / ``set_attribute``, the discoverability helpers,
    and the metadata passthrough -- delegating reads/writes to the mock bindings, which
    already implement ``read``/``write`` against numpy buffers.

    Names resolve like the real view: pass either a :class:`TensorType` member or its
    lowercased name (e.g. ``"articulation_dof_stiffness"``). Unlike the real view this
    keeps everything on CPU and applies no device/dtype/read-only guards -- it is a test
    double for the binding-routing surface, not the device policy.
    """

    def __init__(self, bindings: dict[int, MockTensorBinding]):
        self._bindings = bindings

    def _resolve(self, name):
        """Resolve a TensorType member or its lowercased name to the member."""
        if isinstance(name, str):
            enum_cls = type(next(iter(self._bindings)))
            try:
                return enum_cls[name.upper()]
            except KeyError:
                raise KeyError(f"Unknown attribute {name!r}") from None
        return name

    def _sample(self) -> MockTensorBinding:
        """A representative binding to read shared metadata from."""
        return next(iter(self._bindings.values()))

    # -- core read / write -----------------------------------------------------

    def get_attribute(self, name, *, out=None):
        """Read the full attribute tensor; fill ``out`` if given, else allocate float32."""
        binding = self._bindings[self._resolve(name)]
        if out is not None:
            binding.read(out)
            return out
        import warp as wp

        buf = wp.zeros(tuple(binding.shape), dtype=wp.float32, device="cpu")
        binding.read(buf)
        return buf

    def read_into(self, name, dst) -> None:
        """Fill ``dst`` in place from the attribute binding."""
        self._bindings[self._resolve(name)].read(dst)

    def set_attribute(self, name, values, *, indices=None, mask=None) -> None:
        """Write a full attribute tensor; ``indices``/``mask`` select which rows apply."""
        self._bindings[self._resolve(name)].write(values, indices=indices, mask=mask)

    # -- raw binding access ----------------------------------------------------

    def binding_for(self, name) -> MockTensorBinding:
        """Return the underlying binding, raising ``KeyError`` if absent for these prims."""
        return self._bindings[self._resolve(name)]

    def try_binding_for(self, name) -> MockTensorBinding | None:
        """Like :meth:`binding_for`, but return ``None`` when the attribute is absent."""
        return self._bindings.get(self._resolve(name))

    # -- discoverability -------------------------------------------------------

    def has_attribute(self, name) -> bool:
        """Return whether a binding is instantiated for ``name`` on these prims."""
        return self._resolve(name) in self._bindings

    def __contains__(self, name) -> bool:
        return self.has_attribute(name)

    @property
    def available_attributes(self) -> list[str]:
        """Names with a live binding (lowercased ``TensorType`` members)."""
        return sorted(tt.name.lower() for tt in self._bindings)

    # -- metadata passthrough (from a sample binding) --------------------------

    @property
    def count(self) -> int:
        return self._sample().count

    @property
    def dof_count(self) -> int:
        return self._sample().dof_count

    @property
    def body_count(self) -> int:
        return self._sample().body_count

    @property
    def joint_count(self) -> int:
        return self._sample().joint_count

    @property
    def is_fixed_base(self) -> bool:
        return self._sample().is_fixed_base

    @property
    def dof_names(self) -> list[str]:
        return list(self._sample().dof_names)

    @property
    def body_names(self) -> list[str]:
        return list(self._sample().body_names)

    @property
    def joint_names(self) -> list[str]:
        return list(self._sample().joint_names)

    @property
    def fixed_tendon_count(self) -> int:
        return self._sample().fixed_tendon_count

    @property
    def spatial_tendon_count(self) -> int:
        return self._sample().spatial_tendon_count


class MockOvPhysxBindingSet:
    """Factory that creates a full set of MockTensorBinding objects
    for a given articulation configuration.

    Mirrors the tensor types that ``Articulation._initialize_impl`` creates.

    With ``asset_kind='rigid_object'`` it produces the smaller set
    consumed by ``RigidObject._initialize_impl``: ``RIGID_BODY_*`` only,
    ``num_joints`` must be 0, ``num_bodies`` must be 1, no tendons.
    """

    def __init__(
        self,
        num_instances: int,
        num_joints: int,
        num_bodies: int,
        is_fixed_base: bool = False,
        joint_names: list[str] | None = None,
        body_names: list[str] | None = None,
        num_fixed_tendons: int = 0,
        num_spatial_tendons: int = 0,
        *,
        asset_kind: Literal["articulation", "rigid_object"] = "articulation",
    ):
        if asset_kind == "rigid_object":
            if num_joints != 0 or num_bodies != 1 or num_fixed_tendons != 0 or num_spatial_tendons != 0:
                raise ValueError(
                    "asset_kind='rigid_object' requires num_joints=0, num_bodies=1, "
                    "num_fixed_tendons=0, num_spatial_tendons=0; got "
                    f"num_joints={num_joints}, num_bodies={num_bodies}, "
                    f"num_fixed_tendons={num_fixed_tendons}, "
                    f"num_spatial_tendons={num_spatial_tendons}"
                )
            N = num_instances
            if body_names is None:
                body_names = ["base_link"]
            common = dict(
                count=N,
                dof_count=0,
                body_count=1,
                joint_count=0,
                is_fixed_base=is_fixed_base,
                dof_names=[],
                body_names=body_names,
                joint_names=[],
                fixed_tendon_count=0,
                spatial_tendon_count=0,
            )
            self.bindings: dict[int, MockTensorBinding] = {
                TT.RIGID_BODY_POSE: MockTensorBinding(TT.RIGID_BODY_POSE, (N, 7), **common),
                TT.RIGID_BODY_VELOCITY: MockTensorBinding(TT.RIGID_BODY_VELOCITY, (N, 6), **common),
                TT.RIGID_BODY_WRENCH: MockTensorBinding(TT.RIGID_BODY_WRENCH, (N, 9), write_only=True, **common),
                TT.RIGID_BODY_MASS: MockTensorBinding(TT.RIGID_BODY_MASS, (N,), **common),
                TT.RIGID_BODY_COM_POSE: MockTensorBinding(TT.RIGID_BODY_COM_POSE, (N, 7), **common),
                TT.RIGID_BODY_INERTIA: MockTensorBinding(TT.RIGID_BODY_INERTIA, (N, 9), **common),
            }
            # Optional bindings: only present when the wheel exposes the alias.
            if hasattr(TT, "RIGID_BODY_ACCELERATION"):
                self.bindings[TT.RIGID_BODY_ACCELERATION] = MockTensorBinding(
                    TT.RIGID_BODY_ACCELERATION, (N, 6), **common
                )
            if hasattr(TT, "RIGID_BODY_INV_MASS"):
                self.bindings[TT.RIGID_BODY_INV_MASS] = MockTensorBinding(TT.RIGID_BODY_INV_MASS, (N,), **common)
            if hasattr(TT, "RIGID_BODY_INV_INERTIA"):
                self.bindings[TT.RIGID_BODY_INV_INERTIA] = MockTensorBinding(
                    TT.RIGID_BODY_INV_INERTIA, (N, 9), **common
                )
            return

        N = num_instances
        D = num_joints
        L = num_bodies
        T_fix = num_fixed_tendons
        T_spa = num_spatial_tendons

        if joint_names is None:
            joint_names = [f"joint_{i}" for i in range(D)]
        if body_names is None:
            body_names = [f"body_{i}" for i in range(L)]

        common = dict(
            count=N,
            dof_count=D,
            body_count=L,
            joint_count=D,
            is_fixed_base=is_fixed_base,
            dof_names=joint_names,
            body_names=body_names,
            joint_names=joint_names,
            fixed_tendon_count=T_fix,
            spatial_tendon_count=T_spa,
        )

        self.bindings: dict[int, MockTensorBinding] = {
            TT.ROOT_POSE: MockTensorBinding(TT.ROOT_POSE, (N, 7), **common),
            TT.ROOT_VELOCITY: MockTensorBinding(TT.ROOT_VELOCITY, (N, 6), **common),
            TT.LINK_POSE: MockTensorBinding(TT.LINK_POSE, (N, L, 7), **common),
            TT.LINK_VELOCITY: MockTensorBinding(TT.LINK_VELOCITY, (N, L, 6), **common),
            TT.LINK_ACCELERATION: MockTensorBinding(TT.LINK_ACCELERATION, (N, L, 6), **common),
            TT.DOF_POSITION: MockTensorBinding(TT.DOF_POSITION, (N, D), **common),
            TT.DOF_VELOCITY: MockTensorBinding(TT.DOF_VELOCITY, (N, D), **common),
            TT.DOF_POSITION_TARGET: MockTensorBinding(TT.DOF_POSITION_TARGET, (N, D), **common),
            TT.DOF_VELOCITY_TARGET: MockTensorBinding(TT.DOF_VELOCITY_TARGET, (N, D), **common),
            TT.DOF_ACTUATION_FORCE: MockTensorBinding(TT.DOF_ACTUATION_FORCE, (N, D), **common),
            TT.DOF_STIFFNESS: MockTensorBinding(TT.DOF_STIFFNESS, (N, D), **common),
            TT.DOF_DAMPING: MockTensorBinding(TT.DOF_DAMPING, (N, D), **common),
            TT.DOF_LIMIT: MockTensorBinding(TT.DOF_LIMIT, (N, D, 2), **common),
            TT.DOF_MAX_VELOCITY: MockTensorBinding(TT.DOF_MAX_VELOCITY, (N, D), **common),
            TT.DOF_MAX_FORCE: MockTensorBinding(TT.DOF_MAX_FORCE, (N, D), **common),
            TT.DOF_ARMATURE: MockTensorBinding(TT.DOF_ARMATURE, (N, D), **common),
            TT.DOF_FRICTION_PROPERTIES: MockTensorBinding(TT.DOF_FRICTION_PROPERTIES, (N, D, 3), **common),
            TT.LINK_WRENCH: MockTensorBinding(TT.LINK_WRENCH, (N, L, 9), write_only=True, **common),
            TT.BODY_MASS: MockTensorBinding(TT.BODY_MASS, (N, L), **common),
            TT.BODY_COM_POSE: MockTensorBinding(TT.BODY_COM_POSE, (N, L, 7), **common),
            TT.BODY_INERTIA: MockTensorBinding(TT.BODY_INERTIA, (N, L, 9), **common),
            TT.BODY_INV_MASS: MockTensorBinding(TT.BODY_INV_MASS, (N, L), **common),
            TT.BODY_INV_INERTIA: MockTensorBinding(TT.BODY_INV_INERTIA, (N, L, 9), **common),
            TT.LINK_INCOMING_JOINT_FORCE: MockTensorBinding(TT.LINK_INCOMING_JOINT_FORCE, (N, L, 6), **common),
            TT.DOF_PROJECTED_JOINT_FORCE: MockTensorBinding(TT.DOF_PROJECTED_JOINT_FORCE, (N, D), **common),
        }

        # Fixed tendon bindings (only when tendons are present)
        if T_fix > 0:
            self.bindings.update(
                {
                    TT.FIXED_TENDON_STIFFNESS: MockTensorBinding(TT.FIXED_TENDON_STIFFNESS, (N, T_fix), **common),
                    TT.FIXED_TENDON_DAMPING: MockTensorBinding(TT.FIXED_TENDON_DAMPING, (N, T_fix), **common),
                    TT.FIXED_TENDON_LIMIT_STIFFNESS: MockTensorBinding(
                        TT.FIXED_TENDON_LIMIT_STIFFNESS, (N, T_fix), **common
                    ),
                    TT.FIXED_TENDON_LIMIT: MockTensorBinding(TT.FIXED_TENDON_LIMIT, (N, T_fix, 2), **common),
                    TT.FIXED_TENDON_REST_LENGTH: MockTensorBinding(TT.FIXED_TENDON_REST_LENGTH, (N, T_fix), **common),
                    TT.FIXED_TENDON_OFFSET: MockTensorBinding(TT.FIXED_TENDON_OFFSET, (N, T_fix), **common),
                }
            )

        # Spatial tendon bindings
        if T_spa > 0:
            self.bindings.update(
                {
                    TT.SPATIAL_TENDON_STIFFNESS: MockTensorBinding(TT.SPATIAL_TENDON_STIFFNESS, (N, T_spa), **common),
                    TT.SPATIAL_TENDON_DAMPING: MockTensorBinding(TT.SPATIAL_TENDON_DAMPING, (N, T_spa), **common),
                    TT.SPATIAL_TENDON_LIMIT_STIFFNESS: MockTensorBinding(
                        TT.SPATIAL_TENDON_LIMIT_STIFFNESS, (N, T_spa), **common
                    ),
                    TT.SPATIAL_TENDON_OFFSET: MockTensorBinding(TT.SPATIAL_TENDON_OFFSET, (N, T_spa), **common),
                }
            )

    @property
    def view(self) -> MockOvPhysxView:
        """A mock :class:`OvPhysxView` over this set's bindings.

        Inject as an asset's ``_root_view`` to exercise the migrated binding-routing
        code paths without a real view or ``PhysX``. Cached so repeated access returns
        the same object, like the single view an asset holds.
        """
        v = getattr(self, "_view", None)
        if v is None:
            v = MockOvPhysxView(self.bindings)
            self._view = v
        return v

    def set_random_data(self) -> None:
        """Fill all bindings with random data."""
        for b in self.bindings.values():
            if not b._write_only:
                b.set_random_data()
        if TT.DOF_LIMIT in self.bindings:
            lim = self.bindings[TT.DOF_LIMIT]
            lim._data[..., 0] = -3.14
            lim._data[..., 1] = 3.14
        pose_keys = [
            k
            for k in (TT.ROOT_POSE, TT.LINK_POSE, TT.BODY_COM_POSE, TT.RIGID_BODY_POSE, TT.RIGID_BODY_COM_POSE)
            if k in self.bindings
        ]
        for tt in pose_keys:
            b = self.bindings[tt]
            b._data[..., 3:6] = 0.0
            b._data[..., 6] = 1.0
        for mass_key in (TT.BODY_MASS, TT.RIGID_BODY_MASS):
            if mass_key in self.bindings:
                self.bindings[mass_key]._data = np.abs(self.bindings[mass_key]._data) + 0.1
        for max_key in (TT.DOF_MAX_VELOCITY, TT.DOF_MAX_FORCE):
            if max_key in self.bindings:
                self.bindings[max_key]._data = np.abs(self.bindings[max_key]._data) + 1.0
        if TT.FIXED_TENDON_LIMIT in self.bindings:
            tlim = self.bindings[TT.FIXED_TENDON_LIMIT]
            tlim._data[..., 0] = -1.0
            tlim._data[..., 1] = 1.0
