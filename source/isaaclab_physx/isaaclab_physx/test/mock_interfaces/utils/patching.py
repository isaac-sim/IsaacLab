# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for patching PhysX views with mock implementations."""

from __future__ import annotations

import functools
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, TypeVar
from unittest.mock import patch

F = TypeVar("F", bound=Callable[..., Any])


@contextmanager
def patch_rigid_body_view(
    target: str,
    count: int = 1,
    prim_paths: list[str] | None = None,
    device: str = "cpu",
) -> Generator[Any, None, None]:
    """Context manager for patching physx.RigidBodyView with a mock.

    Args:
        target: The target to patch (e.g., "my_module.physx.RigidBodyView").
        count: Number of rigid body instances.
        prim_paths: USD prim paths for each instance.
        device: Device for tensor allocation.

    Yields:
        The mock class that will be used when the target is instantiated.

    Example:
        >>> with patch_rigid_body_view("my_module.view", count=4) as mock:
        ...     view = my_function()
        ...     view.set_mock_transforms(torch.randn(4, 7))
    """
    from ..views import MockRigidBodyView

    def create_mock(*args, **kwargs):
        return MockRigidBodyView(
            count=kwargs.get("count", count),
            prim_paths=kwargs.get("prim_paths", prim_paths),
            device=kwargs.get("device", device),
        )

    with patch(target, side_effect=create_mock) as mock_class:
        yield mock_class


@contextmanager
def patch_articulation_view(
    target: str,
    count: int = 1,
    num_dofs: int = 1,
    num_links: int = 2,
    dof_names: list[str] | None = None,
    link_names: list[str] | None = None,
    fixed_base: bool = False,
    prim_paths: list[str] | None = None,
    device: str = "cpu",
) -> Generator[Any, None, None]:
    """Context manager for patching physx.ArticulationView with a mock.

    Args:
        target: The target to patch (e.g., "my_module.physx.ArticulationView").
        count: Number of articulation instances.
        num_dofs: Number of degrees of freedom (joints).
        num_links: Number of links (bodies).
        dof_names: Names of the DOFs.
        link_names: Names of the links.
        fixed_base: Whether the articulation has a fixed base.
        prim_paths: USD prim paths for each instance.
        device: Device for tensor allocation.

    Yields:
        The mock class that will be used when the target is instantiated.

    Example:
        >>> with patch_articulation_view("my_module.view", num_dofs=12) as mock:
        ...     view = my_function()
        ...     positions = view.get_dof_positions()
    """
    from ..views import MockArticulationView

    def create_mock(*args, **kwargs):
        return MockArticulationView(
            count=kwargs.get("count", count),
            num_dofs=kwargs.get("num_dofs", num_dofs),
            num_links=kwargs.get("num_links", num_links),
            dof_names=kwargs.get("dof_names", dof_names),
            link_names=kwargs.get("link_names", link_names),
            fixed_base=kwargs.get("fixed_base", fixed_base),
            prim_paths=kwargs.get("prim_paths", prim_paths),
            device=kwargs.get("device", device),
        )

    with patch(target, side_effect=create_mock) as mock_class:
        yield mock_class


@contextmanager
def patch_rigid_contact_view(
    target: str,
    count: int = 1,
    num_bodies: int = 1,
    filter_count: int = 0,
    max_contact_data_count: int = 16,
    device: str = "cpu",
) -> Generator[Any, None, None]:
    """Context manager for patching physx.RigidContactView with a mock.

    Args:
        target: The target to patch (e.g., "my_module.physx.RigidContactView").
        count: Number of instances.
        num_bodies: Number of bodies per instance.
        filter_count: Number of filter bodies for contact filtering.
        max_contact_data_count: Maximum number of contact data points.
        device: Device for tensor allocation.

    Yields:
        The mock class that will be used when the target is instantiated.

    Example:
        >>> with patch_rigid_contact_view("my_module.view", num_bodies=4) as mock:
        ...     view = my_function()
        ...     forces = view.get_net_contact_forces(0.01)
    """
    from ..views import MockRigidContactView

    def create_mock(*args, **kwargs):
        return MockRigidContactView(
            count=kwargs.get("count", count),
            num_bodies=kwargs.get("num_bodies", num_bodies),
            filter_count=kwargs.get("filter_count", filter_count),
            max_contact_data_count=kwargs.get("max_contact_data_count", max_contact_data_count),
            device=kwargs.get("device", device),
        )

    with patch(target, side_effect=create_mock) as mock_class:
        yield mock_class


# -- Decorators --


def mock_rigid_body_view(
    count: int = 1,
    prim_paths: list[str] | None = None,
    device: str = "cpu",
) -> Callable[[F], F]:
    """Decorator for injecting MockRigidBodyView into test function.

    The mock view is passed as the first argument to the decorated function.

    Args:
        count: Number of rigid body instances.
        prim_paths: USD prim paths for each instance.
        device: Device for tensor allocation.

    Returns:
        A decorator function.

    Example:
        >>> @mock_rigid_body_view(count=4)
        ... def test_my_function(mock_view):
        ...     transforms = mock_view.get_transforms()
        ...     assert transforms.shape == (4, 7)
    """
    from ..views import MockRigidBodyView

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mock = MockRigidBodyView(count=count, prim_paths=prim_paths, device=device)
            return func(mock, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def mock_articulation_view(
    count: int = 1,
    num_dofs: int = 1,
    num_links: int = 2,
    dof_names: list[str] | None = None,
    link_names: list[str] | None = None,
    fixed_base: bool = False,
    prim_paths: list[str] | None = None,
    device: str = "cpu",
) -> Callable[[F], F]:
    """Decorator for injecting MockArticulationView into test function.

    The mock view is passed as the first argument to the decorated function.

    Args:
        count: Number of articulation instances.
        num_dofs: Number of degrees of freedom (joints).
        num_links: Number of links (bodies).
        dof_names: Names of the DOFs.
        link_names: Names of the links.
        fixed_base: Whether the articulation has a fixed base.
        prim_paths: USD prim paths for each instance.
        device: Device for tensor allocation.

    Returns:
        A decorator function.

    Example:
        >>> @mock_articulation_view(count=4, num_dofs=12, num_links=13)
        ... def test_my_function(mock_view):
        ...     positions = mock_view.get_dof_positions()
        ...     assert positions.shape == (4, 12)
    """
    from ..views import MockArticulationView

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mock = MockArticulationView(
                count=count,
                num_dofs=num_dofs,
                num_links=num_links,
                dof_names=dof_names,
                link_names=link_names,
                fixed_base=fixed_base,
                prim_paths=prim_paths,
                device=device,
            )
            return func(mock, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def mock_rigid_contact_view(
    count: int = 1,
    num_bodies: int = 1,
    filter_count: int = 0,
    max_contact_data_count: int = 16,
    device: str = "cpu",
) -> Callable[[F], F]:
    """Decorator for injecting MockRigidContactView into test function.

    The mock view is passed as the first argument to the decorated function.

    Args:
        count: Number of instances.
        num_bodies: Number of bodies per instance.
        filter_count: Number of filter bodies for contact filtering.
        max_contact_data_count: Maximum number of contact data points.
        device: Device for tensor allocation.

    Returns:
        A decorator function.

    Example:
        >>> @mock_rigid_contact_view(count=4, num_bodies=4, filter_count=2)
        ... def test_my_function(mock_view):
        ...     forces = mock_view.get_net_contact_forces(0.01)
        ...     assert forces.shape == (16, 3)
    """
    from ..views import MockRigidContactView

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mock = MockRigidContactView(
                count=count,
                num_bodies=num_bodies,
                filter_count=filter_count,
                max_contact_data_count=max_contact_data_count,
                device=device,
            )
            return func(mock, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
