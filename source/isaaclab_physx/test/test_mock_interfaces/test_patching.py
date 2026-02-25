# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for patching utilities."""

from isaaclab_physx.test.mock_interfaces.utils import (
    mock_articulation_view,
    mock_rigid_body_view,
    mock_rigid_contact_view,
    patch_articulation_view,
    patch_rigid_body_view,
    patch_rigid_contact_view,
)


class TestPatchRigidBodyView:
    """Tests for patch_rigid_body_view context manager."""

    def test_basic_patching(self):
        """Test basic patching functionality.

        Note: Patching works when the target module imports the class dynamically.
        In tests where we import at module level, we need to import inside the
        context to get the patched version.
        """
        target = "isaaclab_physx.test.mock_interfaces.views.mock_rigid_body_view.MockRigidBodyView"
        with patch_rigid_body_view(target, count=4):
            # Import inside context to get patched version
            from isaaclab_physx.test.mock_interfaces.views import mock_rigid_body_view

            view = mock_rigid_body_view.MockRigidBodyView()
            assert view.count == 4

    def test_patching_preserves_configuration(self):
        """Test that patching preserves configuration."""
        target = "isaaclab_physx.test.mock_interfaces.views.mock_rigid_body_view.MockRigidBodyView"
        prim_paths = ["/World/A", "/World/B", "/World/C", "/World/D"]
        with patch_rigid_body_view(target, count=4, prim_paths=prim_paths):
            from isaaclab_physx.test.mock_interfaces.views import mock_rigid_body_view

            view = mock_rigid_body_view.MockRigidBodyView()
            assert view.prim_paths == prim_paths


class TestPatchArticulationView:
    """Tests for patch_articulation_view context manager."""

    def test_basic_patching(self):
        """Test basic patching functionality."""
        target = "isaaclab_physx.test.mock_interfaces.views.mock_articulation_view.MockArticulationView"
        with patch_articulation_view(target, count=4, num_dofs=12, num_links=13):
            from isaaclab_physx.test.mock_interfaces.views import mock_articulation_view

            view = mock_articulation_view.MockArticulationView()
            assert view.count == 4
            assert view.shared_metatype.dof_count == 12
            assert view.shared_metatype.link_count == 13

    def test_patching_with_names(self):
        """Test patching with custom names."""
        target = "isaaclab_physx.test.mock_interfaces.views.mock_articulation_view.MockArticulationView"
        dof_names = ["joint_a", "joint_b"]
        with patch_articulation_view(target, num_dofs=2, dof_names=dof_names):
            from isaaclab_physx.test.mock_interfaces.views import mock_articulation_view

            view = mock_articulation_view.MockArticulationView()
            assert view.shared_metatype.dof_names == dof_names


class TestPatchRigidContactView:
    """Tests for patch_rigid_contact_view context manager."""

    def test_basic_patching(self):
        """Test basic patching functionality."""
        target = "isaaclab_physx.test.mock_interfaces.views.mock_rigid_contact_view.MockRigidContactView"
        with patch_rigid_contact_view(target, count=4, num_bodies=5, filter_count=3):
            from isaaclab_physx.test.mock_interfaces.views import mock_rigid_contact_view

            view = mock_rigid_contact_view.MockRigidContactView()
            assert view.count == 4
            assert view.num_bodies == 5
            assert view.filter_count == 3


class TestDecoratorUtilities:
    """Tests for decorator utilities.

    Note: These decorators are designed for wrapping functions, not pytest tests.
    Pytest inspects function signatures for fixtures, which conflicts with our
    decorator pattern. We test them by manually invoking the decorated functions.
    """

    def test_mock_rigid_body_view_decorator(self):
        """Test mock_rigid_body_view decorator injects mock view."""

        @mock_rigid_body_view(count=4)
        def my_function(view):
            return view.count, view.get_transforms().shape

        count, shape = my_function()
        assert count == 4
        assert shape == (4, 7)

    def test_mock_articulation_view_decorator(self):
        """Test mock_articulation_view decorator injects mock view."""

        @mock_articulation_view(count=4, num_dofs=12, num_links=13)
        def my_function(view):
            return (
                view.count,
                view.shared_metatype.dof_count,
                view.get_dof_positions().shape,
            )

        count, dof_count, shape = my_function()
        assert count == 4
        assert dof_count == 12
        assert shape == (4, 12)

    def test_mock_rigid_contact_view_decorator(self):
        """Test mock_rigid_contact_view decorator injects mock view."""

        @mock_rigid_contact_view(count=4, num_bodies=5, filter_count=3)
        def my_function(view):
            return (
                view.count,
                view.num_bodies,
                view.get_net_contact_forces(0.01).shape,
            )

        count, num_bodies, shape = my_function()
        assert count == 4
        assert num_bodies == 5
        assert shape == (20, 3)  # 4 * 5 = 20

    def test_decorator_with_prim_paths(self):
        """Test decorator with custom prim paths."""

        @mock_rigid_body_view(count=2, prim_paths=["/World/A", "/World/B"])
        def my_function(view):
            return view.prim_paths

        paths = my_function()
        assert paths == ["/World/A", "/World/B"]

    def test_decorator_with_fixed_base(self):
        """Test decorator with fixed base."""

        @mock_articulation_view(count=2, num_dofs=6, fixed_base=True)
        def my_function(view):
            return view.shared_metatype.fixed_base

        fixed_base = my_function()
        assert fixed_base is True


class TestDecoratorFunctionPreservation:
    """Tests that decorators preserve function metadata."""

    def test_function_name_preserved(self):
        """Test that function name is preserved."""

        @mock_rigid_body_view(count=4)
        def my_documented_function(view):
            """This is a documented function."""
            pass

        assert my_documented_function.__name__ == "my_documented_function"

    def test_docstring_preserved(self):
        """Test that docstring is preserved."""

        @mock_rigid_body_view(count=4)
        def my_documented_function(view):
            """This is a documented function."""
            pass

        assert "documented function" in my_documented_function.__doc__
