# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for stage utilities."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import tempfile
from pathlib import Path

import pytest

from pxr import Usd

import isaaclab.sim as sim_utils


def test_create_new_stage():
    """Test creating a new stage attached to USD context."""
    stage = sim_utils.create_new_stage()

    # Should return a valid stage
    assert stage is not None
    assert isinstance(stage, Usd.Stage)

    # Stage should be the current stage
    current_stage = sim_utils.get_current_stage()
    assert stage == current_stage

    # Stage should have a root prim
    root_prim = stage.GetPseudoRoot()
    assert root_prim.IsValid()


def test_create_multiple_stages():
    """Test creating multiple stages."""
    stage1 = sim_utils.create_new_stage()
    stage2 = sim_utils.create_new_stage()
    stage3 = sim_utils.create_new_stage()

    assert stage1 is not None
    assert stage2 is not None
    assert stage3 is not None
    assert stage1 != stage2
    assert stage1 != stage3
    assert stage2 != stage3


def test_create_new_stage_in_memory():
    """Test creating a new stage in memory (Isaac Sim 5.0+)."""
    stage = sim_utils.create_new_stage()

    # Should return a valid stage
    assert stage is not None
    assert isinstance(stage, Usd.Stage)

    # Stage should have a root prim
    root_prim = stage.GetPseudoRoot()
    assert root_prim.IsValid()


def test_is_current_stage_in_memory():
    """Test checking if current stage is in memory."""
    # Create a stage - in kitless mode, this creates an in-memory stage
    sim_utils.create_new_stage()
    is_in_memory = sim_utils.is_current_stage_in_memory()

    # Should return a boolean
    assert isinstance(is_in_memory, bool)
    # With kitless mode support, create_new_stage() creates an in-memory stage
    assert is_in_memory is True

    # Create a stage in memory explicitly
    stage = sim_utils.create_new_stage()
    with sim_utils.use_stage(stage):
        is_in_memory = sim_utils.is_current_stage_in_memory()
        assert isinstance(is_in_memory, bool)
        assert is_in_memory is True


def test_save_and_open_stage():
    """Test saving and opening a stage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a stage with some content
        stage = sim_utils.create_new_stage()
        stage.DefinePrim("/World", "Xform")
        stage.DefinePrim("/World/TestCube", "Cube")

        # Save the stage
        save_path = Path(temp_dir) / "test_stage.usd"
        result = sim_utils.save_stage(str(save_path), save_and_reload_in_place=False)

        # Save should succeed
        assert result is True
        assert save_path.exists()

        # Open the saved stage
        opened_stage = sim_utils.open_stage(str(save_path))
        assert isinstance(opened_stage, Usd.Stage)

        # Verify content was preserved
        test_cube = opened_stage.GetPrimAtPath("/World/TestCube")
        assert test_cube.IsValid()
        assert test_cube.GetTypeName() == "Cube"


def test_open_stage_invalid_path():
    """Test opening a stage with invalid path."""
    with pytest.raises(ValueError, match="not supported"):
        sim_utils.open_stage("/invalid/path/to/stage.invalid")


def test_use_stage_context_manager():
    """Test use_stage context manager."""
    # Create two stages
    stage1 = sim_utils.create_new_stage()
    stage1.DefinePrim("/World", "Xform")
    stage1.DefinePrim("/World/Stage1Marker", "Xform")

    stage2 = Usd.Stage.CreateInMemory()
    stage2.DefinePrim("/World", "Xform")
    stage2.DefinePrim("/World/Stage2Marker", "Xform")

    # Initially on stage1
    current = sim_utils.get_current_stage()
    marker1 = current.GetPrimAtPath("/World/Stage1Marker")
    assert marker1.IsValid()

    # Switch to stage2 temporarily
    with sim_utils.use_stage(stage2):
        temp_current = sim_utils.get_current_stage()
        # Should be on stage2 now
        marker2 = temp_current.GetPrimAtPath("/World/Stage2Marker")
        assert marker2.IsValid()

    # Should be back on stage1
    final_current = sim_utils.get_current_stage()
    marker1_again = final_current.GetPrimAtPath("/World/Stage1Marker")
    assert marker1_again.IsValid()


def test_use_stage_with_invalid_input():
    """Test use_stage with invalid input."""
    with pytest.raises((TypeError, AssertionError)):
        with sim_utils.use_stage("not a stage"):  # type: ignore
            pass


def test_update_stage():
    """Test updating the stage."""
    # Create a new stage
    stage = sim_utils.create_new_stage()

    # Add a prim
    prim_path = "/World/Test"
    stage.DefinePrim(prim_path, "Xform")

    # Update stage should not raise errors
    sim_utils.update_stage()

    # Prim should still exist
    prim = stage.GetPrimAtPath(prim_path)
    assert prim.IsValid()


def test_save_stage_with_reload():
    """Test saving stage with reload in place."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a stage with content
        stage = sim_utils.create_new_stage()
        stage.DefinePrim("/World", "Xform")
        stage.DefinePrim("/World/TestSphere", "Sphere")

        # Save with reload
        save_path = Path(temp_dir) / "test_reload.usd"
        result = sim_utils.save_stage(str(save_path), save_and_reload_in_place=True)

        assert result is True
        assert save_path.exists()

        # Stage should be reloaded, content should be preserved
        current_stage = sim_utils.get_current_stage()
        test_sphere = current_stage.GetPrimAtPath("/World/TestSphere")
        assert test_sphere.IsValid()


def test_save_stage_invalid_path():
    """Test saving stage with invalid path."""
    _ = sim_utils.create_new_stage()

    with pytest.raises(ValueError, match="not supported"):
        sim_utils.save_stage("/tmp/test.invalid")


def test_close_stage():
    """Test closing the current stage."""
    # Create a stage
    stage = sim_utils.create_new_stage()
    assert stage is not None

    # Close it
    result = sim_utils.close_stage()

    # Should succeed (or return bool)
    assert isinstance(result, bool)


def test_clear_stage():
    """Test clearing the stage."""
    # Create a new stage
    stage = sim_utils.create_new_stage()

    # Add some prims
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Cube", "Cube")
    stage.DefinePrim("/World/Sphere", "Sphere")

    # Clear the stage
    sim_utils.clear_stage()

    # Stage should still exist but prims should be removed
    assert stage is not None


def test_get_current_stage():
    """Test getting the current stage."""
    # Create a new stage
    created_stage = sim_utils.create_new_stage()

    # Get current stage should return the same stage
    current_stage = sim_utils.get_current_stage()
    assert current_stage == created_stage
    assert isinstance(current_stage, Usd.Stage)


def test_get_current_stage_id():
    """Test getting the current stage ID."""
    # Create a new stage
    sim_utils.create_new_stage()

    # Get stage ID
    stage_id = sim_utils.get_current_stage_id()

    # Should be a valid integer ID
    assert isinstance(stage_id, int)
    assert stage_id >= 0


def test_resolve_paths():
    """Test resolve_paths helper for asset path resolution."""
    from isaaclab.sim.utils.stage import resolve_paths

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a source stage with a sublayer reference
        source_path = Path(temp_dir) / "source" / "source_stage.usd"
        source_path.parent.mkdir(parents=True, exist_ok=True)

        # Create source stage with some content
        source_stage = Usd.Stage.CreateNew(str(source_path))
        source_stage.DefinePrim("/World", "Xform")
        source_stage.DefinePrim("/World/Cube", "Cube")
        source_stage.GetRootLayer().Save()

        # Copy to a different location using layer transfer
        dest_path = Path(temp_dir) / "dest" / "dest_stage.usd"
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        from pxr import Sdf

        dest_layer = Sdf.Layer.CreateNew(str(dest_path))
        dest_layer.TransferContent(source_stage.GetRootLayer())

        # Resolve paths (should not raise any errors)
        resolve_paths(str(source_path), str(dest_path))
        dest_layer.Save()

        # Open destination stage and verify content was preserved
        dest_stage = Usd.Stage.Open(str(dest_path))
        cube_prim = dest_stage.GetPrimAtPath("/World/Cube")
        assert cube_prim.IsValid()
        assert cube_prim.GetTypeName() == "Cube"


def test_stage_context_tracking():
    """Test that stage context is properly tracked across operations."""
    # Create initial stage
    stage1 = sim_utils.create_new_stage()
    stage1.DefinePrim("/Stage1Marker", "Xform")

    # Verify it's the current stage
    current = sim_utils.get_current_stage()
    assert current.GetPrimAtPath("/Stage1Marker").IsValid()

    # Create another stage - should become current
    stage2 = sim_utils.create_new_stage()
    stage2.DefinePrim("/Stage2Marker", "Xform")

    current = sim_utils.get_current_stage()
    assert current.GetPrimAtPath("/Stage2Marker").IsValid()
    assert not current.GetPrimAtPath("/Stage1Marker").IsValid()

    # Use stage context manager to temporarily switch
    with sim_utils.use_stage(stage1):
        current = sim_utils.get_current_stage()
        assert current.GetPrimAtPath("/Stage1Marker").IsValid()

    # After context manager, should be back to stage2
    current = sim_utils.get_current_stage()
    assert current.GetPrimAtPath("/Stage2Marker").IsValid()


def test_is_prim_deletable():
    """Test _is_prim_deletable with various prim types."""
    from isaaclab.sim.utils.stage import _is_prim_deletable

    stage = sim_utils.create_new_stage()

    # Create a locally authored prim - should be deletable
    local_prim = stage.DefinePrim("/World/LocalPrim", "Xform")
    assert _is_prim_deletable(local_prim) is True

    # Create another deletable prim
    another_prim = stage.DefinePrim("/World/AnotherPrim", "Cube")
    assert _is_prim_deletable(another_prim) is True

    # Root prim should not be deletable
    root_prim = stage.GetPseudoRoot()
    assert _is_prim_deletable(root_prim) is False


def test_open_stage_sets_current():
    """Test that open_stage sets the opened stage as current."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create and save a stage
        stage = sim_utils.create_new_stage()
        stage.DefinePrim("/TestPrim", "Xform")

        save_path = Path(temp_dir) / "test.usd"
        sim_utils.save_stage(str(save_path), save_and_reload_in_place=False)

        # Create a different stage
        sim_utils.create_new_stage()
        sim_utils.get_current_stage().DefinePrim("/DifferentPrim", "Xform")

        # Open the saved stage
        opened = sim_utils.open_stage(str(save_path))

        # Opened stage should now be current
        current = sim_utils.get_current_stage()
        assert current == opened
        assert current.GetPrimAtPath("/TestPrim").IsValid()
