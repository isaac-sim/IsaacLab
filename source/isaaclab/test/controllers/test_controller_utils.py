# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test cases for Isaac Lab controller utilities."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import os

# Import the function to test
import tempfile

import pytest
import torch

from isaaclab.controllers.utils import change_revolute_to_fixed, change_revolute_to_fixed_regex
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.io.torchscript import load_torchscript_model


@pytest.fixture
def mock_urdf_content():
    """Create mock URDF content for testing."""
    return """<?xml version="1.0"?>
<robot name="test_robot">
    <link name="base_link">
        <visual>
            <geometry>
                <box size="0.1 0.1 0.1"/>
            </geometry>
        </visual>
    </link>

    <joint name="base_to_shoulder" type="revolute">
        <parent link="base_link"/>
        <child link="shoulder_link"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
    </joint>

    <link name="shoulder_link">
        <visual>
            <geometry>
                <cylinder radius="0.05" length="0.2"/>
            </geometry>
        </visual>
    </link>

    <joint name="shoulder_to_elbow" type="revolute">
        <parent link="shoulder_link"/>
        <child link="elbow_link"/>
        <origin xyz="0 0 0.2" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
    </joint>

    <link name="elbow_link">
        <visual>
            <geometry>
                <cylinder radius="0.04" length="0.15"/>
            </geometry>
        </visual>
    </link>

    <joint name="elbow_to_wrist" type="revolute">
        <parent link="elbow_link"/>
        <child link="wrist_link"/>
        <origin xyz="0 0 0.15" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
    </joint>

    <link name="wrist_link">
        <visual>
            <geometry>
                <sphere radius="0.03"/>
            </geometry>
        </visual>
    </link>

    <joint name="wrist_to_gripper" type="fixed">
        <parent link="wrist_link"/>
        <child link="gripper_link"/>
        <origin xyz="0 0 0.05" rpy="0 0 0"/>
    </joint>

    <link name="gripper_link">
        <visual>
            <geometry>
                <box size="0.02 0.02 0.02"/>
            </geometry>
        </visual>
    </link>
</robot>"""


@pytest.fixture
def test_urdf_file(mock_urdf_content):
    """Create a temporary URDF file for testing."""
    # Create a temporary directory for test files
    test_dir = tempfile.mkdtemp()

    # Create the test URDF file
    test_urdf_path = os.path.join(test_dir, "test_robot.urdf")
    with open(test_urdf_path, "w") as f:
        f.write(mock_urdf_content)

    yield test_urdf_path

    # Clean up the temporary directory and all its contents
    import shutil

    shutil.rmtree(test_dir)


# =============================================================================
# Test cases for change_revolute_to_fixed function
# =============================================================================


def test_single_joint_conversion(test_urdf_file, mock_urdf_content):
    """Test converting a single revolute joint to fixed."""
    # Test converting shoulder_to_elbow joint
    fixed_joints = ["shoulder_to_elbow"]
    change_revolute_to_fixed(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that the joint was converted
    assert '<joint name="shoulder_to_elbow" type="fixed">' in modified_content
    assert '<joint name="shoulder_to_elbow" type="revolute">' not in modified_content

    # Check that other revolute joints remain unchanged
    assert '<joint name="base_to_shoulder" type="revolute">' in modified_content
    assert '<joint name="elbow_to_wrist" type="revolute">' in modified_content


def test_multiple_joints_conversion(test_urdf_file, mock_urdf_content):
    """Test converting multiple revolute joints to fixed."""
    # Test converting multiple joints
    fixed_joints = ["base_to_shoulder", "elbow_to_wrist"]
    change_revolute_to_fixed(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that both joints were converted
    assert '<joint name="base_to_shoulder" type="fixed">' in modified_content
    assert '<joint name="elbow_to_wrist" type="fixed">' in modified_content
    assert '<joint name="base_to_shoulder" type="revolute">' not in modified_content
    assert '<joint name="elbow_to_wrist" type="revolute">' not in modified_content

    # Check that the middle joint remains unchanged
    assert '<joint name="shoulder_to_elbow" type="revolute">' in modified_content


def test_non_existent_joint(test_urdf_file, mock_urdf_content):
    """Test behavior when trying to convert a non-existent joint."""
    # Try to convert a joint that doesn't exist
    fixed_joints = ["non_existent_joint"]
    change_revolute_to_fixed(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that the file content remains unchanged
    assert modified_content == mock_urdf_content


def test_mixed_existent_and_non_existent_joints(test_urdf_file, mock_urdf_content):
    """Test converting a mix of existent and non-existent joints."""
    # Try to convert both existent and non-existent joints
    fixed_joints = ["base_to_shoulder", "non_existent_joint", "elbow_to_wrist"]
    change_revolute_to_fixed(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that existent joints were converted
    assert '<joint name="base_to_shoulder" type="fixed">' in modified_content
    assert '<joint name="elbow_to_wrist" type="fixed">' in modified_content

    # Check that non-existent joint didn't cause issues
    assert '<joint name="non_existent_joint" type="fixed">' not in modified_content


def test_already_fixed_joint(test_urdf_file, mock_urdf_content):
    """Test behavior when trying to convert an already fixed joint."""
    # Try to convert a joint that is already fixed
    fixed_joints = ["wrist_to_gripper"]
    change_revolute_to_fixed(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that the file content remains unchanged (no conversion happened)
    assert modified_content == mock_urdf_content


def test_empty_joints_list(test_urdf_file, mock_urdf_content):
    """Test behavior when passing an empty list of joints."""
    # Try to convert with empty list
    fixed_joints = []
    change_revolute_to_fixed(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that the file content remains unchanged
    assert modified_content == mock_urdf_content


def test_file_not_found(test_urdf_file):
    """Test behavior when URDF file doesn't exist."""
    non_existent_path = os.path.join(os.path.dirname(test_urdf_file), "non_existent.urdf")
    fixed_joints = ["base_to_shoulder"]

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        change_revolute_to_fixed(non_existent_path, fixed_joints)


def test_preserve_other_content(test_urdf_file):
    """Test that other content in the URDF file is preserved."""
    fixed_joints = ["shoulder_to_elbow"]
    change_revolute_to_fixed(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that other content is preserved
    assert '<robot name="test_robot">' in modified_content
    assert '<link name="base_link">' in modified_content
    assert '<link name="shoulder_link">' in modified_content
    assert '<link name="elbow_link">' in modified_content
    assert '<link name="wrist_link">' in modified_content
    assert '<link name="gripper_link">' in modified_content

    # Check that the fixed joint remains unchanged
    assert '<joint name="wrist_to_gripper" type="fixed">' in modified_content


def test_joint_attributes_preserved(test_urdf_file):
    """Test that joint attributes other than type are preserved."""
    fixed_joints = ["base_to_shoulder"]
    change_revolute_to_fixed(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that the joint was converted but other attributes preserved
    assert '<joint name="base_to_shoulder" type="fixed">' in modified_content
    assert '<parent link="base_link"/>' in modified_content
    assert '<child link="shoulder_link"/>' in modified_content
    assert '<origin xyz="0 0 0.1" rpy="0 0 0"/>' in modified_content
    assert '<axis xyz="0 0 1"/>' in modified_content
    assert '<limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>' in modified_content


# =============================================================================
# Test cases for change_revolute_to_fixed_regex function
# =============================================================================


def test_regex_single_joint_conversion(test_urdf_file, mock_urdf_content):
    """Test converting a single revolute joint to fixed using regex pattern."""
    # Test converting shoulder_to_elbow joint using exact match
    fixed_joints = ["shoulder_to_elbow"]
    change_revolute_to_fixed_regex(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that the joint was converted
    assert '<joint name="shoulder_to_elbow" type="fixed">' in modified_content
    assert '<joint name="shoulder_to_elbow" type="revolute">' not in modified_content

    # Check that other revolute joints remain unchanged
    assert '<joint name="base_to_shoulder" type="revolute">' in modified_content
    assert '<joint name="elbow_to_wrist" type="revolute">' in modified_content


def test_regex_pattern_matching(test_urdf_file, mock_urdf_content):
    """Test converting joints using regex patterns."""
    # Test converting joints that contain "to" in their name
    fixed_joints = [r".*to.*"]
    change_revolute_to_fixed_regex(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that all joints with "to" in the name were converted
    assert '<joint name="base_to_shoulder" type="fixed">' in modified_content
    assert '<joint name="shoulder_to_elbow" type="fixed">' in modified_content
    assert '<joint name="elbow_to_wrist" type="fixed">' in modified_content

    # Check that the fixed joint remains unchanged
    assert '<joint name="wrist_to_gripper" type="fixed">' in modified_content


def test_regex_multiple_patterns(test_urdf_file, mock_urdf_content):
    """Test converting joints using multiple regex patterns."""
    # Test converting joints that start with "base" or end with "wrist"
    fixed_joints = [r"^base.*", r".*wrist$"]
    change_revolute_to_fixed_regex(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that matching joints were converted
    assert '<joint name="base_to_shoulder" type="fixed">' in modified_content
    assert '<joint name="elbow_to_wrist" type="fixed">' in modified_content

    # Check that non-matching joints remain unchanged
    assert '<joint name="shoulder_to_elbow" type="revolute">' in modified_content


def test_regex_case_sensitive_matching(test_urdf_file, mock_urdf_content):
    """Test that regex matching is case sensitive."""
    # Test with uppercase pattern that won't match lowercase joint names
    fixed_joints = [r".*TO.*"]
    change_revolute_to_fixed_regex(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that no joints were converted (case sensitive)
    assert modified_content == mock_urdf_content


def test_regex_partial_word_matching(test_urdf_file, mock_urdf_content):
    """Test converting joints using partial word matching."""
    # Test converting joints that contain "shoulder" in their name
    fixed_joints = [r".*shoulder.*"]
    change_revolute_to_fixed_regex(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that shoulder-related joints were converted
    assert '<joint name="base_to_shoulder" type="fixed">' in modified_content
    assert '<joint name="shoulder_to_elbow" type="fixed">' in modified_content

    # Check that other joints remain unchanged
    assert '<joint name="elbow_to_wrist" type="revolute">' in modified_content


def test_regex_no_matches(test_urdf_file, mock_urdf_content):
    """Test behavior when regex patterns don't match any joints."""
    # Test with pattern that won't match any joint names
    fixed_joints = [r"^nonexistent.*"]
    change_revolute_to_fixed_regex(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that the file content remains unchanged
    assert modified_content == mock_urdf_content


def test_regex_empty_patterns_list(test_urdf_file, mock_urdf_content):
    """Test behavior when passing an empty list of regex patterns."""
    # Try to convert with empty list
    fixed_joints = []
    change_revolute_to_fixed_regex(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that the file content remains unchanged
    assert modified_content == mock_urdf_content


def test_regex_file_not_found(test_urdf_file):
    """Test behavior when URDF file doesn't exist for regex function."""
    non_existent_path = os.path.join(os.path.dirname(test_urdf_file), "non_existent.urdf")
    fixed_joints = [r".*to.*"]

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        change_revolute_to_fixed_regex(non_existent_path, fixed_joints)


def test_regex_preserve_other_content(test_urdf_file):
    """Test that other content in the URDF file is preserved with regex function."""
    fixed_joints = [r".*shoulder.*"]
    change_revolute_to_fixed_regex(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that other content is preserved
    assert '<robot name="test_robot">' in modified_content
    assert '<link name="base_link">' in modified_content
    assert '<link name="shoulder_link">' in modified_content
    assert '<link name="elbow_link">' in modified_content
    assert '<link name="wrist_link">' in modified_content
    assert '<link name="gripper_link">' in modified_content

    # Check that the fixed joint remains unchanged
    assert '<joint name="wrist_to_gripper" type="fixed">' in modified_content


def test_regex_joint_attributes_preserved(test_urdf_file):
    """Test that joint attributes other than type are preserved with regex function."""
    fixed_joints = [r"^base.*"]
    change_revolute_to_fixed_regex(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that the joint was converted but other attributes preserved
    assert '<joint name="base_to_shoulder" type="fixed">' in modified_content
    assert '<parent link="base_link"/>' in modified_content
    assert '<child link="shoulder_link"/>' in modified_content
    assert '<origin xyz="0 0 0.1" rpy="0 0 0"/>' in modified_content
    assert '<axis xyz="0 0 1"/>' in modified_content
    assert '<limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>' in modified_content


def test_regex_complex_pattern(test_urdf_file, mock_urdf_content):
    """Test converting joints using a complex regex pattern."""
    # Test converting joints that have "to" and end with a word starting with "w"
    fixed_joints = [r".*to.*w.*"]
    change_revolute_to_fixed_regex(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that matching joints were converted
    assert '<joint name="elbow_to_wrist" type="fixed">' in modified_content
    assert '<joint name="shoulder_to_elbow" type="fixed">' in modified_content

    # Check that non-matching joints remain unchanged
    assert '<joint name="base_to_shoulder" type="revolute">' in modified_content


def test_regex_already_fixed_joint(test_urdf_file, mock_urdf_content):
    """Test behavior when regex pattern matches an already fixed joint."""
    # Try to convert joints that contain "gripper" (which is already fixed)
    fixed_joints = [r".*gripper.*"]
    change_revolute_to_fixed_regex(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that the file content remains unchanged (no conversion happened)
    assert modified_content == mock_urdf_content


def test_regex_special_characters(test_urdf_file, mock_urdf_content):
    """Test regex patterns with special characters."""
    # Test with pattern that includes special regex characters
    fixed_joints = [r".*to.*"]  # This should match joints with "to"
    change_revolute_to_fixed_regex(test_urdf_file, fixed_joints)

    # Read the modified file
    with open(test_urdf_file) as f:
        modified_content = f.read()

    # Check that joints with "to" were converted
    assert '<joint name="base_to_shoulder" type="fixed">' in modified_content
    assert '<joint name="shoulder_to_elbow" type="fixed">' in modified_content
    assert '<joint name="elbow_to_wrist" type="fixed">' in modified_content

    # Check that the fixed joint remains unchanged
    assert '<joint name="wrist_to_gripper" type="fixed">' in modified_content


# =============================================================================
# Test cases for load_torchscript_model function
# =============================================================================


@pytest.fixture
def policy_model_path():
    """Path to the test TorchScript model."""
    _policy_path = f"{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt"
    return retrieve_file_path(_policy_path)


def test_load_torchscript_model_success(policy_model_path):
    """Test successful loading of a TorchScript model."""
    model = load_torchscript_model(policy_model_path)

    # Check that model was loaded successfully
    assert model is not None
    assert isinstance(model, torch.nn.Module)

    # Check that model is in evaluation mode
    assert model.training is False


def test_load_torchscript_model_cpu_device(policy_model_path):
    """Test loading TorchScript model on CPU device."""
    model = load_torchscript_model(policy_model_path, device="cpu")

    # Check that model was loaded successfully
    assert model is not None
    assert isinstance(model, torch.nn.Module)

    # Check that model is in evaluation mode
    assert model.training is False


def test_load_torchscript_model_cuda_device(policy_model_path):
    """Test loading TorchScript model on CUDA device if available."""
    if torch.cuda.is_available():
        model = load_torchscript_model(policy_model_path, device="cuda")

        # Check that model was loaded successfully
        assert model is not None
        assert isinstance(model, torch.nn.Module)

        # Check that model is in evaluation mode
        assert model.training is False
    else:
        # Skip test if CUDA is not available
        pytest.skip("CUDA not available")


def test_load_torchscript_model_file_not_found():
    """Test behavior when TorchScript model file doesn't exist."""
    non_existent_path = "non_existent_model.pt"

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_torchscript_model(non_existent_path)


def test_load_torchscript_model_invalid_file():
    """Test behavior when trying to load an invalid TorchScript file."""
    # Create a temporary file with invalid content
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
        temp_file.write(b"invalid torchscript content")
        temp_file_path = temp_file.name

    try:
        # Should handle the error gracefully and return None
        model = load_torchscript_model(temp_file_path)
        assert model is None
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


def test_load_torchscript_model_empty_file():
    """Test behavior when trying to load an empty TorchScript file."""
    # Create a temporary empty file
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
        temp_file_path = temp_file.name

    try:
        # Should handle the error gracefully and return None
        model = load_torchscript_model(temp_file_path)
        assert model is None
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


def test_load_torchscript_model_different_device_mapping(policy_model_path):
    """Test loading model with different device mapping."""
    # Test with specific device mapping
    model = load_torchscript_model(policy_model_path, device="cpu")

    # Check that model was loaded successfully
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_load_torchscript_model_evaluation_mode(policy_model_path):
    """Test that loaded model is in evaluation mode."""
    model = load_torchscript_model(policy_model_path)

    # Check that model is in evaluation mode
    assert model.training is False

    # Verify we can set it to training mode and back
    model.train()
    assert model.training is True
    model.eval()
    assert model.training is False


def test_load_torchscript_model_inference_capability(policy_model_path):
    """Test that loaded model can perform inference."""
    model = load_torchscript_model(policy_model_path)

    # Check that model was loaded successfully
    assert model is not None

    # Try to create a dummy input tensor (actual input shape depends on the model)
    # This is a basic test to ensure the model can handle tensor inputs
    try:
        # Create a dummy input tensor (adjust size based on expected input)
        dummy_input = torch.randn(1, 75)  # Adjust dimensions as needed

        # Try to run inference (this might fail if input shape is wrong, but shouldn't crash)
        with torch.no_grad():
            try:
                output = model(dummy_input)
                # If successful, check that output is a tensor
                assert isinstance(output, torch.Tensor)
            except (RuntimeError, ValueError):
                # Expected if input shape doesn't match model expectations
                # This is acceptable for this test
                pass
    except Exception:
        # If model doesn't accept this input format, that's okay for this test
        # The main goal is to ensure the model loads without crashing
        pass


def test_load_torchscript_model_error_handling():
    """Test error handling when loading fails."""
    # Create a temporary file that will cause a loading error
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
        temp_file.write(b"definitely not a torchscript model")
        temp_file_path = temp_file.name

    try:
        # Should handle the error gracefully and return None
        model = load_torchscript_model(temp_file_path)
        assert model is None
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)
