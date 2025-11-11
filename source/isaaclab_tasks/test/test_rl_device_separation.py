# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test RL device separation across all supported RL libraries.

This test verifies that RL library wrappers correctly handle device transfers when the
simulation device differs from the RL training device.

Device Architecture:
    1. sim_device: Where physics simulation runs and environment buffers live
    2. rl_device: Where policy networks and training computations occur

Test Scenarios:
    - GPU simulation + GPU RL: Same device (no transfers needed, optimal performance)
    - GPU simulation + CPU RL: Cross-device transfers (wrapper handles transfers)
    - CPU simulation + CPU RL: CPU-only operation

Each test verifies the wrapper correctly:
    1. Unwrapped env: operates entirely on sim_device
    2. Wrapper: accepts actions on rl_device (where policy generates them)
    3. Wrapper: internally transfers actions from rl_device → sim_device for env.step()
    4. Wrapper: transfers outputs from sim_device → rl_device (for policy to use)

Tested Libraries:
    - RSL-RL: TensorDict observations, explicit rl_device parameter
        * Transfers observations and rewards to rl_device
    - RL Games: Dict observations, explicit rl_device parameter
        * Transfers observations and rewards to rl_device
    - Stable-Baselines3: Numpy arrays (CPU-only by design)
        * Always converts to/from numpy on CPU
    - skrl: Dict observations, uses skrl.config.torch.device for RL device
        * Keeps observations on sim_device (policy handles transfer)
        * Only transfers actions from rl_device to sim_device

IMPORTANT: Due to Isaac Sim limitations, only ONE test can be run per pytest invocation.
Run tests individually:
    pytest test_rl_device_separation.py::test_rsl_rl_device_separation_gpu_to_gpu -v -s
    pytest test_rl_device_separation.py::test_rsl_rl_device_separation_gpu_to_cpu -v -s
    pytest test_rl_device_separation.py::test_rl_games_device_separation_gpu_to_gpu -v -s
    ...
"""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import carb
import omni.usd
import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Test environment - use Cartpole as it's simple and fast
TEST_ENV = "Isaac-Cartpole-v0"
NUM_ENVS = 4


def _test_rsl_rl_device_separation(sim_device: str, rl_device: str):
    """Helper function to test RSL-RL with specified device configuration.

    Args:
        sim_device: Device for simulation (e.g., "cuda:0", "cpu")
        rl_device: Device for RL agent (e.g., "cuda:0", "cpu")
    """
    from tensordict import TensorDict

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    print(f"\n{'=' * 60}")
    print(f">>> Testing RSL-RL with sim_device={sim_device}, rl_device={rl_device}")
    print(f"{'=' * 60}")

    # Create a new stage
    omni.usd.get_context().new_stage()
    # Reset the rtx sensors carb setting to False
    carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", False)

    try:
        # Parse environment config
        print("  [1/6] Parsing environment config...")
        env_cfg = parse_env_cfg(TEST_ENV, device=sim_device, num_envs=NUM_ENVS)

        # Create environment
        print("  [2/6] Creating environment (may take 5-10s)...")
        env = gym.make(TEST_ENV, cfg=env_cfg)
        print("  [2/6] Environment created successfully")
    except Exception as e:
        # Try to close environment on exception
        if "env" in locals() and hasattr(env, "_is_closed"):
            env.close()
        else:
            if hasattr(e, "obj") and hasattr(e.obj, "_is_closed"):
                e.obj.close()
        pytest.fail(f"Failed to set-up the environment for task {TEST_ENV}. Error: {e}")

    # Disable control on stop
    env.unwrapped.sim._app_control_on_stop_handle = None

    # Verify environment device
    print("  [3/6] Verifying environment device...")
    assert (
        env.unwrapped.device == sim_device
    ), f"Environment device mismatch: expected {sim_device}, got {env.unwrapped.device}"

    # Test environment directly before wrapping to verify it returns data on sim device
    print("  [3/6] Testing unwrapped environment returns data on sim_device...")
    obs_dict, _ = env.reset()
    for key, value in obs_dict.items():
        if isinstance(value, torch.Tensor):
            assert (
                value.device.type == torch.device(sim_device).type
            ), f"Unwrapped env obs '{key}' should be on {sim_device}, got {value.device}"

    # Step unwrapped environment to verify outputs are on sim device
    action_space = env.unwrapped.single_action_space
    test_action = torch.zeros(NUM_ENVS, action_space.shape[0], device=sim_device)
    obs_dict, rew, term, trunc, extras = env.step(test_action)
    assert (
        rew.device.type == torch.device(sim_device).type
    ), f"Unwrapped env rewards should be on {sim_device}, got {rew.device}"
    assert (
        term.device.type == torch.device(sim_device).type
    ), f"Unwrapped env terminated should be on {sim_device}, got {term.device}"
    print(f"  [3/6] Verified: Unwrapped environment returns data on {sim_device}")

    # Create RSL-RL wrapper with RL device
    print("  [4/6] Creating RSL-RL wrapper...")
    env = RslRlVecEnvWrapper(env, rl_device=rl_device)
    print(f"  [4/6] Wrapper created (env_device={env.env_device}, rl_device={env.rl_device})")

    # Verify devices
    assert env.env_device == sim_device, f"Wrapper env_device should be {sim_device}"
    assert env.rl_device == rl_device, f"Wrapper RL device should be {rl_device}"
    assert env.device == rl_device, f"Wrapper device property should be {rl_device}"

    # Reset and step to test device transfers
    print("  [5/6] Testing reset and step operations...")
    obs, extras = env.reset()
    print("  [5/6] Reset completed")

    # Verify observations are on RL device (RSL-RL returns TensorDict)
    assert isinstance(obs, TensorDict), f"Expected TensorDict, got {type(obs)}"
    for key, value in obs.items():
        if isinstance(value, torch.Tensor):
            assert (
                value.device.type == torch.device(rl_device).type
            ), f"Observation '{key}' should be on {rl_device}, got {value.device}"

    # Sample random action on RL device (simulating policy output)
    # RSL-RL: action_space.shape is (num_envs, action_dim)
    action = 2 * torch.rand(env.action_space.shape, device=rl_device) - 1
    print(f"  [5/6] Action created on rl_device: {action.device}, shape: {action.shape}")

    # Verify action is on RL device before calling step
    assert (
        action.device.type == torch.device(rl_device).type
    ), f"Action should be on {rl_device} before step, got {action.device}"

    # Step environment - wrapper should:
    # 1. Accept action on rl_device
    # 2. Transfer action from rl_device to sim_device internally
    # 3. Call unwrapped env.step() with action on sim_device
    # 4. Transfer outputs from sim_device to rl_device
    obs, reward, dones, extras = env.step(action)
    print("  [5/6] Step completed - wrapper handled device transfers")

    # Verify all outputs are on RL device (wrapper transferred from sim_device)
    print("  [6/6] Verifying device transfers...")
    assert isinstance(obs, TensorDict), f"Expected TensorDict, got {type(obs)}"
    for key, value in obs.items():
        if isinstance(value, torch.Tensor):
            assert (
                value.device.type == torch.device(rl_device).type
            ), f"Step observation '{key}' should be on {rl_device}, got {value.device}"
    assert reward.device.type == torch.device(rl_device).type, f"Rewards should be on {rl_device}, got {reward.device}"
    assert dones.device.type == torch.device(rl_device).type, f"Dones should be on {rl_device}, got {dones.device}"

    # Cleanup
    print("  [6/6] Cleaning up environment...")
    env.close()
    print(f"✓ RSL-RL test PASSED for sim_device={sim_device}, rl_device={rl_device}")
    print("  Wrapper device transfer verified:")
    print(f"    1. Unwrapped env: expects actions on {sim_device}, returns data on {sim_device}")
    print(f"    2. Wrapper: accepts actions on {rl_device} (from policy)")
    print(f"    3. Wrapper: internally transfers actions to {sim_device} for env.step()")
    print(f"    4. Wrapper: transfers outputs from {sim_device} to {rl_device} (for policy)")
    print("-" * 80)


def _test_rl_games_device_separation(sim_device: str, rl_device: str):
    """Helper function to test RL Games with specified device configuration.

    Args:
        sim_device: Device for simulation (e.g., "cuda:0", "cpu")
        rl_device: Device for RL agent (e.g., "cuda:0", "cpu")
    """
    from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

    print(f"\n{'=' * 60}")
    print(f">>> Testing RL Games with sim_device={sim_device}, rl_device={rl_device}")
    print(f"{'=' * 60}")

    # Create a new stage
    omni.usd.get_context().new_stage()
    # Reset the rtx sensors carb setting to False
    carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", False)

    try:
        # Parse environment config
        print("  [1/5] Parsing environment config...")
        env_cfg = parse_env_cfg(TEST_ENV, device=sim_device, num_envs=NUM_ENVS)

        # Create environment
        print("  [2/5] Creating environment (may take 5-10s)...")
        env = gym.make(TEST_ENV, cfg=env_cfg)
        print("  [2/5] Environment created successfully")
    except Exception as e:
        # Try to close environment on exception
        if "env" in locals() and hasattr(env, "_is_closed"):
            env.close()
        else:
            if hasattr(e, "obj") and hasattr(e.obj, "_is_closed"):
                e.obj.close()
        pytest.fail(f"Failed to set-up the environment for task {TEST_ENV}. Error: {e}")

    # Disable control on stop
    env.unwrapped.sim._app_control_on_stop_handle = None

    # Verify environment device
    print("  [3/5] Verifying environment device...")
    assert (
        env.unwrapped.device == sim_device
    ), f"Environment device mismatch: expected {sim_device}, got {env.unwrapped.device}"

    # Test environment directly before wrapping to verify it returns data on sim device
    print("  [3/5] Testing unwrapped environment returns data on sim_device...")
    obs_dict, _ = env.reset()
    for key, value in obs_dict.items():
        if isinstance(value, torch.Tensor):
            assert (
                value.device.type == torch.device(sim_device).type
            ), f"Unwrapped env obs '{key}' should be on {sim_device}, got {value.device}"

    # Step unwrapped environment to verify outputs are on sim device
    action_space = env.unwrapped.single_action_space
    test_action = torch.zeros(NUM_ENVS, action_space.shape[0], device=sim_device)
    obs_dict, rew, term, trunc, extras = env.step(test_action)
    assert (
        rew.device.type == torch.device(sim_device).type
    ), f"Unwrapped env rewards should be on {sim_device}, got {rew.device}"
    assert (
        term.device.type == torch.device(sim_device).type
    ), f"Unwrapped env terminated should be on {sim_device}, got {term.device}"
    print(f"  [3/5] Verified: Unwrapped environment returns data on {sim_device}")

    # Create RL Games wrapper with RL device
    print("  [3/5] Creating RL Games wrapper...")
    env = RlGamesVecEnvWrapper(env, rl_device=rl_device, clip_obs=10.0, clip_actions=1.0)

    # Reset and step to test device transfers
    print("  [4/5] Testing reset and step operations...")
    obs = env.reset()
    print("  [4/5] Reset completed")

    # Verify observations are on RL device
    if isinstance(obs, dict):
        for key, value in obs.items():
            assert (
                value.device.type == torch.device(rl_device).type
            ), f"Observation '{key}' should be on {rl_device}, got {value.device}"
    else:
        assert (
            obs.device.type == torch.device(rl_device).type
        ), f"Observation should be on {rl_device}, got {obs.device}"

    # Sample random action on RL device (simulating policy output)
    action = 2 * torch.rand(NUM_ENVS, *env.action_space.shape, device=rl_device) - 1
    print(f"  [4/5] Action created on rl_device: {action.device}, shape: {action.shape}")

    # Verify action is on RL device before calling step
    assert (
        action.device.type == torch.device(rl_device).type
    ), f"Action should be on {rl_device} before step, got {action.device}"

    # Step environment - wrapper should:
    # 1. Accept action on rl_device
    # 2. Transfer action from rl_device to sim_device internally
    # 3. Call unwrapped env.step() with action on sim_device
    # 4. Transfer outputs from sim_device to rl_device
    obs, reward, dones, info = env.step(action)
    print("  [4/5] Step completed - wrapper handled device transfers")

    # Verify all outputs are on RL device (wrapper transferred from sim_device)
    print("  [5/5] Verifying device transfers...")
    # RL Games returns flat tensor for observations
    if isinstance(obs, dict):
        for key, value in obs.items():
            assert (
                value.device.type == torch.device(rl_device).type
            ), f"Observation '{key}' should be on {rl_device}, got {value.device}"
    else:
        assert (
            obs.device.type == torch.device(rl_device).type
        ), f"Observations should be on {rl_device}, got {obs.device}"
    assert reward.device.type == torch.device(rl_device).type, f"Rewards should be on {rl_device}, got {reward.device}"
    assert dones.device.type == torch.device(rl_device).type, f"Dones should be on {rl_device}, got {dones.device}"

    # Cleanup
    print("  [5/5] Cleaning up environment...")
    env.close()
    print(f"✓ RL Games test PASSED for sim_device={sim_device}, rl_device={rl_device}")
    print("  Wrapper device transfer verified:")
    print(f"    1. Unwrapped env: expects actions on {sim_device}, returns data on {sim_device}")
    print(f"    2. Wrapper: accepts actions on {rl_device} (from policy)")
    print(f"    3. Wrapper: internally transfers actions to {sim_device} for env.step()")
    print(f"    4. Wrapper: transfers outputs from {sim_device} to {rl_device} (for policy)")
    print("-" * 80)


def _test_sb3_device_separation(sim_device: str):
    """Helper function to test Stable-Baselines3 with specified device configuration.

    Note: SB3 always converts to CPU/numpy, so we don't test rl_device parameter.

    Args:
        sim_device: Device for simulation (e.g., "cuda:0", "cpu")
    """
    import numpy as np

    from isaaclab_rl.sb3 import Sb3VecEnvWrapper

    print(f"\n{'=' * 60}")
    print(f">>> Testing SB3 with sim_device={sim_device}")
    print(f"{'=' * 60}")

    # Create a new stage
    omni.usd.get_context().new_stage()
    # Reset the rtx sensors carb setting to False
    carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", False)

    try:
        # Parse environment config
        print("  [1/5] Parsing environment config...")
        env_cfg = parse_env_cfg(TEST_ENV, device=sim_device, num_envs=NUM_ENVS)

        # Create environment
        print("  [2/5] Creating environment (may take 5-10s)...")
        env = gym.make(TEST_ENV, cfg=env_cfg)
        print("  [2/5] Environment created successfully")
    except Exception as e:
        # Try to close environment on exception
        if "env" in locals() and hasattr(env, "_is_closed"):
            env.close()
        else:
            if hasattr(e, "obj") and hasattr(e.obj, "_is_closed"):
                e.obj.close()
        pytest.fail(f"Failed to set-up the environment for task {TEST_ENV}. Error: {e}")

    # Disable control on stop
    env.unwrapped.sim._app_control_on_stop_handle = None

    # Verify environment device
    print("  [3/5] Verifying environment device...")
    assert (
        env.unwrapped.device == sim_device
    ), f"Environment device mismatch: expected {sim_device}, got {env.unwrapped.device}"

    # Test environment directly before wrapping to verify it returns data on sim device
    print("  [3/5] Testing unwrapped environment returns data on sim_device...")
    obs_dict, _ = env.reset()
    for key, value in obs_dict.items():
        if isinstance(value, torch.Tensor):
            assert (
                value.device.type == torch.device(sim_device).type
            ), f"Unwrapped env obs '{key}' should be on {sim_device}, got {value.device}"
    print(f"  [3/5] Verified: Unwrapped environment returns data on {sim_device}")

    # Create SB3 wrapper (always converts to numpy/CPU)
    print("  [3/5] Creating SB3 wrapper...")
    env = Sb3VecEnvWrapper(env)

    # Reset and step to test device transfers
    print("  [4/5] Testing reset and step operations...")
    obs = env.reset()
    print("  [4/5] Reset completed")

    # SB3 observations should always be numpy arrays (on CPU)
    assert isinstance(obs, np.ndarray), f"SB3 observations should be numpy arrays, got {type(obs)}"

    # Sample random action (SB3 uses numpy)
    action = 2 * np.random.rand(env.num_envs, *env.action_space.shape) - 1
    assert isinstance(action, np.ndarray), f"Action should be numpy array, got {type(action)}"
    print(f"  [4/5] Action sampled (numpy array), shape: {action.shape}")

    # Step environment - wrapper should:
    # 1. Convert numpy action to torch tensor on sim_device internally
    # 2. Call unwrapped env.step() with action on sim_device
    # 3. Convert outputs from sim_device tensors to numpy arrays
    obs, reward, done, info = env.step(action)
    print("  [4/5] Step completed, outputs converted to numpy")

    # Verify all outputs are numpy arrays (wrapper transferred and converted)
    print("  [5/5] Verifying numpy conversions...")
    assert isinstance(obs, np.ndarray), f"Observations should be numpy arrays, got {type(obs)}"
    assert isinstance(reward, np.ndarray), f"Rewards should be numpy arrays, got {type(reward)}"
    assert isinstance(done, np.ndarray), f"Dones should be numpy arrays, got {type(done)}"

    # Cleanup
    print("  [5/5] Cleaning up environment...")
    env.close()
    print(f"✓ SB3 test PASSED for sim_device={sim_device}")
    print("  Wrapper device transfer verified:")
    print(f"    1. Unwrapped env: expects actions on {sim_device}, returns data on {sim_device}")
    print("    2. Wrapper: accepts numpy arrays (from policy on CPU)")
    print(f"    3. Wrapper: internally converts to tensors on {sim_device} for env.step()")
    print(f"    4. Wrapper: converts outputs from {sim_device} tensors to numpy arrays (for policy)")
    print("-" * 80)


def _test_skrl_device_separation(sim_device: str, rl_device: str):
    """Helper function to test skrl with specified device configuration.

    Note: skrl uses skrl.config.torch.device for device configuration.
    This can be set via agent_cfg["device"] for consistency with other libraries.

    Args:
        sim_device: Device for simulation (e.g., "cuda:0", "cpu")
        rl_device: Device for RL agent (e.g., "cuda:0", "cpu") - set via skrl.config.torch.device
    """
    try:
        import skrl
        from skrl.envs.wrappers.torch import wrap_env
    except ImportError:
        pytest.skip("skrl not installed")

    print(f"\n{'=' * 60}")
    print(f">>> Testing skrl with sim_device={sim_device}, rl_device={rl_device}")
    print(f"    Using skrl.config.torch.device = {rl_device}")
    print(f"{'=' * 60}")

    # Create agent config with device parameter (for demonstration/consistency)
    agent_cfg = {"device": rl_device}

    # Configure skrl device (can be set from agent_cfg for consistency with other libraries)
    if "device" in agent_cfg:
        skrl.config.torch.device = torch.device(agent_cfg["device"])
    else:
        skrl.config.torch.device = torch.device(rl_device)

    # Create a new stage
    omni.usd.get_context().new_stage()
    # Reset the rtx sensors carb setting to False
    carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", False)

    try:
        # Parse environment config
        print("  [1/6] Parsing environment config...")
        env_cfg = parse_env_cfg(TEST_ENV, device=sim_device, num_envs=NUM_ENVS)

        # Create environment
        print("  [2/6] Creating environment (may take 5-10s)...")
        env = gym.make(TEST_ENV, cfg=env_cfg)
        print("  [2/6] Environment created successfully")
    except Exception as e:
        # Try to close environment on exception
        if "env" in locals() and hasattr(env, "_is_closed"):
            env.close()
        else:
            if hasattr(e, "obj") and hasattr(e.obj, "_is_closed"):
                e.obj.close()
        pytest.fail(f"Failed to set-up the environment for task {TEST_ENV}. Error: {e}")

    # Disable control on stop
    env.unwrapped.sim._app_control_on_stop_handle = None

    # Verify environment device
    print("  [3/6] Verifying environment device...")
    assert (
        env.unwrapped.device == sim_device
    ), f"Environment device mismatch: expected {sim_device}, got {env.unwrapped.device}"

    # Test environment directly before wrapping to verify it returns data on sim device
    print("  [3/6] Testing unwrapped environment returns data on sim_device...")
    obs_dict, _ = env.reset()
    for key, value in obs_dict.items():
        if isinstance(value, torch.Tensor):
            assert (
                value.device.type == torch.device(sim_device).type
            ), f"Unwrapped env obs '{key}' should be on {sim_device}, got {value.device}"
    print(f"  [3/6] Verified: Unwrapped environment returns data on {sim_device}")

    # Wrap with skrl (will use skrl.config.torch.device for policy)
    print("  [3/6] Creating skrl wrapper...")
    env = wrap_env(env, wrapper="isaaclab")

    # Reset to test basic functionality
    print("  [4/6] Testing reset and step operations...")
    obs, info = env.reset()
    print("  [4/6] Reset completed")

    # Verify observations are tensors or dict
    # skrl can return either dict or tensor depending on configuration
    if isinstance(obs, dict):
        assert isinstance(obs["policy"], torch.Tensor), f"Observations should be tensors, got {type(obs['policy'])}"
    else:
        assert isinstance(obs, torch.Tensor), f"Observations should be tensors, got {type(obs)}"

    # Sample random action on RL device (simulating policy output - skrl always uses GPU for training)
    rl_device_obj = skrl.config.torch.device
    action = 2 * torch.rand(NUM_ENVS, *env.action_space.shape, device=rl_device_obj) - 1
    print(f"  [4/6] Action created on rl_device: {rl_device_obj}, shape: {action.shape}")

    # Verify action is on RL device before calling step
    assert (
        action.device.type == rl_device_obj.type
    ), f"Action should be on {rl_device_obj} before step, got {action.device}"

    # Step environment - wrapper should:
    # 1. Accept action on rl_device
    # 2. Transfer action from rl_device to sim_device internally
    # 3. Call unwrapped env.step() with action on sim_device
    # 4. Return outputs on sim_device (skrl policy handles device transfer)
    print("  [5/6] Testing step with action on rl_device...")
    transition = env.step(action)
    print("  [5/6] Step completed - wrapper handled action device transfer")

    # Verify outputs are tensors
    # Note: skrl wrapper returns outputs on sim_device, not rl_device
    # The policy is responsible for transferring observations when needed
    print("  [6/6] Verifying outputs are on sim_device (skrl behavior)...")
    if len(transition) == 5:
        obs, reward, terminated, truncated, info = transition
        # Check observations (can be dict or tensor)
        if isinstance(obs, dict):
            assert isinstance(obs["policy"], torch.Tensor), "Observations should be tensors"
            assert (
                obs["policy"].device.type == torch.device(sim_device).type
            ), f"Observations should be on {sim_device}, got {obs['policy'].device}"
        else:
            assert isinstance(obs, torch.Tensor), "Observations should be tensors"
            assert (
                obs.device.type == torch.device(sim_device).type
            ), f"Observations should be on {sim_device}, got {obs.device}"
        assert isinstance(reward, torch.Tensor), "Rewards should be tensors"
        assert (
            reward.device.type == torch.device(sim_device).type
        ), f"Rewards should be on {sim_device}, got {reward.device}"
        assert isinstance(terminated, torch.Tensor), "Terminated should be tensors"
        assert (
            terminated.device.type == torch.device(sim_device).type
        ), f"Terminated should be on {sim_device}, got {terminated.device}"
        assert isinstance(truncated, torch.Tensor), "Truncated should be tensors"
        assert (
            truncated.device.type == torch.device(sim_device).type
        ), f"Truncated should be on {sim_device}, got {truncated.device}"
    elif len(transition) == 4:
        obs, reward, done, info = transition
        # Check observations (can be dict or tensor)
        if isinstance(obs, dict):
            assert isinstance(obs["policy"], torch.Tensor), "Observations should be tensors"
            assert (
                obs["policy"].device.type == torch.device(sim_device).type
            ), f"Observations should be on {sim_device}, got {obs['policy'].device}"
        else:
            assert isinstance(obs, torch.Tensor), "Observations should be tensors"
            assert (
                obs.device.type == torch.device(sim_device).type
            ), f"Observations should be on {sim_device}, got {obs.device}"
        assert isinstance(reward, torch.Tensor), "Rewards should be tensors"
        assert (
            reward.device.type == torch.device(sim_device).type
        ), f"Rewards should be on {sim_device}, got {reward.device}"
        assert isinstance(done, torch.Tensor), "Dones should be tensors"
        assert done.device.type == torch.device(sim_device).type, f"Dones should be on {sim_device}, got {done.device}"
    else:
        pytest.fail(f"Unexpected number of return values from step: {len(transition)}")

    # Cleanup
    print("  [6/6] Cleaning up environment...")
    env.close()
    print(f"✓ skrl test PASSED for sim_device={sim_device}, rl_device={rl_device_obj}")
    print("  Wrapper device transfer verified (skrl-specific behavior):")
    print(f"    1. Unwrapped env: expects actions on {sim_device}, returns data on {sim_device}")
    print(f"    2. Wrapper: accepts actions on {rl_device_obj} (from policy)")
    print(f"    3. Wrapper: internally transfers actions to {sim_device} for env.step()")
    print(f"    4. Wrapper: returns outputs on {sim_device} (policy handles obs device transfer)")
    print("    Note: Unlike RSL-RL/RL-Games, skrl keeps observations on sim_device")
    print("-" * 80)


# ============================================================================
# Test Functions
# ============================================================================


def test_rsl_rl_device_separation_gpu_to_gpu():
    """Test RSL-RL with GPU simulation and GPU RL (default configuration)."""
    try:
        import isaaclab_rl.rsl_rl  # noqa: F401
    except ImportError:
        pytest.skip("RSL-RL not installed")

    _test_rsl_rl_device_separation(sim_device="cuda:0", rl_device="cuda:0")


def test_rsl_rl_device_separation_gpu_to_cpu():
    """Test RSL-RL with GPU simulation and CPU RL (cross-device transfer)."""
    try:
        import isaaclab_rl.rsl_rl  # noqa: F401
    except ImportError:
        pytest.skip("RSL-RL not installed")

    _test_rsl_rl_device_separation(sim_device="cuda:0", rl_device="cpu")


def test_rl_games_device_separation_gpu_to_gpu():
    """Test RL Games with GPU simulation and GPU RL (default configuration)."""
    try:
        import isaaclab_rl.rl_games  # noqa: F401
    except ImportError:
        pytest.skip("RL Games not installed")

    _test_rl_games_device_separation(sim_device="cuda:0", rl_device="cuda:0")


def test_rl_games_device_separation_gpu_to_cpu():
    """Test RL Games with GPU simulation and CPU RL (cross-device transfer)."""
    try:
        import isaaclab_rl.rl_games  # noqa: F401
    except ImportError:
        pytest.skip("RL Games not installed")

    _test_rl_games_device_separation(sim_device="cuda:0", rl_device="cpu")


def test_sb3_device_separation_gpu():
    """Test Stable-Baselines3 with GPU simulation.

    Note: SB3 always converts to CPU/numpy, so only GPU simulation is tested.
    """
    try:
        import isaaclab_rl.sb3  # noqa: F401
    except ImportError:
        pytest.skip("Stable-Baselines3 not installed")

    _test_sb3_device_separation(sim_device="cuda:0")


def test_skrl_device_separation_gpu():
    """Test skrl with GPU simulation and GPU policy (matching devices)."""
    try:
        import skrl  # noqa: F401
    except ImportError:
        pytest.skip("skrl not installed")

    _test_skrl_device_separation(sim_device="cuda:0", rl_device="cuda:0")


def test_skrl_device_separation_cpu_to_gpu():
    """Test skrl with CPU simulation and GPU policy.

    Note: Uses skrl.config.torch.device to set the policy device to GPU
    while the environment runs on CPU.
    """
    try:
        import skrl  # noqa: F401
    except ImportError:
        pytest.skip("skrl not installed")

    _test_skrl_device_separation(sim_device="cpu", rl_device="cuda:0")
