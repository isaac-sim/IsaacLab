# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
    - RSL-RL: TensorDict observations, device separation via OnPolicyRunner (agent_cfg.device)
        * Wrapper returns data on sim_device, Runner handles transfers to rl_device
    - RL Games: Dict observations, explicit rl_device parameter in wrapper
        * Wrapper transfers data from sim_device to rl_device
    - Stable-Baselines3: Numpy arrays (CPU-only by design)
        * Wrapper converts tensors to/from numpy on CPU
    - skrl: Dict observations, uses skrl.config.torch.device for RL device
        * Wrapper keeps observations on sim_device, only transfers actions

"""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Test environment - use Cartpole as it's simple and fast
TEST_ENV = "Isaac-Cartpole-v0"
NUM_ENVS = 4


def _create_env(sim_device: str):
    """Create and initialize a test environment.

    Args:
        sim_device: Device for simulation (e.g., "cuda:0", "cpu")

    Returns:
        Initialized gym environment
    """
    # Create a new stage
    sim_utils.create_new_stage()
    # Reset the rtx sensors setting to False
    from isaaclab.app.settings_manager import get_settings_manager

    get_settings_manager().set_bool("/isaaclab/render/rtx_sensors", False)

    try:
        env_cfg = parse_env_cfg(TEST_ENV, device=sim_device, num_envs=NUM_ENVS)
        env = gym.make(TEST_ENV, cfg=env_cfg)
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
    return env


def _verify_unwrapped_env(env, sim_device: str):
    """Verify unwrapped environment operates entirely on sim_device.

    Args:
        env: Unwrapped gym environment
        sim_device: Expected simulation device
    """
    assert env.unwrapped.device == sim_device, (
        f"Environment device mismatch: expected {sim_device}, got {env.unwrapped.device}"
    )

    # Verify reset returns data on sim device
    obs_dict, _ = env.reset()
    for key, value in obs_dict.items():
        if isinstance(value, torch.Tensor):
            assert value.device.type == torch.device(sim_device).type, (
                f"Unwrapped env obs '{key}' should be on {sim_device}, got {value.device}"
            )

    # Verify step returns data on sim device
    action_space = env.unwrapped.single_action_space
    test_action = torch.zeros(NUM_ENVS, action_space.shape[0], device=sim_device)
    obs_dict, rew, term, trunc, extras = env.step(test_action)
    assert rew.device.type == torch.device(sim_device).type, (
        f"Unwrapped env rewards should be on {sim_device}, got {rew.device}"
    )
    assert term.device.type == torch.device(sim_device).type, (
        f"Unwrapped env terminated should be on {sim_device}, got {term.device}"
    )


def _verify_tensor_device(data, expected_device: str, name: str):
    """Verify tensor or dict of tensors is on expected device.

    Args:
        data: Tensor, dict of tensors, or numpy array
        expected_device: Expected device string
        name: Name for error messages
    """
    if isinstance(data, torch.Tensor):
        assert data.device.type == torch.device(expected_device).type, (
            f"{name} should be on {expected_device}, got {data.device}"
        )
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                assert value.device.type == torch.device(expected_device).type, (
                    f"{name}['{key}'] should be on {expected_device}, got {value.device}"
                )


def _test_rsl_rl_device_separation(sim_device: str, rl_device: str):
    """Helper function to test RSL-RL with specified device configuration.

    Note: RSL-RL device separation is handled by the OnPolicyRunner, not the wrapper.
    The wrapper returns observations on sim_device, and the runner handles device transfers.
    This test verifies the wrapper works correctly when actions come from a different device.

    Args:
        sim_device: Device for simulation (e.g., "cuda:0", "cpu")
        rl_device: Device for RL agent (e.g., "cuda:0", "cpu") - where policy generates actions
    """
    from tensordict import TensorDict

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env = _create_env(sim_device)
    _verify_unwrapped_env(env, sim_device)

    # Create wrapper - it uses sim_device, runner handles rl_device
    env = RslRlVecEnvWrapper(env)
    assert env.device == sim_device, f"Wrapper device should be {sim_device}"

    # Test reset - wrapper returns observations on sim_device
    obs, extras = env.reset()
    assert isinstance(obs, TensorDict), f"Expected TensorDict, got {type(obs)}"
    _verify_tensor_device(obs, sim_device, "Observation")

    # Test step with action from RL device (simulating policy output)
    # The wrapper should handle transferring action to sim_device internally
    action = 2 * torch.rand(env.action_space.shape, device=rl_device) - 1
    obs, reward, dones, extras = env.step(action)

    # Verify outputs are on sim_device (runner would transfer to rl_device)
    assert isinstance(obs, TensorDict), f"Expected TensorDict, got {type(obs)}"
    _verify_tensor_device(obs, sim_device, "Step observation")
    _verify_tensor_device(reward, sim_device, "Reward")
    _verify_tensor_device(dones, sim_device, "Dones")

    env.close()


def _test_rl_games_device_separation(sim_device: str, rl_device: str):
    """Helper function to test RL Games with specified device configuration.

    Args:
        sim_device: Device for simulation (e.g., "cuda:0", "cpu")
        rl_device: Device for RL agent (e.g., "cuda:0", "cpu")
    """
    from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

    env = _create_env(sim_device)
    _verify_unwrapped_env(env, sim_device)

    # Create wrapper
    env = RlGamesVecEnvWrapper(env, rl_device=rl_device, clip_obs=10.0, clip_actions=1.0)

    # Test reset
    obs = env.reset()
    _verify_tensor_device(obs, rl_device, "Observation")

    # Test step with action on RL device
    action = 2 * torch.rand(NUM_ENVS, *env.action_space.shape, device=rl_device) - 1
    obs, reward, dones, info = env.step(action)

    # Verify outputs are on RL device
    _verify_tensor_device(obs, rl_device, "Observation")
    _verify_tensor_device(reward, rl_device, "Reward")
    _verify_tensor_device(dones, rl_device, "Dones")

    env.close()


def _test_sb3_device_separation(sim_device: str):
    """Helper function to test Stable-Baselines3 with specified device configuration.

    Note: SB3 always converts to CPU/numpy, so we don't test rl_device parameter.

    Args:
        sim_device: Device for simulation (e.g., "cuda:0", "cpu")
    """
    import numpy as np

    from isaaclab_rl.sb3 import Sb3VecEnvWrapper

    env = _create_env(sim_device)
    _verify_unwrapped_env(env, sim_device)

    # Create wrapper
    env = Sb3VecEnvWrapper(env)

    # Test reset - SB3 should return numpy arrays
    obs = env.reset()
    assert isinstance(obs, np.ndarray), f"SB3 observations should be numpy arrays, got {type(obs)}"

    # Test step with numpy action
    action = 2 * np.random.rand(env.num_envs, *env.action_space.shape) - 1
    obs, reward, done, info = env.step(action)

    # Verify outputs are numpy arrays
    assert isinstance(obs, np.ndarray), f"Observations should be numpy arrays, got {type(obs)}"
    assert isinstance(reward, np.ndarray), f"Rewards should be numpy arrays, got {type(reward)}"
    assert isinstance(done, np.ndarray), f"Dones should be numpy arrays, got {type(done)}"

    env.close()


def _test_skrl_device_separation(sim_device: str, rl_device: str):
    """Helper function to test skrl with specified device configuration.

    Note: skrl uses skrl.config.torch.device for device configuration.
    Observations remain on sim_device; only actions are transferred from rl_device.

    Args:
        sim_device: Device for simulation (e.g., "cuda:0", "cpu")
        rl_device: Device for RL agent (e.g., "cuda:0", "cpu")
    """
    try:
        import skrl
        from skrl.envs.wrappers.torch import wrap_env
    except ImportError:
        pytest.skip("skrl not installed")

    # Configure skrl device
    skrl.config.torch.device = torch.device(rl_device)

    env = _create_env(sim_device)
    _verify_unwrapped_env(env, sim_device)

    # Wrap with skrl
    env = wrap_env(env, wrapper="isaaclab")

    # Test reset
    obs, info = env.reset()
    assert isinstance(obs, (dict, torch.Tensor)), f"Observations should be dict or tensor, got {type(obs)}"

    # Test step with action on RL device
    action = 2 * torch.rand(NUM_ENVS, *env.action_space.shape, device=skrl.config.torch.device) - 1
    transition = env.step(action)

    # Verify outputs - skrl keeps them on sim_device
    if len(transition) == 5:
        obs, reward, terminated, truncated, info = transition
        _verify_tensor_device(obs, sim_device, "Observation")
        _verify_tensor_device(reward, sim_device, "Reward")
        _verify_tensor_device(terminated, sim_device, "Terminated")
        _verify_tensor_device(truncated, sim_device, "Truncated")
    elif len(transition) == 4:
        obs, reward, done, info = transition
        _verify_tensor_device(obs, sim_device, "Observation")
        _verify_tensor_device(reward, sim_device, "Reward")
        _verify_tensor_device(done, sim_device, "Done")
    else:
        pytest.fail(f"Unexpected number of return values from step: {len(transition)}")

    env.close()


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
