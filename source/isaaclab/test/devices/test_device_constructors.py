# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import json

import pytest
import torch

# Import device classes to test
from isaaclab.devices import HaplyDevice, HaplyDeviceCfg

pytestmark = pytest.mark.integration


@pytest.fixture
def mock_environment(mocker):
    """Set up common mock objects for tests."""
    # Mock Haply WebSocket
    websockets_mock = mocker.MagicMock()
    websocket_mock = mocker.MagicMock()
    websockets_mock.connect.return_value.__aenter__.return_value = websocket_mock

    return {"websockets": websockets_mock, "websocket": websocket_mock}


def test_haply_constructors(mock_environment, mocker):
    """Test constructor for HaplyDevice."""
    # Test config-based constructor
    config = HaplyDeviceCfg(
        websocket_uri="ws://localhost:10001",
        pos_sensitivity=1.5,
        data_rate=250.0,
    )

    # Mock the websockets module and asyncio
    device_mod = importlib.import_module("isaaclab.devices.haply.se3_haply")
    mocker.patch.dict("sys.modules", {"websockets": mock_environment["websockets"]})
    mocker.patch.object(device_mod, "websockets", mock_environment["websockets"])

    # Mock asyncio to prevent actual async operations
    asyncio_mock = mocker.MagicMock()
    mocker.patch.object(device_mod, "asyncio", asyncio_mock)

    # Mock threading to prevent actual thread creation
    threading_mock = mocker.MagicMock()
    thread_instance = mocker.MagicMock()
    threading_mock.Thread.return_value = thread_instance
    thread_instance.is_alive.return_value = False
    mocker.patch.object(device_mod, "threading", threading_mock)

    # Mock time.time() for connection timeout simulation
    time_mock = mocker.MagicMock()
    time_mock.time.side_effect = [0.0, 0.1, 0.2, 0.3, 6.0]  # Will timeout
    mocker.patch.object(device_mod, "time", time_mock)

    # Create sample WebSocket response data
    ws_response = {
        "inverse3": [
            {
                "device_id": "test_inverse3_123",
                "state": {"cursor_position": {"x": 0.1, "y": 0.2, "z": 0.3}},
            }
        ],
        "wireless_verse_grip": [
            {
                "device_id": "test_versegrip_456",
                "state": {
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    "buttons": {"a": False, "b": False, "c": False},
                },
            }
        ],
    }

    # Configure websocket mock to return JSON data
    mock_environment["websocket"].recv = mocker.AsyncMock(return_value=json.dumps(ws_response))
    mock_environment["websocket"].send = mocker.AsyncMock()

    # The constructor will raise RuntimeError due to timeout, which is expected in test
    with pytest.raises(RuntimeError, match="Failed to connect both Inverse3 and VerseGrip devices"):
        haply = HaplyDevice(config)

    # Now test successful connection by mocking time to not timeout
    time_mock.time.side_effect = [0.0, 0.1, 0.2, 0.3, 0.4]  # Won't timeout

    # Mock the connection status
    mocker.patch.object(device_mod.HaplyDevice, "_start_websocket_thread")
    haply = device_mod.HaplyDevice.__new__(device_mod.HaplyDevice)
    haply._sim_device = config.sim_device
    haply.websocket_uri = config.websocket_uri
    haply.pos_sensitivity = config.pos_sensitivity
    haply.data_rate = config.data_rate
    haply.limit_force = config.limit_force
    haply.connected = True
    haply.inverse3_device_id = "test_inverse3_123"
    haply.verse_grip_device_id = "test_versegrip_456"
    haply.data_lock = threading_mock.Lock()
    haply.force_lock = threading_mock.Lock()
    haply._connected_lock = threading_mock.Lock()
    haply._additional_callbacks = {}
    haply._prev_buttons = {"a": False, "b": False, "c": False}
    haply._websocket_thread = None  # Initialize to prevent AttributeError in __del__
    haply.running = True
    haply.cached_data = {
        "position": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32).numpy(),
        "quaternion": torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32).numpy(),
        "buttons": {"a": False, "b": False, "c": False},
        "inverse3_connected": True,
        "versegrip_connected": True,
    }
    haply.feedback_force = {"x": 0.0, "y": 0.0, "z": 0.0}

    # Test advance() returns expected type
    result = haply.advance()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (10,)  # (pos_x, pos_y, pos_z, qx, qy, qz, qw, btn_a, btn_b, btn_c)

    # Test push_force with tensor (single force vector)
    forces_within = torch.tensor([[1.0, 1.5, -0.5]], dtype=torch.float32)
    position_zero = torch.tensor([0], dtype=torch.long)
    haply.push_force(forces_within, position_zero)
    assert haply.feedback_force["x"] == pytest.approx(1.0)
    assert haply.feedback_force["y"] == pytest.approx(1.5)
    assert haply.feedback_force["z"] == pytest.approx(-0.5)

    # Test push_force with tensor (force limiting, default limit is 2.0 N)
    forces_exceed = torch.tensor([[5.0, -10.0, 1.5]], dtype=torch.float32)
    haply.push_force(forces_exceed, position_zero)
    assert haply.feedback_force["x"] == pytest.approx(2.0)
    assert haply.feedback_force["y"] == pytest.approx(-2.0)
    assert haply.feedback_force["z"] == pytest.approx(1.5)

    # Test push_force with position tensor (single index)
    forces_multi = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.8, -0.3], [0.1, 0.2, 0.3]], dtype=torch.float32)
    position_single = torch.tensor([1], dtype=torch.long)
    haply.push_force(forces_multi, position=position_single)
    assert haply.feedback_force["x"] == pytest.approx(0.5)
    assert haply.feedback_force["y"] == pytest.approx(0.8)
    assert haply.feedback_force["z"] == pytest.approx(-0.3)

    # Test push_force with position tensor (multiple indices)
    position_multi = torch.tensor([0, 2], dtype=torch.long)
    haply.push_force(forces_multi, position=position_multi)
    # Should sum forces[0] and forces[2]: [1.0+0.1, 2.0+0.2, 3.0+0.3] = [1.1, 2.2, 3.3]
    # But clipped to [-2.0, 2.0]: [1.1, 2.0, 2.0]
    assert haply.feedback_force["x"] == pytest.approx(1.1)
    assert haply.feedback_force["y"] == pytest.approx(2.0)
    assert haply.feedback_force["z"] == pytest.approx(2.0)

    # Test reset functionality
    haply.reset()
    assert haply.feedback_force == {"x": 0.0, "y": 0.0, "z": 0.0}
