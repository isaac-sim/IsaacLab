# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pipe behavior at the observation manager boundary."""

from builtins import ExceptionGroup
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import (
    ManagerTermBase,
    ManagerTermBaseCfg,
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
)
from isaaclab.utils import configclass, tensor_processors
from isaaclab.utils.tensor_processors import DictTensorProcessor, DictTensorProcessorCfg
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.unit


def test_environment_releases_simulation_when_observation_cleanup_fails():
    """An operation teardown error must not strand the environment or its simulation singleton."""
    events = []

    def close_observations():
        events.append("observations")
        raise RuntimeError("Operation cleanup failed")

    env = ManagerBasedEnv.__new__(ManagerBasedEnv)
    env._is_closed = False
    env._window = None
    env.sim = SimpleNamespace(stop=lambda: events.append("stop"), clear_instance=lambda: events.append("clear"))
    env.action_manager = object()
    env.observation_manager = SimpleNamespace(close=close_observations)
    env.event_manager = object()
    env.recorder_manager = object()
    env.scene = object()
    env.obs_buf = {"policy": torch.ones(1)}
    with pytest.raises(RuntimeError, match="Operation cleanup failed"):
        env.close()
    assert events == ["stop", "observations", "clear"]
    assert env.obs_buf == {}
    env.close()
    assert events == ["stop", "observations", "clear"]


@wp.kernel
def _gain(source: wp.array4d(dtype=wp.float32), gain: float, result: wp.array4d(dtype=wp.float32)):
    n, y, x, c = wp.tid()
    result[n, y, x, c] = source[n, y, x, c] * gain


@wp.kernel
def _depth_weighted_half(
    color: wp.array4d(dtype=wp.float32),
    depth: wp.array4d(dtype=wp.float32),
    result: wp.array4d(dtype=wp.float32),
):
    n, y, x, c = wp.tid()
    result[n, y, x, c] = color[n, y, 2 * x, c] * depth[n, y, 2 * x, 0]


class _Gain(DictTensorProcessor):
    def __init__(self, cfg, data):
        super().__init__(cfg, data)
        self.outputs[cfg.outputs[0]] = ProxyArray(wp.empty_like(data[cfg.inputs[0]].warp))

    def __call__(self, data):
        output = self.outputs[self.cfg.outputs[0]]
        wp.launch(
            _gain,
            dim=output.shape,
            inputs=[data[self.cfg.inputs[0]].warp, self.cfg.params.get("gain", 2.0), output.warp],
            device=output.device,
        )
        return dict(self.outputs)


def _read_ranges(env):
    return env.ranges


def _env(**kwargs):
    return SimpleNamespace(num_envs=2, device="cpu", sim=SimpleNamespace(is_playing=lambda: True), **kwargs)


class _NestedImage(ManagerTermBase):
    """Compose an image term through the manager's supported nested configuration containers."""

    def __call__(self, env, terms, *, out=None):
        term = terms[0]["image"]
        params = {name: value for name, value in term.params.items() if name != "out"}
        return term.func(env, **params, out=out)

    def reset(self, env_ids=None):
        self.cfg.params["terms"][0]["image"].func.reset(env_ids)

    def close(self):
        self._env.lifecycle.append(("outer_close",))


def test_pipe_modifier_uses_preceding_modifier_output():
    """A pipe consumes the current non-image observation after preceding tensor_processors."""
    ranges = torch.ones((2, 4))
    preparations = []

    class RangeGain(DictTensorProcessor):
        def __init__(self, cfg, data):
            super().__init__(cfg, data)
            preparations.append(data["ranges"].shape)
            self.outputs["ranges"] = ProxyArray(wp.empty_like(data["ranges"].warp))

        def __call__(self, data):
            torch.mul(data["ranges"].torch, self.cfg.params["gain"], out=self.outputs["ranges"].torch)
            return dict(self.outputs)

    @configclass
    class Policy(ObservationGroupCfg):
        concatenate_terms = False
        ranges = ObservationTermCfg(
            func=_read_ranges,
            tensor_processors=[
                tensor_processors.TensorProcessorCfg(func=tensor_processors.bias, params={"value": 3.0}),
                tensor_processors.DictTensorProcessorAdapterCfg(
                    pipeline=[
                        DictTensorProcessorCfg(
                            class_type=RangeGain,
                            inputs=["ranges"],
                            outputs=["ranges"],
                            params={"gain": 2.0},
                        )
                    ],
                    input_name="ranges",
                    output_name="ranges",
                ),
            ],
        )

    env = _env(ranges=ranges)
    manager = ObservationManager({"policy": Policy()}, env)
    manager.reset([1])
    assert preparations == []
    first = manager.compute()["policy"]["ranges"]
    torch.testing.assert_close(first, torch.full_like(ranges, 8.0))
    torch.testing.assert_close(ranges, torch.ones_like(ranges))
    env.ranges = torch.full_like(ranges, 2.0)
    torch.testing.assert_close(manager.compute()["policy"]["ranges"], torch.full_like(ranges, 10.0))
    torch.testing.assert_close(first, torch.full_like(ranges, 8.0))
    assert preparations == [tuple(ranges.shape)]
    manager.close()


def test_pipe_modifier_retries_failed_preparation_and_stays_closed():
    """A rejected output is released; a later valid preparation works only before close."""
    image = torch.ones((2, 2, 4, 3))
    lifecycle = []
    valid_shape = False

    class TrackingGain(_Gain):
        def __init__(self, cfg, data):
            super().__init__(cfg, data)
            lifecycle.append("prepare")
            if not valid_shape:
                self.outputs["rgb"] = ProxyArray(wp.empty((2, 2, 2, 3), dtype=wp.float32, device="cpu"))

        def close(self):
            lifecycle.append("close")

    cfg = tensor_processors.DictTensorProcessorAdapterCfg(
        pipeline=[DictTensorProcessorCfg(class_type=TrackingGain, inputs=["rgb"], outputs=["rgb"])]
    )
    modifier = tensor_processors.DictTensorProcessorAdapter(cfg, tuple(image.shape), "cpu")
    modifier.reset([1])
    with pytest.raises(ValueError, match="input shape"):
        modifier(image[:, :, ::2])
    assert lifecycle == []
    with pytest.raises(ValueError, match="must preserve shape"):
        modifier(image)
    assert lifecycle == ["prepare", "close"]
    valid_shape = True
    torch.testing.assert_close(modifier(image), image * 2)
    modifier.close()
    modifier.close()
    with pytest.raises(RuntimeError, match="closed"):
        modifier(image)
    assert lifecycle == ["prepare", "close", "prepare", "close"]

    unused_modifier = tensor_processors.DictTensorProcessorAdapter(cfg, tuple(image.shape), "cpu")
    unused_modifier.close()
    with pytest.raises(RuntimeError, match="closed"):
        unused_modifier(image)
    assert lifecycle == ["prepare", "close", "prepare", "close"]


def test_processed_sensor_declares_private_inputs_and_reports_transformed_shape():
    """Preparation resolves private AOVs before binding; manager dimensions describe the processed output."""
    lifecycle = []

    class DepthWeighted(DictTensorProcessor):
        def __init__(self, cfg, data):
            super().__init__(cfg, data)
            color = data["rgb_hdr"]
            n, h, w, c = color.shape
            self.outputs["rgb"] = ProxyArray(wp.empty((n, h, w // 2, c), dtype=wp.float32, device=color.device))

        def __call__(self, data):
            output = self.outputs["rgb"]
            wp.launch(
                _depth_weighted_half,
                dim=output.shape,
                inputs=[data["rgb_hdr"].warp, data["depth"].warp, output.warp],
                device=output.device,
            )
            return dict(self.outputs)

        def reset(self, env_ids=None):
            lifecycle.append(("reset", env_ids))

        def close(self):
            lifecycle.append(("close",))
            raise ValueError("operation close failed")

    radiance = torch.full((2, 2, 4, 3), 2.0)
    depth = torch.tensor([1.0, 9.0, 3.0, 9.0]).reshape(1, 1, 4, 1).expand(2, 2, 4, 1).contiguous()
    requests = []
    camera = SimpleNamespace(
        request_render_inputs=lambda names: requests.append(set(names)),
        render_outputs={"rgb_hdr": ProxyArray(wp.from_torch(radiance)), "depth": ProxyArray(wp.from_torch(depth))},
    )
    sensors = {"camera": camera}
    env = _env(scene=SimpleNamespace(sensors=sensors, keys=sensors.keys), lifecycle=lifecycle)

    @configclass
    class Policy(ObservationGroupCfg):
        concatenate_terms = False
        image = ObservationTermCfg(
            func=_NestedImage,
            params={
                "terms": [
                    {
                        "image": ManagerTermBaseCfg(
                            func="isaaclab.envs.mdp:processed_sensor",
                            params={
                                "pipeline": [
                                    DictTensorProcessorCfg(
                                        class_type=DepthWeighted,
                                        inputs=["rgb_hdr", "depth"],
                                        outputs=["rgb"],
                                    )
                                ]
                            },
                        )
                    }
                ]
            },
        )

    @configclass
    class Observations:
        policy = Policy()

    cfg = Observations()
    ObservationManager.prepare_scene(cfg, env)
    assert requests == [{"rgb_hdr", "depth"}]
    manager = ObservationManager(cfg, env)
    assert manager.group_obs_dim["policy"] == [(2, 2, 3)]
    result = manager.compute()["policy"]["image"]
    expected = torch.tensor([2.0, 6.0]).reshape(1, 1, 2, 1).expand(2, 2, 2, 3)
    torch.testing.assert_close(result, expected)
    radiance.fill_(4.0)
    torch.testing.assert_close(manager.compute()["policy"]["image"], expected * 2)
    torch.testing.assert_close(result, expected)
    manager.reset([1])
    with pytest.raises(ExceptionGroup, match="Failed to close dictionary processors") as error:
        manager.close()
    assert str(error.value.exceptions[0]) == "operation close failed"
    manager.close()
    assert lifecycle[-3:] == [("reset", [1]), ("close",), ("outer_close",)]
