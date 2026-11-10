# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image transfer processors running in a camera: rendered depth in, generated rgb out."""

import pytest
import torch
from isaaclab_experimental.image_transfer import FunctionProcessorCfg, ImageTransferProcessorCfg, depth_to_control

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import CAPTURE_ENV_MASK_NAME, CameraCfg
from isaaclab.sim import BackendCfg, PinholeCameraCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import configclass

pytestmark = [pytest.mark.integration, pytest.mark.rendering]


class _InvertStream:
    """Return each control inverted and record the chunks, resets, and closing."""

    def __init__(self, model):
        self.model = model

    def step(self, controls, reset_rows, seeds):
        self.model.steps.append(([chunk.shape[0] for chunk in controls], reset_rows))
        return [255 - chunk for chunk in controls]

    def close(self):
        self.model.closed += 1


class _InvertModel:
    instances = []

    def __init__(self, cfg):
        self.steps = []
        self.closed = 0
        _InvertModel.instances.append(self)

    def open_stream(self, num_views, seeds):
        self.num_views = num_views
        return _InvertStream(self)

    def close(self):
        pass


@configclass
class _InvertModelCfg(BackendCfg):
    class_type: type = _InvertModel


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_camera_publishes_generated_rgb_from_rendered_depth(device):
    """The camera renders only depth, runs the processors per capture, and resets and closes them."""
    pytest.importorskip("isaaclab_newton")
    pytest.importorskip("newton")
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    _InvertModel.instances.clear()
    processors = [
        FunctionProcessorCfg(
            function=depth_to_control,
            params={"near": 0.5, "far": 3.0},
            inputs=["distance_to_image_plane"],
            outputs=["control"],
        ),
        ImageTransferProcessorCfg(backend=_InvertModelCfg(), initial_frames=1, update_frames=2),
    ]
    sim_cfg = sim_utils.SimulationCfg(
        device=device, gravity=(0.0, 0.0, 0.0), physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())
    )
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene_cfg = InteractiveSceneCfg(num_envs=2, env_spacing=2.0)
        scene_cfg.cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.3, 0.3, 0.3),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                mass_props=sim_utils.MassCfg(mass=1.0),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            ),
        )
        scene_cfg.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            width=32,
            height=32,
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 2.0), convention="opengl"),
            spawn=PinholeCameraCfg(clipping_range=(0.1, 10.0)),
            renderer_cfg=NewtonWarpRendererCfg(),
            data_types=["rgb"],
            tensor_processors=processors,
        )
        scene = InteractiveScene(scene_cfg)
        camera = scene["camera"]
        sim.reset()

        def capture() -> torch.Tensor:
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            camera.update(sim.get_physics_dt(), force_recompute=True)
            return camera.data.output["rgb"].torch

        rgb = capture()
        model = _InvertModel.instances[0]
        assert model.num_views == 2 and model.steps == [([1, 1], ())]
        # The cube top at 1.7 m is a gray control, the floor beyond far a black one; the model inverts them.
        assert rgb.shape == (2, 32, 32, 3) and rgb.dtype == torch.uint8
        assert 0 < int(rgb[0, 16, 16, 0]) < 255 and int(rgb[0, 0, 0, 0]) == 255
        assert "control" in camera.data.output and CAPTURE_ENV_MASK_NAME not in camera.data.output
        capture()
        assert len(model.steps) == 1  # the update chunk needs two captures
        camera.reset(env_ids=[1])
        capture()
        capture()
        # Environment 0 continues with two captures; environment 1 restarts with its newest one.
        assert model.steps[-1] == ([2, 1], (1,))
    assert model.closed == 1


def _rendered_to_control(data: torch.Tensor) -> torch.Tensor:
    return data[..., :3].contiguous()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_camera_generates_rgb_from_a_control_made_of_the_rendered_rgb(device):
    """A control computed from the rendered image, as edge and blur controls are: the published rgb is generated."""
    pytest.importorskip("isaaclab_newton")
    pytest.importorskip("newton")
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    _InvertModel.instances.clear()
    processors = [
        FunctionProcessorCfg(function=_rendered_to_control, inputs=["rgb"], outputs=["control"]),
        ImageTransferProcessorCfg(backend=_InvertModelCfg(), initial_frames=1, update_frames=1),
    ]
    sim_cfg = sim_utils.SimulationCfg(
        device=device, gravity=(0.0, 0.0, 0.0), physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())
    )
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene_cfg = InteractiveSceneCfg(num_envs=2, env_spacing=2.0)
        scene_cfg.cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.3, 0.3, 0.3),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                mass_props=sim_utils.MassCfg(mass=1.0),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            ),
        )
        scene_cfg.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            width=32,
            height=32,
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 2.0), convention="opengl"),
            spawn=PinholeCameraCfg(clipping_range=(0.1, 10.0)),
            renderer_cfg=NewtonWarpRendererCfg(),
            data_types=["rgb"],
            tensor_processors=processors,
        )
        scene = InteractiveScene(scene_cfg)
        camera = scene["camera"]
        sim.reset()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())
        camera.update(sim.get_physics_dt(), force_recompute=True)
        output = camera.data.output
        rendered, published = output["control"].torch, output["rgb"].torch

        assert published.shape == (2, 32, 32, 3) and published.dtype == torch.uint8
        # The rendered image has content (the cube differs from the floor), and the published one is its inverse.
        assert rendered.float().std() > 0
        assert torch.equal(published, 255 - rendered)
