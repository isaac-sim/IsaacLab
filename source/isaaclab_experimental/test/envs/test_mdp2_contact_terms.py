# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity of the built-in MDP2 contact terms with their MDP1 functions, and capture with contact sensors."""

from pathlib import Path

import pytest
import torch
import warp as wp
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2Env, MDP2EnvCfg
from isaaclab_experimental.mdp2 import mdp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.envs.mdp as mdp1
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.warp import ProxyArray

import isaaclab_tasks.core.velocity.mdp as velocity_mdp1

pytestmark = [pytest.mark.kitless, pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")]

ROBOT_USD = Path(__file__).parents[3] / "isaaclab_newton/test/assets/data/floating_two_leg.usda"
NUM_ENVS = 8

robot = mdp2.Entity("robot")
legs = robot.joints("LF_.*", "RH_KFE")
contact_forces = mdp2.Entity("contact_forces")
feet = contact_forces.bodies(".*_SHANK")

feet_contacts = {"net_normal_forces_w_history": contact_forces.net_normal_forces_w_history, "body_ids": feet}
feet_air_times = {
    "current_contact_time": contact_forces.current_contact_time,
    "last_air_time": contact_forces.last_air_time,
    "body_ids": feet,
}


@wp.func
def unit_command(world: int, command: wp.array2d(dtype=float)):
    for axis in range(3):
        command[world, axis] = 1.0


forward_command = mdp2.Term(torch=lambda mask, command: command.fill_(1.0), warp=unit_command, shape=("worlds", 3))
"""A unit command for every world, so that ``feet_air_time`` rewards steps."""


class MDP1Env:
    """What the MDP1 contact functions read from an environment."""

    def __init__(self, env: MDP2Env, command: torch.Tensor):
        self.scene = env.scene
        self.step_dt = env.step_dt
        self.command_manager = self
        self._command = command

    def get_command(self, name: str) -> torch.Tensor:
        return self._command


@pytest.fixture(scope="module")
def robot_with_feet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The two-legged robot, with a sphere foot on each shank and a box base, so that it touches the ground."""
    stage = Usd.Stage.Open(Usd.Stage.Open(str(ROBOT_USD)).Flatten())
    for shank in ("LF_SHANK", "RH_SHANK"):
        foot = UsdGeom.Sphere.Define(stage, f"/Robot/{shank}/foot")
        foot.CreateRadiusAttr(0.04)
        UsdGeom.Xformable(foot).AddTranslateOp().Set((0.0, 0.0, -0.25))
        UsdPhysics.CollisionAPI.Apply(foot.GetPrim())

    base = UsdGeom.Cube.Define(stage, "/Robot/base/collider")
    base.CreateSizeAttr(0.2)
    UsdPhysics.CollisionAPI.Apply(base.GetPrim())

    path = tmp_path_factory.mktemp("mdp2_terms") / "robot_with_feet.usda"
    stage.Export(str(path))
    return path


def contact_env_cfg(robot_usd: Path, backend: str, capture: bool) -> MDP2EnvCfg:
    """The two-legged robots falling onto a ground plane, with contact sensors on their shanks."""
    scene_cfg = InteractiveSceneCfg(num_envs=NUM_ENVS, env_spacing=3.0)
    scene_cfg.terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    scene_cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(robot_usd)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8), joint_pos={".*HFE": 0.4, ".*KFE": -0.8}),
        actuators={"legs": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=20.0, damping=1.0)},
    )
    scene_cfg.contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_SHANK", history_length=3, track_air_time=True
    )

    air_time = {**feet_air_times, "command": mdp2.commands.base_velocity}
    task = mdp2.TaskCfg(
        actions={"joint_effort": mdp2.DirectAction(robot.joint_effort_target, scale=5.0)},
        commands={
            "base_velocity": mdp2.Command(
                mdp.heading_velocity_command,
                inputs={"root_link_pose_w": robot.root_link_pose_w},
                params={
                    "lin_vel_x": (0.5, 1.0),
                    "lin_vel_y": (0.0, 0.0),
                    "ang_vel_z": (-1.0, 1.0),
                    "heading": (-1.0, 1.0),
                    "rel_standing_envs": 0.0,
                    "rel_heading_envs": 1.0,
                    "heading_control_stiffness": 0.5,
                },
            )
        },
        observations={
            "policy": {
                "joint_pos": mdp2.Observation(
                    mdp.joint_pos, inputs={"joint_pos": robot.joint_pos, "joint_ids": legs}, history_length=2
                ),
            }
        },
        rewards={
            "undesired_contacts": mdp2.Reward(mdp.undesired_contacts, inputs=feet_contacts, params={"threshold": 1.0}),
            "feet_air_time": mdp2.Reward(
                mdp.feet_air_time, inputs=air_time, params={"threshold": 0.1, "step_dt": mdp2.step_dt}
            ),
        },
        terminations={
            "time_out": mdp2.TimeLimit(episode_length_s=0.5),
            "illegal_contact": mdp2.Termination(mdp.illegal_contact, inputs=feet_contacts, params={"threshold": 500.0}),
        },
    )

    return MDP2EnvCfg(
        sim=SimulationCfg(dt=0.005, device="cuda:0", physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())),
        scene=scene_cfg,
        task=task,
        decimation=4,
        backend=mdp2.Backend(backend),
        seed=0,
        capture=capture,
    )


def test_contact_terms_match_mdp1(robot_with_feet: Path):
    env = MDP2Env(contact_env_cfg(robot_with_feet, "warp", capture=False))
    try:
        air_time = {**feet_air_times, "command": mdp2.commands.forward}
        builder = mdp2.MDPBuilder(env.scene_data.schema)
        builder.add_command("forward", mdp2.Command(forward_command))
        builder.add_observation(
            "undesired_contacts",
            "value",
            mdp2.Observation(mdp.undesired_contacts, inputs=feet_contacts, params={"threshold": 1.0}),
        )
        builder.add_observation(
            "feet_air_time",
            "value",
            mdp2.Observation(mdp.feet_air_time, inputs=air_time, params={"threshold": 0.1, "step_dt": mdp2.step_dt}),
        )
        builder.add_observation(
            "feet_air_time_positive_biped",
            "value",
            mdp2.Observation(
                mdp.feet_air_time_positive_biped,
                inputs={
                    "current_air_time": contact_forces.current_air_time,
                    "current_contact_time": contact_forces.current_contact_time,
                    "command": mdp2.commands.forward,
                    "body_ids": feet,
                },
                params={"threshold": 0.1, "overstay_scale": 0.5},
            ),
        )
        builder.add_observation(
            "feet_air_time_variance",
            "value",
            mdp2.Observation(
                mdp.feet_air_time_variance,
                inputs={
                    "last_air_time": contact_forces.last_air_time,
                    "last_contact_time": contact_forces.last_contact_time,
                    "command": mdp2.commands.forward,
                    "body_ids": feet,
                },
                params={"max_time": 0.5},
            ),
        )
        builder.add_observation(
            "feet_slide",
            "value",
            mdp2.Observation(
                mdp.feet_slide,
                inputs={
                    "body_com_vel_w": robot.body_com_vel_w,
                    "net_normal_forces_w_history": contact_forces.net_normal_forces_w_history,
                    "body_ids": robot.bodies(".*_SHANK"),
                    "sensor_ids": feet,
                },
            ),
        )
        builder.add_termination(
            "illegal_contact", mdp2.Termination(mdp.illegal_contact, inputs=feet_contacts, params={"threshold": 50.0})
        )

        # World configs that define a feet_slide use differently share its foot matching: with the same feet, each
        # world reads its own contacts, and different feet fail the build rather than read another foot's sensor.
        def replicated_feet_slide(*shanks: str) -> mdp2.MDPBuilder:
            replicated = mdp2.MDPBuilder(env.scene_data.schema)
            for shank, scale in zip(shanks, (1.0, 2.0)):
                config = mdp2.MDPBuilder(env.scene_data.schema)
                inputs = {
                    "body_com_vel_w": robot.body_com_vel_w,
                    "net_normal_forces_w_history": contact_forces.net_normal_forces_w_history,
                    "body_ids": robot.bodies(shank),
                    "sensor_ids": contact_forces.bodies(shank),
                }
                config.add_observation(
                    "feet_slide", "value", mdp2.Observation(mdp.feet_slide, inputs=inputs, scale=scale)
                )
                replicated.replicate(config, NUM_ENVS // 2)
            return replicated

        mixed = replicated_feet_slide("LF_SHANK", "RH_SHANK").finalize(device=env.device)
        with pytest.raises(ValueError, match="same feet"):
            mdp2.startup(mixed, mixed.state(env.scene_data))

        # Both backends read the same scene.
        models, same_feet = {}, {}
        for backend in ("torch", "warp"):
            for built, source in ((models, builder), (same_feet, replicated_feet_slide(".*_SHANK", ".*_SHANK"))):
                model = source.finalize(device=env.device, backend=backend, diagnostics=True)
                state, transition = model.state(env.scene_data), model.transition()
                mdp2.startup(model, state)  # Runs the init hooks, such as the foot matching of feet_slide.
                mdp2.reset(model, state, ProxyArray(wp.ones(NUM_ENVS, dtype=wp.bool, device=env.device)))
                built[backend] = model, state, transition
        world_scales = torch.tensor([1.0, 2.0], device=env.device).repeat_interleave(NUM_ENVS // 2)

        mdp1_env = MDP1Env(env, torch.ones(NUM_ENVS, 3, device=env.device))
        sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_SHANK")
        sensor_cfg.resolve(env.scene)
        feet_cfg = SceneEntityCfg("robot", body_names=".*_SHANK")
        feet_cfg.resolve(env.scene)

        env.reset()
        # Tilted robots land on one foot first, so that their feet step unevenly and every reward is nonzero.
        articulation = env.scene["robot"]
        tilt = torch.linspace(-0.3, 0.3, NUM_ENVS, device=env.device)
        root_pose = articulation.data.root_link_pose_w.torch.clone()
        root_pose[:, 3:] = math_utils.quat_from_euler_xyz(tilt, tilt.flip(0), torch.zeros_like(tilt))
        articulation.write_root_pose_to_sim_index(root_pose=root_pose)

        rewards = ("feet_air_time", "feet_air_time_positive_biped", "feet_air_time_variance", "feet_slide")
        nonzero = dict.fromkeys(rewards, False)
        for _ in range(40):
            env.step(torch.zeros(NUM_ENVS, env.mdp.action_dim, device=env.device))
            undesired_contacts = mdp1.undesired_contacts(mdp1_env, 1.0, sensor_cfg).float()
            illegal_contact = mdp1.illegal_contact(mdp1_env, 50.0, sensor_cfg)
            expected = {
                "feet_air_time": velocity_mdp1.feet_air_time(mdp1_env, "forward", sensor_cfg, 0.1),
                "feet_air_time_positive_biped": velocity_mdp1.feet_air_time_positive_biped(
                    mdp1_env, "forward", 0.1, sensor_cfg, overstay_scale=0.5
                ),
                "feet_air_time_variance": velocity_mdp1.feet_air_time_variance(mdp1_env, "forward", sensor_cfg, 0.5),
                "feet_slide": velocity_mdp1.feet_slide(mdp1_env, sensor_cfg, feet_cfg),
            }
            for name, value in expected.items():
                nonzero[name] = nonzero[name] or bool(value.abs().sum() > 0.0)

            for model, state, transition in models.values():
                mdp2.evaluate(model, state, transition)
                observations = transition.observations
                torch.testing.assert_close(observations["undesired_contacts"].torch[:, 0], undesired_contacts)
                for name, value in expected.items():
                    torch.testing.assert_close(observations[name].torch[:, 0], value, msg=name)
                assert torch.equal(transition.termination_terms.torch[:, 0], illegal_contact)
            for model, state, transition in same_feet.values():
                mdp2.evaluate(model, state, transition)
                feet_slide = transition.observations["feet_slide"].torch[:, 0]
                torch.testing.assert_close(feet_slide, world_scales * expected["feet_slide"])

        assert all(nonzero.values()), nonzero  # The feet landed and slid during the rollout.

        # A reset clears the contacts that ended the last episode.
        env.reset()
        scene_data = env.scene_data
        assert (scene_data.data("contact_forces", "net_normal_forces_w_history").torch == 0.0).all()
        assert (scene_data.data("contact_forces", "last_air_time").torch == 0.0).all()
    finally:
        env.close()


def test_contact_task_replays_like_eager_steps(robot_with_feet: Path):
    """Captured steps, with sensors refreshed in the graph, give the results of eager steps, up to contact noise."""
    runs = []
    for capture in (True, False):
        env = MDP2Env(contact_env_cfg(robot_with_feet, "warp", capture))
        try:
            generator = torch.Generator(device=env.device).manual_seed(0)
            env.reset()
            history = []
            for _ in range(60):
                actions = 2.0 * torch.rand(NUM_ENVS, env.mdp.action_dim, device=env.device, generator=generator) - 1.0
                env.step(actions)
                transition = env.transition
                history.append(
                    torch.cat(
                        [
                            transition.rewards.torch[:, None],
                            transition.done.torch[:, None].float(),
                            transition.observations["policy"].torch,
                        ],
                        dim=-1,
                    )
                )

            assert env.is_captured == capture
            runs.append(torch.stack(history))
        finally:
            env.close()

    # Contact physics is not bitwise deterministic, even between eager runs: it differs by about 1e-6.
    assert runs[0][:, :, 0].abs().sum() > 0.0
    torch.testing.assert_close(runs[0], runs[1], atol=1e-4, rtol=1e-4)
