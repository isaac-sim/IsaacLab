# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a chase-camera clip of one checkpoint, under the evaluation conditions.

Neither of the obvious routes works in this environment. ``isaaclab play --video`` without an
explicit visualizer pre-injects a headless Kit one, and Kit segfaults here on an Isaac Sim / USD
version mismatch. Adding a ``CameraCfg`` sensor and rendering it with Newton's Warp rasteriser
produces frames of the terrain with no robot in them, from any camera position and with both
``load_visual_shapes`` and ``make_uninstanceable`` on.

What does work is the Newton GL visualizer, which draws the articulation and exposes both
:meth:`set_camera_view` and :meth:`render_rgb_array`. Its camera takes absolute world coordinates
and has no follow mode, so this drives it from the robot's own pose every step -- without that the
default camera looks at the world origin while the robot stands tens of metres away on its terrain
tile, which is what an unaimed recording shows.

The rollout matches :mod:`eval_checkpoints`: deterministic policy, terrain curriculum off, the same
evaluation seed. A clip recorded under training settings would show a different robot on different
terrain from the one the numbers describe.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True, help="Task the checkpoint was trained on.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to record.")
parser.add_argument("--out", type=str, required=True, help="Output .mp4 path.")
parser.add_argument("--steps", type=int, default=500, help="Control steps to record.")
parser.add_argument("--seed", type=int, default=12345, help="Evaluation seed; must differ from training's.")
parser.add_argument("--num_envs", type=int, default=1, help="Environments to simulate.")
parser.add_argument(
    "--env_id",
    type=int,
    default=0,
    help="Environment to follow. Each environment sits on a different column of the terrain grid, so"
    " this is how a clip is pointed at a particular sub-terrain.",
)
parser.add_argument("--back", type=float, default=2.6, help="How far behind the robot the camera trails [m].")
parser.add_argument("--side", type=float, default=1.6, help="Lateral camera offset [m]; 0 is straight behind.")
parser.add_argument("--up", type=float, default=1.3, help="Camera height above the robot's root [m].")
parser.add_argument(
    "--smoothing",
    type=float,
    default=0.92,
    help="Low-pass coefficient for the camera pose, 0 to follow the root exactly and 1 to freeze."
    " The root bobs at every footfall and its heading swings with each step; passing that straight"
    " to the camera makes the background shake.",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# The GL visualizer is the default because it both draws the robot and hands back frames without
# Kit. ``--viz`` (added by AppLauncher) overrides it: Kit renders far better when it works, and
# whether it works depends on the physics backend and the Isaac Sim install, so it is worth trying
# rather than ruling out.
args_cli.visualizer = args_cli.visualizer or ["newton_gl"]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent, play_mode=True)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if getattr(env_cfg, "curriculum", None) is not None:
        for term in [t for t in vars(env_cfg.curriculum) if not t.startswith("_")]:
            setattr(env_cfg.curriculum, term, None)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    import importlib.metadata as metadata  # noqa: PLC0415

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    runner_cls = OnPolicyRunner if agent_cfg.class_name == "OnPolicyRunner" else DistillationRunner
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if "kit" in (args_cli.visualizer or []):
        # ``omni.replicator.core`` ships as a Kit extension rather than an importable module, so the
        # Kit visualizer's offscreen capture fails with ModuleNotFoundError until the running app
        # enables it. Doing it here keeps the GL path free of any Isaac Sim import.
        # Isaac Sim's own helper lives in an extension too, so go straight to Kit's extension
        # manager, which is importable as soon as the app is running.
        import omni.kit.app  # noqa: PLC0415

        manager = omni.kit.app.get_app().get_extension_manager()
        ok = manager.set_extension_enabled_immediate("omni.replicator.core", True)
        # Enabling is asynchronous: the extension's Python module only appears after the app pumps.
        for _ in range(10):
            simulation_app.update()
        try:
            import omni.replicator.core  # noqa: F401, PLC0415

            print(f"[record] omni.replicator.core enabled={ok}, import OK")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"omni.replicator.core did not become importable (enable_extension returned {ok});"
                " the Kit visualizer cannot hand back frames without it"
            ) from exc

    visualizers = getattr(env.unwrapped.sim, "visualizers", [])
    viz = next((v for v in visualizers if hasattr(v, "render_rgb_array")), None)
    if viz is None:
        raise RuntimeError(f"no frame-producing visualizer among {[type(v).__name__ for v in visualizers]}")

    robot = env.unwrapped.scene["robot"]

    # Camera state, low-pass filtered. Driving the camera straight off the root pose puts every
    # footfall and every step's yaw swing into the camera, which reads as a shaking background --
    # the robot looks steady and the world jitters around it. Filtering the look-at point and the
    # heading leaves the robot's own motion visible while the camera itself travels smoothly.
    smooth: dict[str, torch.Tensor] = {}

    def aim() -> None:
        """Park the camera behind, beside and above the robot, looking at its chest."""
        i = args_cli.env_id
        pos = robot.data.root_pos_w.torch[i]
        quat = robot.data.root_quat_w.torch[i : i + 1]
        from isaaclab.utils.math import quat_apply  # noqa: PLC0415

        fwd = quat_apply(quat, torch.tensor([[1.0, 0.0, 0.0]], device=pos.device))[0]
        fwd = torch.tensor([fwd[0], fwd[1], 0.0], device=pos.device)
        fwd = fwd / fwd.norm().clamp_min(1e-6)

        if not smooth:
            smooth["target"] = pos.clone()
            smooth["fwd"] = fwd.clone()
        else:
            a = args_cli.smoothing
            smooth["target"] = a * smooth["target"] + (1.0 - a) * pos
            smooth["fwd"] = a * smooth["fwd"] + (1.0 - a) * fwd
        target = smooth["target"]
        fwd = smooth["fwd"] / smooth["fwd"].norm().clamp_min(1e-6)

        left = torch.tensor([-fwd[1], fwd[0], 0.0], device=pos.device)
        eye = target - args_cli.back * fwd + args_cli.side * left
        viz.set_camera_view(
            [eye[0].item(), eye[1].item(), target[2].item() + args_cli.up],
            [target[0].item(), target[1].item(), target[2].item() + 0.1],
        )

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    frames, resets = [], 0
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            aim()
            stepped = env.step(policy(obs))
            obs, dones = stepped[0], stepped[2]
            resets += int(dones[args_cli.env_id].item())
            frame = viz.render_rgb_array()
            if frame is not None:
                frames.append(frame)

    if not frames:
        raise RuntimeError("the visualizer returned no frames")
    imageio.mimwrite(args_cli.out, frames, fps=50, quality=8, macro_block_size=1)
    print(f"[record] wrote {args_cli.out}: {len(frames)} frames, robot reset {resets} times")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
