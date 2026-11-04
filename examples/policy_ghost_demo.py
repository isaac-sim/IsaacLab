# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo of Policy Ghosts: one G1 checkpoint played by the reference and by ghosts with heavier feet.

The reference slot and every ghost slot load the same checkpoint. The ghosts start empty and are added from the
HUD's Add Checkpoint... list, or at start with ``--preload``. Each ghost scales the mass of both ankle links of its
robots, so the policy meets a different body and its gait drifts away from the reference. This is a stand-in for
comparing different checkpoints and is not a benchmark of the checkpoint itself.

With ``--record`` the Newton window is screen-recorded from the start, with the HUD and the mouse pointer, to
``recording/screen/`` until the window closes. The capture is a fixed region of the X11 display, so keep the window
where it opened and in front.

.. code-block:: bash

    uv run python examples/policy_ghost_demo.py --checkpoint <path to a G1 rough-terrain model>
    uv run python examples/policy_ghost_demo.py --checkpoint <path> --feet_scales 2 4 --visible_envs 4
"""

import argparse
import atexit
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import warp as wp

wp.config.enable_backward = False

from isaaclab.visualizers.policy_ghost import PolicyGhostCheckpoint  # noqa: E402

from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402
from isaaclab_rl.entrypoints.backends import play_rsl_rl  # noqa: E402

DEFAULT_TASK = "Isaac-Velocity-Rough-G1"
"""Task whose pretrained checkpoint the demo plays."""

FEET_BODIES = ".*_ankle_roll_link"
"""Bodies whose mass the ghosts scale."""


def _parse() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint every slot plays.")
    parser.add_argument(
        "--feet_scales",
        type=float,
        nargs="+",
        default=[2.0, 4.0],
        help="Ankle link mass multiplier of each ghost slot; one ghost per value.",
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task to play.")
    parser.add_argument("--num_envs", type=int, default=4, help="Environments per slot, so 3 slots simulate 12.")
    parser.add_argument("--visible_envs", type=int, default=1, help="Environments per slot shown at start.")
    parser.add_argument(
        "--record", action="store_true", help="Screen-record the Newton window from the start until it closes."
    )
    parser.add_argument("--preload", action="store_true", help="Load every ghost at start instead of from the HUD.")
    return parser.parse_known_args()


def _scale_slot_feet(env, slice_size: int, slot: int, scale: float) -> None:
    """Set the foot masses of one slot to ``scale`` times those of the reference slot."""
    import torch

    robot = env.unwrapped.scene["robot"]
    body_ids, names = robot.find_bodies(FEET_BODIES)
    reference_envs = torch.arange(slice_size, device=env.unwrapped.device)
    base = robot.data.body_mass.torch[reference_envs][:, body_ids].clone()
    robot.set_masses_index(masses=base * scale, body_ids=body_ids, env_ids=reference_envs + slot * slice_size)
    print(f"Slot {slot}: {names} masses {base[0].tolist()} kg x {scale:g}")


def _make_variants(checkpoint_path: str, scales: list[float]) -> dict[str, float]:
    """Link the checkpoint once per feet scale, so the HUD lists each variant as its own checkpoint."""
    folder = Path(tempfile.mkdtemp(prefix="policy_ghost_demo_"))
    variants = {}
    for scale in scales:
        link = folder / f"feet_x{scale:g}.pt"
        link.symlink_to(Path(checkpoint_path).resolve())
        variants[str(link)] = scale
    return variants


def _place_first_env_on_stairs(env) -> None:
    """Move reference environment 0 onto a mid-difficulty pyramid-stairs tile of the generated terrain."""
    terrain = env.unwrapped.scene.terrain
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    if generator is None or getattr(terrain, "terrain_origins", None) is None:
        print("The scene has no generated terrain, so environment 0 keeps its spawn tile.")
        return
    names = list(generator.sub_terrains)
    proportions = np.array([cfg.proportion for cfg in generator.sub_terrains.values()], dtype=float)
    cumulative = np.cumsum(proportions / proportions.sum())
    # curriculum terrain gives every column one sub-terrain type, in proportion
    columns = [
        column
        for column in range(generator.num_cols)
        if names[int(np.min(np.where(column / generator.num_cols + 0.001 < cumulative)[0]))] == "pyramid_stairs"
    ]
    if not columns:
        print("No pyramid-stairs column was found, so environment 0 keeps its spawn tile.")
        return
    row, column = generator.num_rows // 2, columns[0]
    terrain.terrain_levels[0], terrain.terrain_types[0] = row, column
    terrain.env_origins[0] = terrain.terrain_origins[row, column]
    print(f"Environment 0 spawns on pyramid stairs (terrain row {row}, column {column}).")


class ScreenRecording:
    """Records a region of the X11 display, including the mouse pointer, with ``ffmpeg``."""

    def __init__(self, window_title: str = "Newton Visualizer"):
        self.window_title = window_title
        self.path: Path | None = None
        self._process: subprocess.Popen | None = None

    def _window_region(self) -> tuple[int, int, int, int] | None:
        """Return ``(x, y, width, height)`` of the window titled exactly :attr:`window_title`, without its frame."""
        tree = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True).stdout
        for line in tree.splitlines():
            match = re.match(r'\s*(0x[0-9a-f]+) "([^"]*)": \("([^"]*)" "([^"]*)"\)', line)
            # the window manager's frame of the window has the same title but another class
            if match is None or match.group(2) != self.window_title or match.group(3) != "newton":
                continue
            info = subprocess.run(["xwininfo", "-id", match.group(1)], capture_output=True, text=True).stdout
            values = {
                key: int(re.search(rf"{key}:\s+(-?\d+)", info).group(1))
                for key in ("Absolute upper-left X", "Absolute upper-left Y", "Width", "Height")
            }
            # the encoder needs even dimensions
            return (
                values["Absolute upper-left X"],
                values["Absolute upper-left Y"],
                values["Width"] // 2 * 2,
                values["Height"] // 2 * 2,
            )
        return None

    def start(self, output_dir: str = "recording/screen") -> None:
        region = self._window_region()
        if region is None:
            print(f"Screen recording skipped: no window titled {self.window_title!r} was found.")
            return
        x, y, width, height = region
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"newton_screen_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
        self._process = subprocess.Popen(  # noqa: S603
            [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-draw_mouse", "1", "-framerate", "30",
                "-video_size", f"{width}x{height}", "-i", f"{os.environ['DISPLAY']}+{x},{y}",
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(self.path),
            ],
            stdin=subprocess.PIPE,
        )  # fmt: skip
        atexit.register(self.stop)
        print(f"Screen recording {width}x{height} at +{x}+{y} to {self.path}")

    def stop(self) -> None:
        """Finish the file. ffmpeg closes it properly when it reads ``q``."""
        process, self._process = self._process, None
        if process is None:
            return
        try:
            process.stdin.write(b"q")
            process.stdin.flush()
            process.stdin.close()
            process.wait(timeout=30)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            process.terminate()
        print(f"Screen recording saved to {self.path}")


def main() -> None:
    args, passthrough = _parse()
    scales = args.feet_scales

    def run_with_heavy_feet(env, args_cli, agent_cfg, reference_path, log_root_path, slice_size, restart):
        """Offer the reference checkpoint with heavier feet as ghosts, loaded now or from the HUD."""
        _place_first_env_on_stairs(env)  # before the ghosts copy the reference's terrain
        session = play_rsl_rl.create_policy_ghost_session(
            env, args_cli, agent_cfg, reference_path, log_root_path, slice_size, args_cli.policy_ghost_slots
        )
        variants = _make_variants(reference_path, scales)
        offered = [
            PolicyGhostCheckpoint(path, "G1 demo", f"feet x{scale:g}", Path(reference_path).stat().st_mtime)
            for path, scale in variants.items()
        ]
        add_ghost, remove_ghost, replace_ghost = session.add_ghost, session.remove_ghost, session.replace_ghost

        def add_variant(checkpoint):
            slot = next((s for s in session.layout.slots[1:] if not s.active), None)
            error = add_ghost(checkpoint)
            if error is None and slot is not None:
                _scale_slot_feet(env, slice_size, slot.index, variants.get(checkpoint.path, 1.0))
            return error

        def remove_variant(slot_index):
            remove_ghost(slot_index)
            _scale_slot_feet(env, slice_size, slot_index, 1.0)

        def replace_variant(slot_index, checkpoint):
            error = replace_ghost(slot_index, checkpoint)
            if error is None:
                _scale_slot_feet(env, slice_size, slot_index, variants.get(checkpoint.path, 1.0))
            return error

        session.replace_ghost = replace_variant
        session.discover = lambda: list(offered)
        session.add_ghost = add_variant
        session.remove_ghost = remove_variant
        if args.preload:
            for checkpoint in offered:
                if error := session.add_ghost(checkpoint):
                    raise RuntimeError(error)
        attached = False
        for visualizer in env.unwrapped.sim.visualizers:
            if hasattr(visualizer, "attach_policy_ghosts"):
                visualizer.attach_policy_ghosts(session, restart.request)
                attached = True
        if not attached:
            raise RuntimeError("The demo needs the Newton GL visualizer.")
        for visualizer in env.unwrapped.sim.visualizers:
            if hasattr(visualizer, "set_visible_env_count"):
                visualizer.set_visible_env_count(args.visible_envs)  # the HUD's Visible Envs slider raises it
        screen = ScreenRecording()
        if args.record:
            screen.start()
        visualizers = list(env.unwrapped.sim.visualizers)

        def step() -> None:
            session.step()
            if all(getattr(visualizer, "_is_closed", False) for visualizer in visualizers):
                screen.stop()  # the window was closed, so the recording ends with it
                raise KeyboardInterrupt  # and so does the run, which would otherwise keep stepping without a window

        try:
            play_rsl_rl.run_playback(
                play_rsl_rl._stop_on_restart(step, restart),
                dt=env.unwrapped.step_dt,
                args_cli=args_cli,
                env_cfg=env.unwrapped.cfg,
            )
        finally:
            screen.stop()
            session.close()

    resolve_task_config = play_rsl_rl.resolve_task_config

    def resolve_with_curriculum_terrain(*arguments, **keywords):
        """Generate the terrain by columns of one sub-terrain type, so a stairs tile can be picked."""
        env_cfg, agent_cfg = resolve_task_config(*arguments, **keywords)
        generator = env_cfg.scene.terrain.terrain_generator
        if generator is not None:
            generator.curriculum = True
        env_cfg.scene.num_envs = args.num_envs
        return env_cfg, agent_cfg

    play_rsl_rl.resolve_task_config = resolve_with_curriculum_terrain
    play_rsl_rl.run_policy_ghosts = run_with_heavy_feet
    argv = [
        "--task",
        args.task,
        "--checkpoint",
        args.checkpoint,
        "--num_envs",
        str(args.num_envs),
        "--policy_ghosts",
        "--policy_ghost_slots",
        str(len(scales) + 1),
        "--visualizer",
        "newton_gl",
        "presets=newton_mjwarp",
        *passthrough,
    ]
    sys.exit(run_play_cli(argv))


if __name__ == "__main__":
    main()
