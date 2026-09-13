# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Film the final moments of the environment that blows up on rough terrain.

The blow-up is impossible to film the obvious way: it hits exactly one environment out of
512, a different one each run, and the NaN only reproduces at 512 environments — a count
this machine cannot render continuously, because Warp gets no CUDA/OpenGL interop here and
every frame is copied through host memory.

``nanwatch.py`` showed the failure is preceded by a single-step contact-force spike, from
under 200 N to over 20 kN, six steps before the NaN. The spike arms the capture, but it
does not identify the victim: spikes of 60-75 kN are routine and survivable, and the first
attempt filmed a 48 kN environment while a different one died. Joint speed is the
discriminating signal — it runs 5-16 rad/s in normal walking and then climbs without
bound (71, 216, 1348) over the final steps — so the camera re-picks the fastest-jointed
environment on every filmed step and converges onto the one that is actually diverging.

* ``sim.render()`` is a no-op, so the run costs what a headless run costs.
* When any environment's contact force crosses ``TRIGGER_N``, the camera snaps to that
  environment and rendering turns on for ``ARM_STEPS`` steps.
* Frames are pulled straight off the visualizer with ``render_rgb_array()`` into a ring
  buffer and encoded here. ``VideoRecorder`` is deliberately bypassed: its clip state
  machine silently captured nothing once rendering was gated.

Environment variables:
    TRIGGER_N: contact force magnitude that arms the capture [N] (default 5000).
    ARM_STEPS: env steps to keep filming after arming (default 12).
    CLIP_FPS: playback frame rate of the written clip (default 5, i.e. 10x slow motion).
    VIZ_WIDTH / VIZ_HEIGHT: Newton GL window size [px] (default 1280x720).
    OUT_MP4: output clip path (default ``nanvid/nanshot.mp4``).

Every argument is forwarded verbatim to the training CLI. Pass ``--viz newton``; ``--video``
is not needed and not used.
"""

import os
import sys
from collections import deque

TAG = "NANSHOT:"
TRIGGER_N = float(os.environ.get("TRIGGER_N", "5000"))
ARM_STEPS = int(os.environ.get("ARM_STEPS", "12"))
CLIP_FPS = int(os.environ.get("CLIP_FPS", "5"))
VIZ_WIDTH = int(os.environ.get("VIZ_WIDTH", "1280"))
VIZ_HEIGHT = int(os.environ.get("VIZ_HEIGHT", "720"))
OUT_MP4 = os.environ.get("OUT_MP4", "nanvid/nanshot.mp4")

_S: dict = {"armed": False, "armed_for": 0, "target": None, "env": None, "arms": 0}
_TRACE: deque = deque(maxlen=40)
_FRAMES: deque = deque(maxlen=ARM_STEPS + 4)


def _patch_visualizer_window() -> None:
    from isaaclab_visualizers.newton.newton_visualizer import NewtonGLVisualizer

    original = NewtonGLVisualizer.__init__

    def patched(self, cfg, *args, **kwargs):
        cfg.window_width = VIZ_WIDTH
        cfg.window_height = VIZ_HEIGHT
        original(self, cfg, *args, **kwargs)

    NewtonGLVisualizer.__init__ = patched


def _patch_render_gate() -> None:
    """Make rendering cost nothing until the capture is armed."""
    from isaaclab.sim.simulation_context import SimulationContext

    original_render = SimulationContext.render

    def gated_render(self, *args, **kwargs):
        if not _S["armed"]:
            return None
        return original_render(self, *args, **kwargs)

    SimulationContext.render = gated_render


def _patch_env() -> None:
    """Install the ring buffer at env creation and the trigger check after every step."""
    import isaaclab.envs.manager_based_env as manager_based_env
    import isaaclab.envs.manager_based_rl_env as manager_based_rl_env

    original_init = manager_based_env.ManagerBasedEnv.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _S["env"] = self
        print(f"{TAG} armed capture ready: trigger={TRIGGER_N} N, window={ARM_STEPS} steps", flush=True)

    manager_based_env.ManagerBasedEnv.__init__ = patched_init

    original_step = manager_based_rl_env.ManagerBasedRLEnv.step

    def patched_step(self, action):
        out = original_step(self, action)
        _S["env"] = self
        try:
            _check_trigger(self)
        except Exception as exc:  # noqa: BLE001
            if not _S.get("trigger_error"):
                _S["trigger_error"] = True
                print(f"{TAG} trigger check failed: {type(exc).__name__}: {exc}", flush=True)
        return out

    manager_based_rl_env.ManagerBasedRLEnv.step = patched_step


def _contact_per_env(env):
    """Peak contact force magnitude per environment [N]."""
    import torch

    forces = env.scene["contact_forces"].data.net_forces_w
    return torch.linalg.norm(forces, dim=-1).amax(dim=-1)


def _visualizer(env):
    """The first visualizer that can hand back a rendered frame."""
    for viz in getattr(env.sim, "visualizers", []):
        if hasattr(viz, "render_rgb_array"):
            return viz
    return None


def _aim_camera(env, idx: int) -> None:
    """Point the Newton GL camera at one environment's robot."""
    root = env.scene["robot"].data.root_state_w[idx, :3].tolist()
    if not all(abs(v) < 1e6 for v in root):  # already diverged; keep the last good pose
        return
    eye = (root[0] + 2.0, root[1] + 2.0, root[2] + 1.2)
    target = (root[0], root[1], root[2])
    viz = _visualizer(env)
    if viz is not None and hasattr(viz, "set_camera_view"):
        viz.set_camera_view(eye, target)


def _grab_frame(env) -> None:
    """Re-render with the camera pose just set, and buffer the result.

    The step's own render already happened before the target was re-picked, so grabbing
    without re-rendering would lag the camera by one frame — long enough to miss the
    hand-off to the environment that actually diverges.
    """
    viz = _visualizer(env)
    if viz is None:
        return
    env.sim.render()
    frame = viz.render_rgb_array()
    if frame is None:
        if not _S.get("frame_warned"):
            _S["frame_warned"] = True
            print(f"{TAG} render_rgb_array() returned None", flush=True)
        return
    _FRAMES.append(frame)


def _worst_joint_speed(env):
    """Peak joint speed per environment [rad/s], with non-finite values neutralised."""
    import torch

    speed = env.scene["robot"].data.joint_vel.abs().amax(dim=-1)
    return torch.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)


def _check_trigger(env) -> None:
    """Arm on a contact spike, then track the fastest-diverging environment while armed."""
    import torch

    contact = torch.nan_to_num(_contact_per_env(env), nan=0.0, posinf=0.0, neginf=0.0)
    peak, idx = torch.max(contact, dim=0)
    peak_v, idx_v = float(peak), int(idx)
    _TRACE.append((peak_v, idx_v))

    if _S["armed"]:
        _S["armed_for"] += 1
        # Re-pick the victim every step: the diverging environment is the one whose joints
        # are running away, which is not in general the one that took the largest impact.
        speed = _worst_joint_speed(env)
        top, top_idx = torch.max(speed, dim=0)
        if int(top_idx) != _S["target"]:
            _S["target"] = int(top_idx)
            print(f"{TAG}   retargeting to env {int(top_idx)} (joint speed {float(top):.0f} rad/s)", flush=True)
        _aim_camera(env, _S["target"])
        _grab_frame(env)
        if _S["armed_for"] >= ARM_STEPS:
            _S["armed"] = False
            _S["armed_for"] = 0
            print(
                f"{TAG} arm expired without a NaN (env {_S['target']}, {len(_FRAMES)} frames buffered),"
                " waiting for the next spike",
                flush=True,
            )
        return

    if peak_v >= TRIGGER_N:
        _S["armed"] = True
        _S["armed_for"] = 0
        _S["target"] = idx_v
        _S["arms"] += 1
        _FRAMES.clear()
        _aim_camera(env, idx_v)
        print(f"{TAG} armed #{_S['arms']}: env {idx_v} hit {peak_v:.0f} N, filming", flush=True)


def _flush() -> None:
    import torch

    env = _S.get("env")
    if env is None:
        return
    try:
        root = env.scene["robot"].data.root_state_w
        bad = (~torch.isfinite(root)).any(dim=-1).nonzero(as_tuple=False).flatten().tolist()
        print(f"{TAG} non-finite envs: {bad[:16]}  (filmed env {_S['target']})", flush=True)
    except Exception:  # noqa: BLE001
        pass
    print(f"{TAG} peak contact over the last {len(_TRACE)} steps: {[f'{p:.0f}@{i}' for p, i in _TRACE]}", flush=True)
    if not _FRAMES:
        print(f"{TAG} no frames buffered, nothing to write", flush=True)
        return
    try:
        from moviepy.editor import ImageSequenceClip

        os.makedirs(os.path.dirname(OUT_MP4) or ".", exist_ok=True)
        ImageSequenceClip(list(_FRAMES), fps=CLIP_FPS).write_videofile(
            OUT_MP4, codec="libx264", audio=False, logger=None
        )
        print(f"{TAG} wrote {len(_FRAMES)} frames to {OUT_MP4}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"{TAG} write failed: {type(exc).__name__}: {exc}", flush=True)


_patch_visualizer_window()
_patch_render_gate()
_patch_env()

from isaaclab_rl.entrypoints import run_train_cli  # noqa: E402

try:
    code = run_train_cli(sys.argv[1:])
except BaseException as exc:  # noqa: BLE001
    print(f"{TAG} aborted: {type(exc).__name__}: {exc}", flush=True)
    _flush()
    raise SystemExit(1)

_flush()
raise SystemExit(code)
