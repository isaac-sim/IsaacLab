# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture the last seconds of simulation before the rough-terrain NaN abort.

``VideoRecorder`` buffers a whole clip in memory and only encodes it when the clip ends, so
a run that dies on the rsl_rl NaN guard normally writes nothing. This wrapper makes two
changes and touches nothing else:

* The recorder's frame buffer becomes a ring buffer of ``KEEP_FRAMES`` frames, so an
  open-ended clip can run until the crash without exhausting memory.
* The NaN abort is caught, the ring buffer is encoded, and the offending environment
  indices are reported.

Environment variables:
    KEEP_FRAMES: number of trailing frames to keep (default 900, ~18 s at 50 fps).
    VIZ_WIDTH / VIZ_HEIGHT: Newton GL window size [px]. The 1920x1080 default is far too
        slow here — this machine has no CUDA/OpenGL interop, so every frame is read back
        through host memory and the cost scales with pixel count.

Every argument is forwarded verbatim to the training CLI. Pass ``--video --viz newton``
and a ``--video_length`` larger than the run so the clip never self-terminates.
"""

import os
import sys

TAG = "NANVIDEO:"
KEEP_FRAMES = int(os.environ.get("KEEP_FRAMES", "900"))
VIZ_WIDTH = int(os.environ.get("VIZ_WIDTH", "640"))
VIZ_HEIGHT = int(os.environ.get("VIZ_HEIGHT", "360"))

_STATE: dict = {}


class _RingList(list):
    """List that discards its oldest item once it grows past ``maxlen``."""

    def __init__(self, maxlen: int):
        super().__init__()
        self.maxlen = maxlen

    def append(self, item):
        super().append(item)
        if len(self) > self.maxlen:
            del self[0]


def _install_ring_buffer() -> None:
    """Swap each recorder's frame list for a ring buffer as soon as the env is built."""
    import isaaclab.envs.manager_based_env as manager_based_env

    original = manager_based_env.ManagerBasedEnv.__init__

    def patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _STATE["env"] = self
        recorders = getattr(self, "video_recorders", [])
        for recorder in recorders:
            recorder._frames = _RingList(KEEP_FRAMES)
        print(f"{TAG} ring buffer of {KEEP_FRAMES} frames on {len(recorders)} recorder(s)", flush=True)

    manager_based_env.ManagerBasedEnv.__init__ = patched


def _report_nan_envs() -> None:
    """Print which environments hold a non-finite root state."""
    import torch

    env = _STATE.get("env")
    if env is None:
        print(f"{TAG} no env captured, cannot report NaN indices", flush=True)
        return
    try:
        robot = env.scene["robot"]
        root = robot.data.root_state_w
        bad = (~torch.isfinite(root)).any(dim=-1).nonzero(as_tuple=False).flatten()
        print(f"{TAG} non-finite root state in {bad.numel()}/{root.shape[0]} envs: {bad.tolist()[:32]}", flush=True)
        origins = env.scene.env_origins
        for idx in bad.tolist()[:8]:
            print(f"{TAG}   env {idx} origin={origins[idx].tolist()}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"{TAG} could not inspect robot state: {exc}", flush=True)


def _flush_video() -> None:
    """Encode whatever the ring buffer is holding."""
    env = _STATE.get("env")
    for recorder in getattr(env, "video_recorders", []) or []:
        n = len(getattr(recorder, "_frames", []))
        print(f"{TAG} flushing {n} buffered frames", flush=True)
        try:
            recorder.close()
        except Exception as exc:  # noqa: BLE001
            print(f"{TAG} recorder.close() failed: {exc}", flush=True)


def _shrink_visualizer_window() -> None:
    """Force the Newton GL window down to a size this machine can read back per step."""
    from isaaclab_visualizers.newton.newton_visualizer import NewtonGLVisualizer

    original = NewtonGLVisualizer.__init__

    def patched(self, cfg, *args, **kwargs):
        cfg.window_width = VIZ_WIDTH
        cfg.window_height = VIZ_HEIGHT
        print(f"{TAG} newton_gl window forced to {VIZ_WIDTH}x{VIZ_HEIGHT}", flush=True)
        original(self, cfg, *args, **kwargs)

    NewtonGLVisualizer.__init__ = patched


_shrink_visualizer_window()
_install_ring_buffer()

from isaaclab_rl.entrypoints import run_train_cli  # noqa: E402

try:
    code = run_train_cli(sys.argv[1:])
except BaseException as exc:  # noqa: BLE001
    print(f"{TAG} run aborted: {type(exc).__name__}: {exc}", flush=True)
    _report_nan_envs()
    _flush_video()
    raise SystemExit(1)

_flush_video()
raise SystemExit(code)
