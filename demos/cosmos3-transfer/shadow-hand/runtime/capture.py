# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the Shadow Hand camera task with Cosmos on its camera modifier and record one aligned set of frames.

One run records, for every camera capture: the plain camera image, the exact depth control that went to Cosmos and, as
soon as the service returns it, the generated frame. All three come from the same capture, so they can be played side by
side, and every generated frame is kept (not the latest one repeated between batches).

The recording subclasses the integration's own Cosmos camera modifier and wraps its stream; it changes no source file.
Random actions do not read images, so the task's feature extractor is switched off here.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
import traceback
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
from isaaclab_experimental.cosmos import CosmosTransferModifierCfg
from isaaclab_experimental.cosmos.client.cosmos_modifier import CosmosTransferModifier

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import validate

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

TASK = "Isaac-Reorient-Cube-Shadow-Camera-Direct"
SIZE = 640
SEED = 42

RECORDER: Recorder | None = None
"""The recorder of this run. Isaac Lab instantiates the modifier class, so the modifier reaches the recorder here."""


class Recorder:
    """Appends aligned frames to ``paired-frames.h5``."""

    NAMES = ("plain", "cosmos", "depth_control")

    def __init__(self, output: Path):
        self.step = 0
        self.active = False
        self.rows = 0
        self.events: list[dict] = []
        self.resets: list[dict] = []
        self.file = h5py.File(output / "paired-frames.h5", "w")
        for name in self.NAMES:
            self.file.create_dataset(
                name,
                shape=(0, SIZE, SIZE, 3),
                maxshape=(None, SIZE, SIZE, 3),
                dtype="u1",
                chunks=(1, SIZE, SIZE, 3),
                compression="lzf",
            )
        self.file.create_dataset("step", shape=(0,), maxshape=(None,), dtype="i8")
        self.file.create_dataset("time_s", shape=(0,), maxshape=(None,), dtype="f8")

    def append(self, pairs: list[dict], generated: np.ndarray, seconds: float) -> None:
        """Store one returned Cosmos chunk next to the captures that produced it."""
        event = {
            "chunk": len(self.events),
            "frames": len(pairs),
            "seconds": seconds,
            "steps": [p["step"] for p in pairs],
        }
        self.events.append(event)
        for pair, frame in zip(pairs, generated, strict=True):
            if not pair["active"]:
                continue
            index = self.rows
            self.rows += 1
            values = (
                ("plain", pair["rgb"]),
                ("cosmos", frame),
                ("depth_control", pair["control"]),
                ("step", pair["step"]),
                ("time_s", pair["time_s"]),
            )
            for key, value in values:
                dataset = self.file[key]
                dataset.resize(self.rows, axis=0)
                dataset[index] = value
        self.file.flush()
        print("Captured chunk " + json.dumps(event), flush=True)

    def close(self) -> None:
        self.file.close()


class StreamObserver:
    """Wraps the modifier's Cosmos stream to pair each returned chunk with the controls that were sent."""

    def __init__(self, stream, modifier: RecordedCosmosModifier):
        self.stream = stream
        self.modifier = modifier

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def step(self, controls, reset_rows, seeds):
        assert len(controls) == 1
        count = len(controls[0])
        pairs = self.modifier.observed[:count]
        assert len(pairs) == count
        sent = controls[0].detach().cpu().numpy()
        assert np.array_equal(np.stack([p["control"] for p in pairs]), sent), (
            "Observed depth must equal the sent controls"
        )
        began = time.monotonic()
        generated = self.stream.step(controls, reset_rows, seeds)
        seconds = time.monotonic() - began
        RECORDER.append(pairs, generated[0].detach().cpu().numpy(), seconds)
        del self.modifier.observed[:count]
        return generated

    def close(self):
        return self.stream.close()


class RecordedCosmosModifier(CosmosTransferModifier):
    """The integration's Cosmos camera modifier, additionally remembering each capture's plain image and control."""

    instances: list[RecordedCosmosModifier] = []

    def __init__(self, cfg, data_dim, device):
        super().__init__(cfg, data_dim, device)
        self.observed: list[dict] = []
        self._stream = StreamObserver(self._stream, self)
        RecordedCosmosModifier.instances.append(self)

    def __call__(self, data):
        raw = self._sensor._data.output["rgba"].torch[0, ..., :3].detach().cpu().numpy().copy()
        self.observed.append(
            {
                "rgb": raw,
                "control": data[0].detach().cpu().numpy().copy(),
                "step": RECORDER.step,
                "time_s": self._sensor_time(),
                "active": RECORDER.active,
            }
        )
        return super().__call__(data)

    def reset(self, env_ids=None):
        if self.observed:
            RECORDER.resets.append({"step": RECORDER.step, "discarded_pending": len(self.observed)})
        self.observed.clear()
        return super().reset(env_ids)


def main() -> None:
    global RECORDER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True, help="Cosmos service endpoint, unix:///path or tcp://host:port")
    parser.add_argument("--renderer", default="newton_renderer", help="The task's renderer preset")
    parser.add_argument("--steps", type=int, default=120, help="Environment steps to record")
    add_launcher_args(parser)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    cfg, _ = resolve_task_config(
        TASK, "", overrides=[f"presets=cosmos,{args.renderer}", f"env.seed={SEED}", "env.scene.num_envs=1"]
    )
    camera = cfg.scene.tiled_camera
    camera.width = SIZE
    camera.height = SIZE
    camera.data_types = ["rgb", "rgba", "distance_to_image_plane"]  # rgba: the plain image beside the generated one
    cfg.feature_extractor.enabled = False
    cfg.feature_extractor.train = False
    cfg.feature_extractor.load_checkpoint = False
    for chain in camera.modifiers.values():
        for modifier in chain:
            if isinstance(modifier, CosmosTransferModifierCfg):
                modifier.func = RecordedCosmosModifier
                modifier.backend.endpoint = args.endpoint
    validate(cfg)
    config = {
        "task": TASK,
        "renderer": type(camera.renderer_cfg).__name__,
        "renderer_preset": args.renderer,
        "seed": SEED,
        "steps": args.steps,
        "size": [SIZE, SIZE],
        "endpoint": args.endpoint,
    }
    print(json.dumps(config), flush=True)
    (out / "capture-config.json").write_text(json.dumps(config, indent=2))

    torch.manual_seed(SEED)
    RECORDER = Recorder(out)
    started = time.monotonic()
    result: dict = {"completed": False}
    try:
        with launch_simulation(cfg, args), contextlib.ExitStack() as cleanup:
            env = gym.make(TASK, cfg=cfg)
            cleanup.callback(env.close)
            RECORDER.active = True
            env.reset()
            env.unwrapped.scene["tiled_camera"].data
            generator = torch.Generator(device=env.unwrapped.device)
            generator.manual_seed(SEED)
            for step in range(1, args.steps + 1):
                RECORDER.step = step
                with torch.inference_mode():
                    action = (
                        2 * torch.rand(env.action_space.shape, device=env.unwrapped.device, generator=generator) - 1
                    )
                    env.step(action)
                    env.unwrapped.scene["tiled_camera"].data
            result["capture_gap"] = [m.capture_gap_s for m in RecordedCosmosModifier.instances]
            result["completed"] = True
    except BaseException as error:
        result["error"] = repr(error)
        traceback.print_exc()
        raise
    finally:
        result.update(
            wall_s=time.monotonic() - started,
            paired_frames=RECORDER.rows,
            events=RECORDER.events,
            resets=RECORDER.resets,
            controls_checked="every control equals the exact array queued for Cosmos (asserted per chunk)",
            renderer=config["renderer"],
        )
        RECORDER.close()
        (out / "capture-result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "events"}), flush=True)


if __name__ == "__main__":
    main()
