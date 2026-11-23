# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure Cosmos3 window-mode throughput and GPU memory against the number of environments in one batch.

Run it in the Cosmos Framework environment, where the world-model service runs, on a GPU without a service:

.. code-block:: bash

    uv run --no-sync python "$ISAAC_LAB_ROOT/scripts/benchmarks/benchmark_cosmos3_world_model.py" \\
        --checkpoint "$COSMOS3_CHECKPOINT"

Every environment starts from the same image with its own seed. The first chunk of each environment in a batch is
compared with the one it gets when generated alone, then the following chunks are timed.
"""

import argparse
import time

import numpy as np
import torch
from isaaclab_cosmos_server.world_model import NUM_CONTROLS, Cosmos3Settings, Cosmos3WorldModel


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="Local HF export, or DCP root with --config-file.")
    parser.add_argument("--config-file", help="Saved YAML configuration of a DCP checkpoint.")
    parser.add_argument("--output-dir", default="outputs/cosmos3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="Batch sizes to measure.")
    parser.add_argument("--chunks", type=int, default=2, help="Timed chunks per batch size, after the first.")
    args = parser.parse_args()

    settings = Cosmos3Settings(
        checkpoint_path=args.checkpoint, config_file=args.config_file, output_dir=args.output_dir
    )
    model = Cosmos3WorldModel(settings, args.device)
    device = torch.device(args.device)
    print(f"weights: {torch.cuda.memory_allocated(device) / 2**30:.1f} GiB, {settings.chunk_size} frames per chunk")
    height, width = settings.image_size
    # A smooth gradient, so the generated frames are not trivially uniform.
    image = np.broadcast_to(np.linspace(0, 255, width, dtype=np.uint8)[None, :, None], (height, width, 3))

    def first_chunk(session, num_envs, seeds):
        images = np.ascontiguousarray(np.broadcast_to(image, (num_envs, height, width, 3)))
        session.reset(list(range(num_envs)), images, ["A robot arm moves."] * num_envs, seeds)
        return session.step(np.zeros((num_envs, settings.chunk_size, NUM_CONTROLS), np.float32))

    # Each environment generated alone, with the seed it has in the batches.
    alone = []
    for env in range(max(args.num_envs)):
        session = model.open_session(1)
        alone.append(first_chunk(session, 1, [env])[0])
        session.close()

    print("envs  env-chunks/s  frames/s  s/chunk  peak GiB  max |diff| to alone  mean |diff|")
    for num_envs in sorted(args.num_envs):
        session = model.open_session(num_envs)
        torch.cuda.reset_peak_memory_stats(device)
        try:
            frames = first_chunk(session, num_envs, list(range(num_envs)))
            difference = (frames.int() - torch.stack(alone[:num_envs]).int()).abs()
            controls = np.zeros((num_envs, settings.chunk_size, NUM_CONTROLS), np.float32)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(args.chunks):
                session.step(controls)
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - start) / args.chunks
        except torch.cuda.OutOfMemoryError:
            print(f"{num_envs:4d}  out of memory")
            break
        finally:
            session.close()
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        rate = num_envs / elapsed
        print(
            f"{num_envs:4d}  {rate:12.2f}  {rate * settings.chunk_size:8.1f}  {elapsed:7.2f}  {peak:8.1f} "
            f" {difference.max().item():19d}  {difference.float().mean().item():11.3f}"
        )
        del frames, difference
        torch.cuda.empty_cache()
    model.close()


if __name__ == "__main__":
    main()
