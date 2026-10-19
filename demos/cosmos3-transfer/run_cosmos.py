# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Download the pinned Cosmos Hugging Face weights and start the Cosmos service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(args: list[str] | None = None) -> None:
    """Resolve the checked-in model pin; forward remaining options to the service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir", type=Path, help="Use an already downloaded export after checking its identity."
    )
    parser.add_argument(
        "--download-only", action="store_true", help="Download and verify weights without starting a GPU service."
    )
    options, service_args = parser.parse_known_args(args)
    if any(arg == "--checkpoint" or arg.startswith("--checkpoint=") for arg in service_args):
        parser.error("Use --checkpoint-dir for a validated local export; the default downloads the Hugging Face pin.")
    pin = json.loads((Path(__file__).resolve().parent / "checkpoints.json").read_text())["cosmos"]
    if options.checkpoint_dir is None:
        from huggingface_hub import snapshot_download

        checkpoint = Path(snapshot_download(repo_id=pin["repo"], revision=pin["revision"]))
    else:
        checkpoint = options.checkpoint_dir.resolve()
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    exported = metadata["cosmos3_nano_sim_bimanual"]
    for key in ("checkpoint_id", "checkpoint_iteration", "checkpoint_hash"):
        if exported[key] != pin[key]:
            raise ValueError(f"Wrong Cosmos export: {key}={exported[key]!r}; expected {pin[key]!r}")
    for key in ("source_weights", "student_only"):
        if metadata[key] != pin[key]:
            raise ValueError(f"Wrong Cosmos export: {key}={metadata[key]!r}; expected {pin[key]!r}")
    print(json.dumps({"repo": pin["repo"], "revision": pin["revision"], "checkpoint": str(checkpoint)}), flush=True)
    if options.download_only:
        return
    from isaaclab_experimental.cosmos.server.worker import main as serve

    # Explicit user options appear last and can override recipe settings.
    serve(
        [
            "--checkpoint",
            str(checkpoint),
            "--kv-window",
            "30",
            "--attention-sink",
            "3",
            "--fps",
            "30",
            "--emphasize-control-in-prompt",
            "--deterministic-cudnn",
            *service_args,
        ]
    )


if __name__ == "__main__":
    main()
