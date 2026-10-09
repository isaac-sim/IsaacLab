# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run GR00T and Isaac Lab in separate uv environments on one GPU."""

from __future__ import annotations

import argparse
import os
import secrets
import subprocess
from pathlib import Path


def main() -> None:
    """Launch the simulation server and model client, preserving their exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr00t_project", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--backbone_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("logs/gr00t/local"))
    parser.add_argument("--rollout_steps", type=int, default=8)
    parser.add_argument("--denoising_steps", type=int, default=2)
    parser.add_argument("--updates", type=int, default=1, help="Zero runs inference only.")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.rollout_steps < 2 or args.denoising_steps < 1 or args.updates < 0:
        parser.error("Require rollout_steps >= 2, denoising_steps >= 1, and updates >= 0.")
    if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").lower() not in {"yes", "y", "1"}:
        parser.error("Accept the NVIDIA Omniverse EULA and set OMNI_KIT_ACCEPT_EULA=YES before launching.")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    socket_path = Path("/tmp") / f"gr00t-{os.getpid()}.sock"
    authkey = secrets.token_hex(32)
    env = {**os.environ, "GR00T_LOCAL_AUTHKEY": authkey, "NO_ALBUMENTATIONS_UPDATE": "1"}
    env.pop("VIRTUAL_ENV", None)
    common = ["--socket_path", str(socket_path), "--output_dir", str(args.output_dir)]
    sim_command = ["uv", "run", "--no-sync", "python", str(script_dir / "sim_server.py"), *common]
    model_command = [
        "uv",
        "run",
        "--project",
        str(args.gr00t_project.resolve()),
        "--no-sync",
        "python",
        str(script_dir / "model_client.py"),
        *common,
        "--model_path",
        str(args.model_path.resolve()),
        "--backbone_path",
        str(args.backbone_path.absolute()),
        "--rollout_steps",
        str(args.rollout_steps),
        "--updates",
        str(args.updates),
        "--denoising_steps",
        str(args.denoising_steps),
    ]
    if args.checkpoint:
        model_command.extend(["--checkpoint", str(args.checkpoint.resolve())])
    with (args.output_dir / "sim.log").open("w") as sim_log:
        sim_process = subprocess.Popen(sim_command, env=env, stdout=sim_log, stderr=subprocess.STDOUT)
        try:
            result = subprocess.run(model_command, env=env, check=False)
            if result.returncode:
                raise SystemExit(result.returncode)
            returncode = sim_process.wait(timeout=60)
            if returncode:
                raise SystemExit(returncode)
        finally:
            if sim_process.poll() is None:
                sim_process.terminate()
                try:
                    sim_process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    sim_process.kill()
                    sim_process.wait()
            socket_path.unlink(missing_ok=True)
    print(f"Completed. Simulation log and results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
