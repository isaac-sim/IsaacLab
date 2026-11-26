# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate pi0.5 DROID joint-position actions on sampled tabletop pick/place tasks.

With Isaac Lab installed, run (Linux with an NVIDIA GPU, git and uv)::

    uv run --with msgpack python scripts/large_models/pi05_evaluate.py \\
        --num_envs 8 --trials 32 --output logs/pi05/tabletop --viz none

The first run fetches a pinned OpenPI checkout, prepares its Python 3.11 environment and
downloads the 12.4 GB checkpoint ``gs://openpi-assets-simeval/pi05_droid_jointpos`` anonymously.
Later runs reuse the caches. No OpenPI installation, Google login or separate server command is
needed. Setup and model loading can take several minutes. OpenPI runs in a child process and is
stopped on exit. By default it uses the first visible GPU without JAX preallocation.
``--policy-device 1`` selects another GPU using CUDA's device identifiers. Model inference and rendering
must both fit in GPU memory.

``--checkpoint /path/to/pi05_droid_jointpos`` reuses an existing checkpoint. ``--policy HOST:PORT``
connects to an existing standard OpenPI joint-position server instead and skips all local setup.
OpenPI source/environment live under ``~/.cache/isaaclab/pi05`` (``--policy-cache``); weights and the
tokenizer use OpenPI's ``~/.cache/openpi`` cache (``OPENPI_DATA_HOME`` overrides it).

``--trials`` counts trials per task family across the whole run. Any idle world takes the next trial,
requests its global family ID and resets to a fresh category composition, object count and table.
The report records the concrete instruction and assets sampled for each trial. ``--tasks`` selects
families; otherwise both registered families are evaluated. Ordinary task overrides can follow the
flags, for example ``env.episode_length_s=10``. The policy must output absolute arm joint positions
and gripper closure in [0, 1]; the standard velocity-action DROID checkpoint is incompatible.

Each active world gets one standard OpenPI request per action chunk (serial inference). A world
that ends mid-chunk discards its remaining actions and holds its reset pose until the next query.
Every success or timeout is counted, including one-step episodes. ``episodes.jsonl`` is flushed per
trial and ``result.json`` is saved periodically and on exit.
"""

import argparse
import json
import os
import signal
import socket as socket_module
import subprocess
import time
from collections import Counter, deque
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="IsaacContrib-TableTop-PickPlace-Franka")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--policy", metavar="HOST:PORT", help="Existing standard OpenPI server; default: start locally.")
parser.add_argument("--checkpoint", default="gs://openpi-assets-simeval/pi05_droid_jointpos", help="Checkpoint.")
parser.add_argument("--policy-cache", type=Path, default=Path.home() / ".cache/isaaclab/pi05")
parser.add_argument("--policy-device", help="Inference GPU(s); default: inherit CUDA_VISIBLE_DEVICES.")
parser.add_argument(
    "--output", type=Path, default=Path("logs/pi05") / datetime.now().strftime("%Y%m%d_%H%M%S"), help="Reports."
)
parser.add_argument("--tasks", nargs="+", metavar="FAMILY", help="Task families; default: all declared families.")
parser.add_argument("--trials", type=int, default=32, help="Trials per family across all environments.")
parser.add_argument("--horizon", type=int, default=15, help="Control steps executed per policy query.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_steps", type=int, default=100000, help="Fail if evaluation exceeds this many control steps.")
add_launcher_args(parser)
args, overrides = parser.parse_known_args()
if min(args.num_envs, args.trials, args.horizon, args.max_steps) < 1:
    parser.error("--num_envs, --trials, --horizon and --max_steps must be positive")
if args.tasks and len(set(args.tasks)) != len(args.tasks):
    parser.error("--tasks must not repeat a family")

import gymnasium as gym  # noqa: E402
import msgpack  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

CAMERAS = ("over_shoulder_left_camera", "wrist_cam")
POLICY_IMAGE_SIZE = 224
OPENPI_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"


@contextmanager
def policy_connection():
    """Own local OpenPI setup and process lifetime, or borrow an external server."""
    if args.policy:
        with connect(f"ws://{args.policy}", compression=None, max_size=None, open_timeout=60) as connection:
            yield connection
        return

    # OpenPI's JAX/Python requirements belong to its own environment, never Isaac Lab's.
    environment = os.environ.copy()
    for name in ("UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "GIT_DIR", "GIT_WORK_TREE"):
        environment.pop(name, None)
    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.policy_device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.policy_device
    checkout = args.policy_cache.expanduser().resolve() / f"openpi-{OPENPI_REVISION}"
    checkout.mkdir(parents=True, exist_ok=True)
    if not (checkout / ".git").exists():
        subprocess.run(["git", "init", str(checkout)], env=environment, check=True)
    git = ["git", "-C", str(checkout)]
    revision = subprocess.run(git + ["rev-parse", "HEAD"], env=environment, capture_output=True, text=True)
    if revision.returncode or revision.stdout.strip() != OPENPI_REVISION:
        print(f"Fetching OpenPI {OPENPI_REVISION} into {checkout}", flush=True)
        subprocess.run(
            git + ["fetch", "--depth=1", "https://github.com/Physical-Intelligence/openpi.git", OPENPI_REVISION],
            env=environment,
            check=True,
        )
        subprocess.run(git + ["checkout", "--detach", OPENPI_REVISION], env=environment, check=True)

    with socket_module.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = ["uv", "run", "--frozen", "--no-dev", "--python", "3.11", "--project", str(checkout)]
    command += ["python", str(Path(__file__).with_name("pi05_serve.py").resolve()), "--port", str(port)]
    command += ["--checkpoint", args.checkpoint]
    print("Preparing OpenPI and loading pi0.5; the first run downloads dependencies and weights.", flush=True)
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    try:
        deadline = time.monotonic() + 3600
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"OpenPI exited with status {process.returncode}; see its output above.")
            try:
                connection = connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None, open_timeout=5)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Local OpenPI setup did not finish within one hour.") from None
                time.sleep(1)
        with connection:
            yield connection
    finally:
        # Include uv's Python child so interrupted setup/inference cannot leave a GPU process behind.
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def resize_with_pad(image: np.ndarray) -> np.ndarray:
    """Match openpi's bilinear letterbox into a 224 by 224 RGB image."""
    picture = Image.fromarray(image)
    ratio = max(picture.width, picture.height) / POLICY_IMAGE_SIZE
    resized = picture.resize((int(picture.width / ratio), int(picture.height / ratio)), Image.Resampling.BILINEAR)
    square = Image.new(picture.mode, (POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE))
    square.paste(resized, ((POLICY_IMAGE_SIZE - resized.width) // 2, (POLICY_IMAGE_SIZE - resized.height) // 2))
    return np.asarray(square)


def pack_array(value):
    """Encode NumPy arrays/scalars in openpi's MessagePack wire format."""
    if isinstance(value, np.ndarray):
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"Cannot encode {type(value).__name__} in the policy request.")


def unpack_array(value: dict):
    """Decode NumPy arrays/scalars from openpi's MessagePack wire format."""
    if b"__ndarray__" in value:
        return np.ndarray(buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]), shape=value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def query_policy(socket, observation: dict, trials: dict[int, dict], device: str) -> torch.Tensor:
    """Query the active worlds in insertion order; return (worlds, horizon, 8) absolute actions."""
    slots = list(trials)
    images = [observation[name][slots].cpu().numpy() for name in CAMERAS]
    arm = observation["arm_joint_pos"][slots].cpu().numpy().astype(np.float32)
    gripper = observation["gripper_pos"][slots].cpu().numpy().astype(np.float32)
    chunks = []
    for row, trial in enumerate(trials.values()):
        request = {
            "observation/exterior_image_1_left": resize_with_pad(images[0][row]),
            "observation/wrist_image_left": resize_with_pad(images[1][row]),
            "observation/joint_position": arm[row],
            "observation/gripper_position": gripper[row],
            "prompt": trial["instruction"],
        }
        trial["queries"] += 1
        socket.send(msgpack.packb(request, default=pack_array))
        response = socket.recv(timeout=600)
        if isinstance(response, str):
            raise RuntimeError(f"Policy server: {response}")
        actions = np.asarray(msgpack.unpackb(response, object_hook=unpack_array)["actions"], dtype=np.float32)
        if (
            actions.ndim != 2
            or actions.shape[0] < args.horizon
            or actions.shape[1] != 8
            or not np.isfinite(actions).all()
        ):
            raise ValueError(f"Expected finite (>= {args.horizon}, 8) policy actions, got {actions.shape}.")
        chunks.append(actions[: args.horizon])
    return torch.tensor(np.stack(chunks), device=device)


def evaluate(env, socket, report: dict, save) -> None:
    """Distribute a finite family/trial schedule across worlds and account for each terminal result once."""
    families = list(env.cfg.tasks)
    selected = args.tasks or families
    pending = deque((families.index(family), trial) for trial in range(args.trials) for family in selected)
    report["expected_trials"] = len(pending)
    active = {}
    observation, _ = env.reset(seed=args.seed)
    actions = torch.cat((observation["policy"]["arm_joint_pos"], torch.ones(env.num_envs, 1, device=env.device)), 1)
    started = last_save = time.perf_counter()
    with (args.output / "episodes.jsonl").open("w") as episodes:
        while pending or active:
            idle = [slot for slot in range(env.num_envs) if slot not in active]
            starting = idle[: len(pending)]
            if starting:
                jobs = [pending.popleft() for _ in starting]
                env.tabletop.set_task(starting, [family for family, _ in jobs])
                observation, _ = env.reset(env_ids=torch.tensor(starting, device=env.device))
                for slot, (family, trial) in zip(starting, jobs, strict=True):
                    active[slot] = {
                        "family": families[family],
                        "task_id": family,
                        "trial": trial,
                        "env_id": slot,
                        "instruction": env.tabletop.instructions[slot],
                        "objects": list(env.tabletop.object_names[slot]),
                        "object_variants": env.scene["objects"].mesh_variant_ids.torch[slot].tolist(),
                        "table_variant": int(env.scene["work_table"].mesh_variant_ids.torch[slot]),
                        "steps": 0,
                        "queries": 0,
                    }
            slots = list(active)
            before_query = time.perf_counter()
            chunks = query_policy(socket, observation["policy"], active, env.device)
            report["policy_seconds"] += time.perf_counter() - before_query
            report["policy_queries"] += len(active)
            rows = {slot: row for row, slot in enumerate(slots)}
            for offset in range(args.horizon):
                if not active:
                    break
                if report["control_steps"] >= args.max_steps:
                    raise TimeoutError(f"Evaluation exceeded {args.max_steps} control steps.")
                running = list(active)
                commands = chunks[[rows[slot] for slot in running], offset]
                actions[running, :7] = commands[:, :7]
                # DROID: 0=open, 1=closed. The environment closes for a negative binary command.
                actions[running, 7] = torch.where(commands[:, 7] > 0.5, -1.0, 1.0)
                before_step = time.perf_counter()
                observation, _, _, _, _ = env.step(actions)
                report["simulation_seconds"] += time.perf_counter() - before_step
                report["control_steps"] += 1
                report["trial_steps"] += len(running)
                success = env.termination_manager.get_term("success").tolist()
                timeout = env.termination_manager.get_term("time_out").tolist()
                for slot in running:
                    trial = active[slot]
                    trial["steps"] += 1
                    if not (success[slot] or timeout[slot]):
                        continue
                    trial.update(success=bool(success[slot]), timeout=bool(timeout[slot]))
                    report["episodes"].append(trial)
                    episodes.write(json.dumps(trial) + "\n")
                    episodes.flush()
                    del active[slot]
                    print(
                        f"{trial['family']} trial {trial['trial']}: success={trial['success']}, steps={trial['steps']}",
                        flush=True,
                    )
                # Auto-reset worlds, including idle ones, hold their current reset arm pose with open grippers.
                idle = [slot for slot in range(env.num_envs) if slot not in active]
                actions[idle, :7] = observation["policy"]["arm_joint_pos"][idle]
                actions[idle, 7] = 1.0
                if time.perf_counter() - last_save >= 10:
                    save()
                    last_save = time.perf_counter()
    if sum(trial["steps"] for trial in report["episodes"]) != report["trial_steps"]:
        raise AssertionError("Terminal trial accounting lost or duplicated simulation steps.")
    if len(report["episodes"]) != report["expected_trials"]:
        raise AssertionError("Evaluation did not complete its declared number of trials.")
    counts = Counter(trial["family"] for trial in report["episodes"])
    successes = Counter(trial["family"] for trial in report["episodes"] if trial["success"])
    report["family_results"] = {
        family: {
            "trials": counts[family],
            "successes": successes[family],
            "success_rate": successes[family] / counts[family],
        }
        for family in selected
    }
    report.update(phase="complete", rollout_seconds=time.perf_counter() - started)


def main() -> None:
    """Resolve the registered scene, prepare inference, then evaluate and save results."""
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, overrides=overrides)
    cfg.seed = args.seed
    if unknown := set(args.tasks or ()) - set(cfg.tasks):
        parser.error(f"Unknown task families: {sorted(unknown)}")
    camera_period = (args.horizon - 0.5) * cfg.decimation * cfg.sim.dt
    cfg.scene.over_shoulder_left_camera = cfg.scene.over_shoulder_left_camera.replace(update_period=camera_period)
    cfg.scene.wrist_cam = cfg.scene.wrist_cam.replace(update_period=camera_period)
    args.output.mkdir(parents=True, exist_ok=False)
    print(f"Saving evaluation reports to {args.output.resolve()}", flush=True)
    started = time.perf_counter()
    report = {
        "phase": "initializing",
        "task": args.task,
        "seed": args.seed,
        "episodes": [],
        "trial_steps": 0,
        "control_steps": 0,
        "policy_queries": 0,
        "policy_seconds": 0.0,
        "simulation_seconds": 0.0,
        "num_envs": args.num_envs,
        "trials_per_family": args.trials,
        "horizon": args.horizon,
        "overrides": overrides,
    }

    def save() -> None:
        report["wall_seconds"] = time.perf_counter() - started
        temporary = args.output / "result.json.tmp"
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(args.output / "result.json")

    try:
        with policy_connection() as socket:
            metadata = msgpack.unpackb(socket.recv(timeout=180), object_hook=unpack_array)
            if metadata.get("action_space") not in (None, "absolute_joint_position"):
                raise ValueError(f"The policy must return absolute joint positions; server reports {metadata}.")
            report["policy_server"] = metadata
            with launch_simulation(cfg, args):
                with closing(gym.make(args.task, cfg=cfg).unwrapped) as env:
                    report.update(phase="running", scene_initialization_seconds=time.perf_counter() - started)
                    evaluate(env, socket, report, save)
    except BaseException as error:
        report.update(phase="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
