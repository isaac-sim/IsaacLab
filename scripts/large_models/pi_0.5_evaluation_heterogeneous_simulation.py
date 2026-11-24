# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a pi0.5 policy on the heterogeneous tasks, with the trials scheduled over the environments.

Every environment of the heterogeneous VLA environment is a slot that runs the tasks of its own objects: the
instructions that ``env.tabletop.available_tasks`` binds to two of the objects sampled into it, of the task families
``--tasks``. Each task gets ``--trials`` trials, which its slot runs one after the other. The slots are synchronized:
every ``--horizon`` steps, all slots that run a trial are queried in one batch from the pi0.5 DROID joint-position
policy that RoboLab's ``policy_server.py`` serves at ``--policy`` (openpi's WebSocket protocol), with both cameras
letterboxed into 224 x 224, the arm joint positions, the gripper opening and the task's instruction. The answered
joint targets and gripper commands are executed open loop. A trial ends on its task's success or time-out, on which
the environment resets its slot; the slot then holds the start pose and starts its next trial at the next query,
where its task is requested and the slot is reset again. A trial that ends within two steps is restarted at the next
query instead of counted.

Writes to the new directory ``--output``: ``result.json`` (the results of every task and task family, the accounting
of the schedule and the time of every phase; saved every ten seconds and when the run ends or fails) and
``episodes.jsonl`` (one line per finished trial). The protocol needs ``msgpack``.

.. code-block:: bash

    uv run --with msgpack python scripts/large_models/pi_0.5_evaluation_heterogeneous_simulation.py \\
        --policy localhost:18013 --num_envs 32 --trials 2 --output logs/heterogeneous_tasks/pi05
"""

import argparse
import json
import statistics
import time
import traceback
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="IsaacContrib-VLA-Heterogeneous-Franka")
parser.add_argument("--num_envs", type=int, default=32, help="Number of slots.")
parser.add_argument("--policy", required=True, metavar="HOST:PORT", help="Address of the policy server.")
parser.add_argument("--output", required=True, help="Directory of the results, which must not exist.")
parser.add_argument("--tasks", nargs="+", metavar="NAME", help="Task families to evaluate (default: all).")
parser.add_argument("--trials", type=int, default=1, help="Trials per task.")
parser.add_argument("--horizon", type=int, default=15, help="Steps executed per policy query.")
parser.add_argument("--seed", type=int, default=0, help="Run seed of the policy's sampling noise.")
parser.add_argument("--max_steps", type=int, default=100000, help="Environment steps after which the run fails.")
parser.add_argument("--mesh_cache_dir", default=None, help="Directory of Newton's mesh cache, shared across runs.")
add_launcher_args(parser)
args = parser.parse_args()
if args.tasks and len(set(args.tasks)) != len(args.tasks):
    parser.error("--tasks must not repeat a task family")
if min(args.trials, args.horizon) < 1:
    parser.error("--trials and --horizon must be at least 1")

import gymnasium as gym  # noqa: E402
import newton  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

CAMERAS = ("over_shoulder_left_camera", "wrist_cam")
POLICY_IMAGE = 224
"""Side of the square images the pi0.5 DROID checkpoint takes; the camera images are letterboxed into them."""


@dataclass(frozen=True)
class Job:
    """One trial of a task."""

    env_id: int
    """Environment, the slot, whose objects the task binds."""
    task: int
    """Index of the task among the available tasks of its environment."""
    task_name: str
    trial_id: int


class JobManager:
    """Gives every slot the pending trials of its tasks in order and accounts for each trial once."""

    def __init__(self, jobs: list[Job]):
        self._pending: dict[int, deque[Job]] = {}
        for job in jobs:
            self._pending.setdefault(job.env_id, deque()).append(job)
        self._expected = len(jobs)
        self._running: dict[int, Job] = {}
        self._chunks: dict[int, int] = {}
        self._completed = 0

    @property
    def done(self) -> bool:
        """Whether every trial has completed."""
        return self._completed == self._expected

    def acquire(self, slot: int) -> Job | None:
        """Give an idle slot the next pending trial of its tasks; None when it has none."""
        if slot in self._running:
            raise ValueError(f"Slot {slot} still runs a trial.")
        if not self._pending.get(slot):
            return None
        self._running[slot], self._chunks[slot] = self._pending[slot].popleft(), 0
        return self._running[slot]

    def complete(self, slot: int) -> None:
        """Release the slot of a trial that ended in its task's success or time-out."""
        del self._running[slot], self._chunks[slot]
        self._completed += 1

    def request_key(self, slot: int) -> dict:
        """Return the key of the slot's next policy query, which seeds the server's sampling noise.

        The key names the trial and its query, so the answers do not depend on the schedule.
        """
        job = self._running[slot]
        key = {
            "run_seed": args.seed,
            "task_name": job.task_name,
            "episode_idx": job.trial_id,
            "chunk_idx": self._chunks[slot],
        }
        self._chunks[slot] += 1
        return key

    def snapshot(self) -> dict:
        """Return the accounting of the trials."""
        return {
            "ordering": "per_slot_in_task_order",
            "expected_jobs": self._expected,
            "pending_jobs": sum(len(jobs) for jobs in self._pending.values()),
            "running_jobs": len(self._running),
            "completed_jobs": self._completed,
        }


@dataclass
class Trial:
    """The trial a slot runs."""

    job: Job
    steps: int = 0
    """Steps of the current attempt."""
    discarded_steps: int = 0
    """Steps of attempts that ended within two steps, and of waiting for their restarts."""
    early_reset_count: int = 0
    restarting: bool = False
    """Whether the attempt ended within two steps and the slot waits for its restart."""


class Timing:
    """Wall-clock time of the phases of the rollout [s]. GPU work is not synchronized."""

    def __init__(self):
        self._samples: dict[str, list[float]] = {}

    @contextmanager
    def measure(self, name: str):
        """Time a scope."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self._samples.setdefault(name, []).append(time.perf_counter() - started)

    def clear(self) -> None:
        """Discard the measurements."""
        self._samples.clear()

    def summary(self) -> dict:
        """Return the count and the distribution of every phase."""
        return {
            name: {
                "count": len(samples),
                "total_s": sum(samples),
                "mean_s": statistics.fmean(samples),
                "p50_s": statistics.median(samples),
                "max_s": max(samples),
            }
            for name, samples in sorted(self._samples.items())
        }


def resize_with_pad(image: np.ndarray, size: int) -> np.ndarray:
    """Fit an image into a black ``size`` x ``size`` square without distortion, as openpi's client does (bilinear)."""
    picture = Image.fromarray(image)
    ratio = max(picture.width / size, picture.height / size)
    resized = picture.resize((int(picture.width / ratio), int(picture.height / ratio)), resample=Image.BILINEAR)
    square = Image.new(picture.mode, (size, size), 0)
    square.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.asarray(square)


def pack_array(obj):
    """msgpack hook for NumPy arrays and scalars: openpi's wire format."""
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj: dict):
    """The inverse of :func:`pack_array`."""
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class Policy:
    """The pi0.5 policy that RoboLab's ``policy_server.py`` serves, queried for several slots in one request."""

    def __init__(self, address: str, env, timing: Timing):
        import msgpack
        import websockets.sync.client

        self._msgpack = msgpack
        host, port = address.rsplit(":", 1)
        # Timeouts make a silent server fail the run instead of hanging it.
        self._socket = websockets.sync.client.connect(
            f"ws://{host}:{port}", compression=None, max_size=None, open_timeout=60
        )
        self.metadata = msgpack.unpackb(self._socket.recv(timeout=180), object_hook=unpack_array)
        self._device, self._timing = env.device, timing

    def close(self) -> None:
        """Close the connection to the server."""
        self._socket.close()

    def __call__(self, obs: dict, slots: list[int], keys: list[dict], instructions: list[str]) -> torch.Tensor:
        """Return the arm joint targets and gripper commands (1 closes, 0 opens) of the next ``--horizon`` steps.

        Args:
            obs: The environment's observations.
            slots: The slots to query.
            keys: The key of each slot's query (see :meth:`JobManager.request_key`).
            instructions: The instruction of each slot's task.

        Returns:
            The actions of the slots, shape (len(slots), horizon, 8).
        """
        with self._timing.measure("observation_readback"):
            policy = obs["policy"]
            images = [policy[name][slots].cpu().numpy() for name in CAMERAS]
            arm = policy["arm_joint_pos"][slots].cpu().numpy().astype(np.float32)
            gripper = policy["gripper_pos"][slots].cpu().numpy().astype(np.float32)
            requests = [
                {
                    "observation/exterior_image_1_left": resize_with_pad(images[0][row], POLICY_IMAGE),
                    "observation/wrist_image_left": resize_with_pad(images[1][row], POLICY_IMAGE),
                    "observation/joint_position": arm[row],
                    "observation/gripper_position": gripper[row],
                    "prompt": instructions[row],
                    "benchmark_key": keys[row],
                    "benchmark_group": "validate",
                }
                for row in range(len(slots))
            ]
        with self._timing.measure("policy_rpc"):
            self._socket.send(self._msgpack.packb({"robolab_batch": requests}, default=pack_array))
            reply = self._socket.recv(timeout=600)
        if isinstance(reply, str):  # the server reports an exception as text
            raise RuntimeError(f"policy server: {reply}")
        actions = np.asarray(self._msgpack.unpackb(reply, object_hook=unpack_array)["actions"], dtype=np.float32)
        if (
            actions.ndim != 3
            or actions.shape[0] != len(slots)
            or actions.shape[1] < args.horizon
            or actions.shape[2] != 8
            or not np.isfinite(actions).all()
        ):
            raise ValueError(f"policy server returned actions of shape {actions.shape}")
        actions = torch.as_tensor(actions[:, : args.horizon], device=self._device)
        actions[..., 7] = (actions[..., 7] > 0.5).float()
        return actions


def rollout(
    env, policy: Policy, manager: JobManager, timing: Timing, obs: dict, report: dict, episodes_stream, save
) -> None:
    """Run every trial on the slots; the module docstring describes the schedule.

    Args:
        env: The heterogeneous VLA environment.
        policy: The policy.
        manager: The trials.
        timing: The time of the phases.
        obs: The environment's observations at its start state.
        report: The report, to which the finished trials and the accounting are added.
        episodes_stream: File that gets one line per finished trial.
        save: Function that saves the report.
    """
    terminations, tabletop = env.termination_manager, env.tabletop
    # Slots without a trial hold the start pose with the gripper open.
    idle = torch.cat((obs["policy"]["arm_joint_pos"], obs["policy"]["gripper_pos"].round()), dim=1)
    actions = idle.clone()
    active: dict[int, Trial] = {}
    # Trials of idle slots, which start at the next query.
    deferred: dict[int, Job] = {}
    started = last_save = time.perf_counter()

    def release(slot: int) -> None:
        """Free a slot whose trial ended, which the environment has reset, and give it its next pending trial."""
        del active[slot]
        actions[slot] = idle[slot]
        job = manager.acquire(slot)
        if job is not None:
            deferred[slot] = job

    def step() -> dict:
        """Step the environments with the current actions and settle the trials that ended."""
        nonlocal last_save
        if report["control_steps"] >= args.max_steps:
            raise TimeoutError(f"The rollout exceeded {args.max_steps} environment steps.")
        report["occupied_env_steps"] += len(active)
        # The environment's gripper action closes below zero.
        with timing.measure("env_step"):
            obs, *_ = env.step(torch.cat((actions[:, :7], 0.5 - actions[:, 7:8]), dim=1))
        report["control_steps"] += 1
        # The environment has reset the slots whose episodes ended; the terms still hold the reasons.
        success, timeout = terminations.get_term("success").tolist(), terminations.get_term("time_out").tolist()
        for slot, trial in list(active.items()):
            if trial.restarting:
                trial.discarded_steps += 1
                continue
            trial.steps += 1
            if success[slot] or timeout[slot]:
                if trial.steps <= 2:
                    trial.early_reset_count += 1
                    trial.discarded_steps += trial.steps
                    trial.restarting = True
                    continue
                record = {
                    "task_name": trial.job.task_name,
                    "trial_id": trial.job.trial_id,
                    "env_id": slot,
                    "success": success[slot],
                    "timeout": timeout[slot],
                    "steps": trial.steps,
                    "early_reset_count": trial.early_reset_count,
                    "discarded_steps": trial.discarded_steps,
                    "shared_control_step": report["control_steps"],
                }
                manager.complete(slot)
                report["episodes"].append(record)
                episodes_stream.write(json.dumps(record) + "\n")
                episodes_stream.flush()
                release(slot)
                print(
                    f"{trial.job.task_name} trial {trial.job.trial_id}: {'success' if success[slot] else 'timeout'}"
                    f" after {trial.steps} steps ({len(report['episodes'])} trials,"
                    f" {time.perf_counter() - started:.0f} s)",
                    flush=True,
                )
        report["rollout_seconds"] = time.perf_counter() - started
        if time.perf_counter() - last_save > 10.0:
            save()
            last_save = time.perf_counter()
        return obs

    # Every slot takes the first trial of its tasks; afterwards a slot takes its next one when a trial ends.
    for slot in range(env.num_envs):
        job = manager.acquire(slot)
        if job is not None:
            deferred[slot] = job
    while not manager.done:
        # At the round barrier the next trials start: their slots get their tasks and are reset, as are the slots
        # whose trials restart.
        restarting = [slot for slot, trial in active.items() if trial.restarting]
        if deferred:
            slots = sorted(deferred)
            tasks = torch.tensor([deferred[slot].task for slot in slots], device=env.device)
            report["task_switches"] += int((tabletop.task_ids[slots] != tasks).sum())
            with timing.measure("task_switch"):
                tabletop.set_task(slots, tasks.tolist())
        if deferred or restarting:
            with timing.measure("reset"):
                obs, _ = env.reset(env_ids=torch.tensor(sorted({*deferred, *restarting}), device=env.device))
            for slot in restarting:
                active[slot].restarting, active[slot].steps = False, 0
            active.update({slot: Trial(job) for slot, job in deferred.items()})
            deferred.clear()
        if not active:
            raise AssertionError("Unfinished trials without a slot.")
        # One query for all slots that run a trial.
        slots = sorted(active)
        keys = [manager.request_key(slot) for slot in slots]
        chunks = policy(obs, slots, keys, [tabletop.instructions[slot] for slot in slots])
        rows = {slot: row for row, slot in enumerate(slots)}
        report["inference_rounds"] += 1
        report["batch_histogram"][str(len(slots))] = report["batch_histogram"].get(str(len(slots)), 0) + 1
        if env.device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info(env.device)
            report["gpu_used_mib_peak"] = max(report.get("gpu_used_mib_peak", 0), round((total - free) / 2**20))
        for offset in range(args.horizon):
            if not active:
                break
            # A slot whose trial ended holds the start pose until the next query.
            running = sorted(active)
            actions[running] = chunks[[rows[slot] for slot in running], offset]
            obs = step()


def finalize(report: dict, tasks: list[dict]) -> None:
    """Check that every step of a slot with a trial is accounted for; record the results of the tasks and families."""
    accounted = sum(episode["steps"] + episode["discarded_steps"] for episode in report["episodes"])
    if accounted != report["occupied_env_steps"]:
        raise AssertionError(f"Unaccounted environment steps: {accounted} != {report['occupied_env_steps']}")
    report["slot_utilization"] = accounted / max(args.num_envs * report["control_steps"], 1)
    episodes = Counter(episode["task_name"] for episode in report["episodes"])
    successes = Counter(episode["task_name"] for episode in report["episodes"] if episode["success"])
    report["task_results"] = [
        {**task, "episodes": episodes[task["task_name"]], "successes": successes[task["task_name"]]} for task in tasks
    ]
    report["family_results"] = [
        {
            "family": family,
            "episodes": sum(task["episodes"] for task in report["task_results"] if task["family"] == family),
            "successes": sum(task["successes"] for task in report["task_results"] if task["family"] == family),
        }
        for family in dict.fromkeys(task["family"] for task in tasks)
    ]
    report["successes"] = sum(successes.values())
    report["success_rate"] = report["successes"] / max(len(report["episodes"]), 1)
    report.update(phase="complete", passed=True)


def main() -> None:
    """Build the environment, run the trials and write the report."""
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    families = args.tasks or list(env_cfg.tasks)
    if unknown := [name for name in families if name not in env_cfg.tasks]:
        parser.error(f"unknown task families: {unknown}")
    if args.mesh_cache_dir:
        newton.config.cache_dir_meshes = args.mesh_cache_dir
    # The policy observes once per chunk, so the cameras render once per chunk: due a little before its last step,
    # as the sensor clocks add up float32 step times.
    for camera in (env_cfg.scene.over_shoulder_left_camera, env_cfg.scene.wrist_cam):
        camera.update_period = (args.horizon - 0.5) * env_cfg.decimation * env_cfg.sim.dt
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    timing, manager, env, policy = Timing(), None, None, None
    report = {
        "passed": False,
        "phase": "scene_initialization",
        "arguments": {key: str(value) for key, value in vars(args).items()},
        "num_slots": args.num_envs,
        "episodes_per_task": args.trials,
        "run_seed": args.seed,
        "episodes": [],
        "control_steps": 0,
        "occupied_env_steps": 0,
        "inference_rounds": 0,
        "batch_histogram": {},
        "task_switches": 0,
    }
    started = time.perf_counter()

    def save() -> None:
        """Write the report, replacing the last one at once."""
        report["wall_seconds"] = time.perf_counter() - started
        report["timing"] = timing.summary()
        if manager is not None:
            report["schedule"] = manager.snapshot()
        temporary = output / "result.json.tmp"
        temporary.write_text(json.dumps(report, indent=1, default=str))
        temporary.replace(output / "result.json")

    try:
        with launch_simulation(env_cfg, args):
            try:
                env = gym.make(args.task, cfg=env_cfg).unwrapped
                report["scene_initialization_seconds"] = time.perf_counter() - started
                # The tasks of every slot: the instructions that its sampled objects afford.
                tasks = [
                    {
                        "task_name": f"env_{env_id}/{task.family}/{task.object}/{task.target}",
                        "env_id": env_id,
                        "task": index,
                        "family": task.family,
                        "instruction": task.instruction,
                    }
                    for env_id, available in enumerate(env.tabletop.available_tasks)
                    for index, task in enumerate(available)
                    if task.family in families
                ]
                if not tasks:
                    raise RuntimeError("The objects sampled into the environments afford no task to evaluate.")
                report.update(task_count=len(tasks), expected_episodes=len(tasks) * args.trials)
                manager = JobManager(
                    [
                        Job(task["env_id"], task["task"], task["task_name"], trial)
                        for task in tasks
                        for trial in range(args.trials)
                    ]
                )
                policy = Policy(args.policy, env, timing)
                print(f"policy server at {args.policy}: {policy.metadata}", flush=True)
                report.update(phase="warmup", policy_server=policy.metadata)
                save()
                # The first render and the first query warm up the renderer and the policy; the rollout excludes them.
                warmup_started = time.perf_counter()
                obs, _ = env.reset()
                warmup_key = {"run_seed": args.seed, "task_name": "warmup", "episode_idx": 0, "chunk_idx": 0}
                policy(obs, [0], [warmup_key], [tasks[0]["instruction"]])
                report["warmup_seconds"] = time.perf_counter() - warmup_started
                timing.clear()
                report["phase"] = "running"
                with (output / "episodes.jsonl").open("w") as episodes_stream:
                    rollout(env, policy, manager, timing, obs, report, episodes_stream, save)
                finalize(report, tasks)
                print(
                    f"SUCCESS {report['successes']}/{len(report['episodes'])} ({report['success_rate']:.1%}),"
                    f" {report['rollout_seconds']:.0f} s",
                    flush=True,
                )
            finally:
                if policy is not None:
                    policy.close()
                if env is not None:
                    env.close()
    except BaseException as error:
        report.update(phase="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
