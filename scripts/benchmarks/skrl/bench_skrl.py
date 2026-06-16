# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SKRL training-benchmark adapter.

Runs real training under a :class:`~isaaclab.test.benchmark.BenchmarkMonitor` and emits a
:class:`~isaaclab.test.benchmark.schema.TrainingBundle` JSON file. Dispatched from
``scripts/benchmarks/training.py`` via ``--rl_library skrl``.

The ``BenchmarkTrainer`` subclass captures per-iteration wall-clock time, mean reward, and
episode length directly from the training loop — the reward/episode-length/timing series need
no TensorBoard round-trip (success-rate is still read from TensorBoard).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from scripts.benchmarks._common import get_backend_types, import_module_from_path, preset_tokens

# ---------------------------------------------------------------------------
# Path setup — locate scripts/reinforcement_learning and load common helpers
# via explicit file path so that scripts/reinforcement_learning is never added
# to sys.path (it has no __init__.py and is not an importable package).
# ---------------------------------------------------------------------------

_BENCH_DIR = Path(__file__).resolve().parents[1]
_RL_SCRIPTS = _BENCH_DIR.parent / "reinforcement_learning"

_common = import_module_from_path("isaaclab_rl_common", _RL_SCRIPTS / "common.py")


# ---------------------------------------------------------------------------
# BenchmarkTrainer — captures per-iteration timing + reward + ep length
# ---------------------------------------------------------------------------


def _build_benchmark_trainer_class():
    """Build and return the BenchmarkTrainer class.

    Deferred to a factory so the skrl import only happens after the
    simulation has been launched (avoids premature CUDA context init).

    Returns:
        The ``BenchmarkTrainer`` class.
    """
    from skrl.trainers.torch import SequentialTrainer

    class BenchmarkTrainer(SequentialTrainer):
        """SequentialTrainer that records per-iteration timing, reward, and episode length.

        Wraps ``agent.post_interaction`` and ``env.step`` at train-time to detect rollout
        boundaries without duplicating the full training loop.  Per-iteration attributes are
        populated after :meth:`train` returns and read directly by the benchmark builder.

        Attributes:
            iter_times_s: Wall-clock seconds per iteration (one rollout-buffer fill).
            iter_rewards: Mean reward across all env steps during each rollout.
            iter_ep_lengths: Last observed mean episode length per iteration.
        """

        def __init__(self, env, agents, cfg=None) -> None:
            super().__init__(env=env, agents=agents, cfg=cfg)
            self.iter_times_s: list[float] = []
            self.iter_rewards: list[float] = []
            self.iter_ep_lengths: list[float] = []

        def train(self) -> None:
            """Run training and record per-iteration metrics.

            Resolves the rollout boundary from ``agent.cfg.rollouts`` (skrl >= 2.x)
            or ``agent._rollouts`` (skrl < 2.x). Falls back to the stock training loop
            for multi-agent and for agents without a rollout boundary — those leave
            the per-iter attributes empty.
            """
            if self.num_simultaneous_agents > 1 or self.env.num_agents > 1:
                print(
                    "[WARNING] BenchmarkTrainer: multi-agent — per-iteration timing/reward/episode-length"
                    " series will be empty; the bundle's runtime/learning metrics will be zero.",
                    file=sys.stderr,
                )
                super().train()
                return

            agent_obj = self.agents
            agent_cfg = getattr(agent_obj, "cfg", None)
            rollouts_val = (
                getattr(agent_cfg, "rollouts", None) if agent_cfg is not None else getattr(agent_obj, "_rollouts", None)
            )
            if not rollouts_val:
                print(
                    "[WARNING] BenchmarkTrainer: unresolved rollout boundary — per-iteration"
                    " timing/reward/episode-length series will be empty;"
                    " the bundle's runtime/learning metrics will be zero.",
                    file=sys.stderr,
                )
                super().train()
                return

            rollouts = int(rollouts_val)
            timesteps = self.cfg.timesteps
            max_iters = timesteps // rollouts

            # Intercept env.step to accumulate per-step rewards.
            _orig_step = self.env.step
            _reward_sum: list[float] = [0.0]
            _reward_count: list[int] = [0]

            def _patched_step(actions):
                result = _orig_step(actions)
                _reward_sum[0] += float(result[1].mean().item())
                _reward_count[0] += 1
                return result

            # Intercept agent.post_interaction to detect rollout boundaries.
            _orig_post = agent_obj.post_interaction
            _iter_start_ns: list[int] = [time.perf_counter_ns()]

            # Holds the last non-empty episode-length snapshot across steps.
            _last_ep_len: list[float] = [0.0]

            def _patched_post(*, timestep: int, timesteps: int) -> None:
                # Snapshot tracking_data BEFORE calling the original post_interaction,
                # which may call write_tracking_data() → tracking_data.clear().
                td = getattr(agent_obj, "tracking_data", {})
                ep_len_key = next((k for k in td if "episode" in k.lower() and "timestep" in k.lower()), None)
                ep_len_val = td.get(ep_len_key, []) if ep_len_key else []
                if ep_len_val:
                    _last_ep_len[0] = float(sum(ep_len_val) / len(ep_len_val))

                _orig_post(timestep=timestep, timesteps=timesteps)

                if (timestep + 1) % rollouts == 0:
                    iter_end_ns = time.perf_counter_ns()
                    self.iter_times_s.append((iter_end_ns - _iter_start_ns[0]) / 1e9)
                    self.iter_rewards.append(_reward_sum[0] / max(_reward_count[0], 1))
                    self.iter_ep_lengths.append(_last_ep_len[0])
                    # Reset accumulators for next iteration.
                    _iter_start_ns[0] = time.perf_counter_ns()
                    _reward_sum[0] = 0.0
                    _reward_count[0] = 0

            agent_obj.post_interaction = _patched_post
            self.env.step = _patched_step
            try:
                super().train()
            finally:
                agent_obj.post_interaction = _orig_post
                self.env.step = _orig_step

            self.iter_times_s = self.iter_times_s[:max_iters]
            self.iter_rewards = self.iter_rewards[:max_iters]
            self.iter_ep_lengths = self.iter_ep_lengths[:max_iters]

    return BenchmarkTrainer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]):
    """Parse CLI arguments and forward the remaining Hydra preset tokens via ``sys.argv``.

    Args:
        argv: Raw command-line arguments (``sys.argv[1:]`` after dispatcher
            strips ``--rl_library``).

    Returns:
        Tuple of ``(parsed_args, remaining)`` where *remaining* are the verbatim Hydra
        preset tokens written back to ``sys.argv`` for ``launch_simulation`` to pick up.
    """
    import argparse

    from isaaclab_tasks.utils import setup_preset_cli

    add_common_train_args = _common.add_common_train_args
    add_isaaclab_launcher_args = _common.add_isaaclab_launcher_args
    enable_cameras_for_video = _common.enable_cameras_for_video

    parser = argparse.ArgumentParser(description="Benchmark RL training with SKRL.")
    add_common_train_args(
        parser,
        agent_default=None,
        agent_help=(
            "Name of the RL agent configuration entry point. Defaults to None, in which"
            " case --algorithm is used to determine the default agent entry point."
        ),
    )
    parser.add_argument(
        "--ml_framework",
        type=str,
        default="torch",
        choices=["torch", "jax"],
        help="ML framework used for training the skrl agent.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["AMP", "PPO", "IPPO", "MAPPO"],
        help="The RL algorithm used for training the skrl agent.",
    )
    parser.add_argument("--output_path", type=str, default=".", help="Directory to write the output JSON.")
    parser.add_argument(
        "--benchmark_backend",
        type=str,
        default="schema",
        help=(
            "Output format(s): comma-separated list of 'schema' (default, the typed benchmark bundle),"
            " 'omniperf', 'osmo', 'json', 'summary'. Legacy long-form aliases accepted."
            " Example: 'schema,omniperf'."
        ),
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.1,
        help="EMA smoothing factor for learning curves (higher = more recent weight).",
    )
    parser.add_argument(
        "--no_series",
        action="store_true",
        default=False,
        help="Omit per-iteration series data from the bundle to reduce file size.",
    )

    from scripts.benchmarks.early_stop import add_success_cli_args

    add_success_cli_args(parser)
    add_isaaclab_launcher_args(parser)

    args_cli, remaining_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    sys.argv = [sys.argv[0]] + remaining_args

    return args_cli, remaining_args


# ---------------------------------------------------------------------------
# Main run function (dispatch contract)
# ---------------------------------------------------------------------------


def run(argv: list[str]) -> None:
    """Run the SKRL training benchmark and write a :class:`~isaaclab.test.benchmark.schema.TrainingBundle`.

    Args:
        argv: Command-line arguments, excluding the script path (i.e. ``sys.argv[1:]``
            after the dispatcher has stripped ``--rl_library``).
    """
    import contextlib
    import os

    import torch

    from isaaclab.app import launch_simulation
    from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, builders, capture
    from isaaclab.test.benchmark.backend_descriptor import BACKEND_DESCRIPTORS
    from isaaclab.test.benchmark.metrics import parse_tf_logs
    from isaaclab.test.benchmark.schema import StartupTime

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401

    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    from isaaclab_tasks.utils import resolve_task_config

    apply_env_overrides = _common.apply_env_overrides
    from scripts.benchmarks.early_stop import get_success_tracker

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    args_cli, remaining_args = _parse_args(argv)

    # Resolve agent entry point (mirrors train_skrl._resolve_agent_entry_point).
    if args_cli.agent is None:
        algorithm = args_cli.algorithm.lower()
        agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"
    else:
        agent_cfg_entry_point = args_cli.agent
        algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, agent_cfg_entry_point)

    start_utc = capture.now_utc_iso()
    app_t0 = time.perf_counter_ns()

    with launch_simulation(env_cfg, args_cli):
        app_t1 = time.perf_counter_ns()

        apply_env_overrides(args_cli, env_cfg)

        if args_cli.max_iterations:
            agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
        agent_cfg["trainer"]["close_environment_at_exit"] = False

        agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 0)
        env_cfg.seed = agent_cfg["seed"]

        # Build log_dir (mirrors train_skrl.py).
        log_root_path = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
        from datetime import datetime

        log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
        if agent_cfg["agent"]["experiment"]["experiment_name"]:
            log_dir_name += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
        agent_cfg["agent"]["experiment"]["directory"] = log_root_path
        agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir_name
        log_dir = os.path.join(log_root_path, log_dir_name)

        tokens = preset_tokens(remaining_args)
        backend_types = get_backend_types(args_cli.benchmark_backend)

        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="benchmark_training",
            backend_type=backend_types,
            output_path=args_cli.output_path,
            use_recorders=True,
            frametime_recorders=any(t in ("summary", "omniperf") for t in backend_types),
            output_prefix=f"benchmark_training_{args_cli.task}",
            workflow_metadata={
                "metadata": [
                    {"name": "task", "data": args_cli.task},
                    {"name": "seed", "data": agent_cfg["seed"]},
                    {"name": "num_envs", "data": args_cli.num_envs},
                    {"name": "max_iterations", "data": args_cli.max_iterations},
                    {"name": "algorithm", "data": args_cli.algorithm},
                    {"name": "presets", "data": ",".join(tokens)},
                ]
            },
        )

        import gymnasium as gym

        env_t0 = time.perf_counter_ns()
        env = gym.make(args_cli.task, cfg=env_cfg)
        env_t1 = time.perf_counter_ns()

        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

        if args_cli.ml_framework.startswith("torch"):
            from skrl.utils.runner.torch import Runner
        else:
            # The per-iteration BenchmarkTrainer subclasses ``skrl.trainers.torch.SequentialTrainer``,
            # so the timing path is torch-only; fail clearly instead of injecting a torch trainer
            # into a non-torch runner.
            raise NotImplementedError(
                f"The skrl training benchmark supports --ml_framework torch only; got {args_cli.ml_framework!r}."
            )

        BenchmarkTrainer = _build_benchmark_trainer_class()

        class _BenchmarkRunner(Runner):
            """Runner variant that injects ``BenchmarkTrainer`` instead of the stock trainer.

            Keeps ``agent.init()`` / ``SummaryWriter`` single-fire by overriding
            ``_generate_trainer`` before the runner's ``__init__`` has run the stock
            trainer construction.
            """

            def _generate_trainer(self, env, cfg, agent):
                from skrl.trainers.torch import SequentialTrainerCfg

                trainer_cfg = SequentialTrainerCfg(**self._process_cfg(cfg["trainer"]))
                return BenchmarkTrainer(env=env, agents=agent, cfg=trainer_cfg)

        runner = _BenchmarkRunner(env, agent_cfg)
        bt = runner._trainer

        with BenchmarkMonitor(benchmark, interval=1.0):
            runner.run()

        benchmark.update_manual_recorders()

        iter_times_s = list(bt.iter_times_s)
        reward_series = list(bt.iter_rewards)
        ep_len_series = list(bt.iter_ep_lengths)

        rollouts = int(agent_cfg["agent"]["rollouts"])
        num_envs = env.unwrapped.num_envs
        steps_per_iteration = num_envs * rollouts
        fps = [steps_per_iteration / t for t in iter_times_s if t > 0]

        startup = StartupTime(
            app_launch=(app_t1 - app_t0) / 1e9,
            env_creation=(env_t1 - env_t0) / 1e9,
            first_step=(iter_times_s[0] if iter_times_s else 0.0),
        )

        runtime = builders.build_runtime(
            startup_time_s=startup,
            iteration_times_s=iter_times_s,
            collection_fps=fps,
            total_fps=fps,
            steps_per_iteration=steps_per_iteration,
        )

        learning = builders.build_learning(
            reward_series=reward_series,
            ep_length_series=ep_len_series,
            ema_alpha=args_cli.ema_alpha,
            keep_series=not args_cli.no_series,
        )

        # Success rate: attempt to parse from TensorBoard logs (best-effort).
        desc = BACKEND_DESCRIPTORS["skrl"]
        log_data = parse_tf_logs(log_dir, desc.tfevents_pattern)
        success_tracker = get_success_tracker(args_cli, None, log_data)
        success_rate = round(success_tracker.tail_mean, 4) if (success_tracker and success_tracker.history) else None

        versions = capture.capture_versions(benchmark)
        hardware = capture.capture_hardware(benchmark)
        resources = capture.capture_resources(benchmark)
        cfg = capture.run_config_from_presets(tokens)

        end_utc = capture.now_utc_iso()
        stamp = end_utc.translate(str.maketrans("", "", ":-"))[:15]
        seed = agent_cfg["seed"] if agent_cfg.get("seed") is not None else 0

        run_identity = builders.build_run_identity(
            run_id=capture.synth_run_id("skrl", cfg.physics_backend, args_cli.task, seed, stamp),
            framework="skrl",
            config=cfg,
            task=args_cli.task,
            seed=seed,
            start_utc=start_utc,
            end_utc=end_utc,
            num_envs=num_envs,
            max_iterations=args_cli.max_iterations,
        )

        series_extra = {"series_unavailable": "skrl_benchmark_trainer_fallback"} if not iter_times_s else None
        bundle = builders.build_training_bundle(
            run=run_identity,
            versions=versions,
            hardware=hardware,
            runtime=runtime,
            resources=resources,
            learning=learning,
            success_rate=success_rate,
            checkpoint_path=None,
            video_path=None,
            extra=series_extra,
        )

        benchmark.attach_bundle(bundle)

        benchmark._finalize_impl()

        env.close()


if __name__ == "__main__":
    run(sys.argv[1:])
