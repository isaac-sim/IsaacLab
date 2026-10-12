# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record, train, and deploy a learned Cartpole, and check that every stage keeps the data semantics."""

import pytest

pytest.importorskip("physicsnemo_robotics")

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import warp as wp
from isaaclab_physicsnemo import (
    CONTROL_KEY,
    NewtonBinding,
    SmoothRandomActions,
    TrainingSettings,
    TrainingSummary,
    TrajectoryDatasetRecorder,
    apply_learned_physics,
    datasets,
    preflight_learned_physics,
    train_dynamics_artifact,
)
from isaaclab_physicsnemo.resident_training import materialize_windows
from physicsnemo_robotics.data import NewtonTrajectoryZarrReader, TransitionWindowReader
from physicsnemo_robotics.dynamics import (
    ControlFieldEncoder,
    DynamicsFrame,
    TransitionAdapter,
    TransitionSpec,
    newton_state_codec,
)
from physicsnemo_robotics.training import fit_transition_normalizer, normalize_transition
from physicsnemo_robotics.training.distributed import build_stock_data_pipeline

from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv

# The Cartpole scenes are those of the example scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "cartpole"))
from cartpole_env_cfg import CARTPOLE_REMOVE_ROOT_TRANSLATION, CartpoleTeacherEnvCfg  # noqa: E402

_NUM_ENVS = 16
_TINY_MODEL = {"temporal_depth": 1, "num_heads": 2, "hidden_size": 32, "head_hidden": 16}


@dataclass(frozen=True)
class _Recording:
    dataset: Path
    finished_episodes: int
    written_episodes: int
    training: TrainingSummary


def _make_cfg(device: str) -> CartpoleTeacherEnvCfg:
    cfg = CartpoleTeacherEnvCfg()
    cfg.scene.num_envs = _NUM_ENVS
    cfg.sim.device = device
    cfg.seed = 0
    return cfg


def _step(env: CartpoleEnv, steps: int, on_step=None) -> int:
    """Step with smooth random actions and return the number of finished episodes."""
    actions = SmoothRandomActions(env.num_envs, env.cfg.action_space, device=env.device, seed=0)
    dones, finished = None, 0
    with torch.inference_mode():
        for _ in range(steps):
            observations, _, terminated, truncated, _ = env.step(actions(dones))
            assert torch.isfinite(observations["policy"]).all()
            dones = terminated | truncated
            finished += int(dones.sum())
            if on_step is not None:
                on_step(dones)
    return finished


@pytest.fixture(scope="module", params=test_devices(DeviceScope.CUDA))
def device(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def recording(device: str, tmp_path_factory: pytest.TempPathFactory) -> _Recording:
    """Record MuJoCo Warp episodes with asynchronous resets and train a tiny artifact on them."""
    root = tmp_path_factory.mktemp("cartpole")
    env = CartpoleEnv(cfg=_make_cfg(device))
    try:
        env.reset(seed=0)
        with TrajectoryDatasetRecorder(
            env, root / "dataset", scenario_id="test-cartpole", remove_root_translation=CARTPOLE_REMOVE_ROOT_TRANSLATION
        ) as recorder:
            finished = _step(env, 240, recorder.record)
        training = train_dynamics_artifact(
            root / "dataset",
            env.sim.physics_manager.get_model(),
            root / "training",
            remove_root_translation=CARTPOLE_REMOVE_ROOT_TRANSLATION,
            settings=TrainingSettings(max_updates=20, batch_size=64, log_every=10, model_kwargs=_TINY_MODEL),
            device=device,
        )
    finally:
        env.close()
    SimulationContext.clear_instance()
    return _Recording(root / "dataset", finished, recorder.episodes_written, training)


def test_sharded_recording_trains_as_one_dataset(device: str, tmp_path: Path):
    """Episodes split over bounded shards train together, and a replaced recording drops its old shards."""
    env = CartpoleEnv(cfg=_make_cfg(device))
    try:
        env.reset(seed=0)
        with TrajectoryDatasetRecorder(
            env, tmp_path / "dataset", scenario_id="test-shards", shard_episodes=5
        ) as recorder:
            _step(env, 240, recorder.record)
        with TrajectoryDatasetRecorder(
            env, tmp_path / "dataset", scenario_id="test-shards", shard_episodes=1000, replace=True
        ) as replacement:
            _step(env, 60, replacement.record)
        replaced_shards = sorted(path.name for path in (tmp_path / "dataset").iterdir())
        with TrajectoryDatasetRecorder(
            env, tmp_path / "sharded", scenario_id="test-shards", shard_episodes=5
        ) as recorder:
            _step(env, 240, recorder.record)
        training = train_dynamics_artifact(
            tmp_path / "sharded",
            env.sim.physics_manager.get_model(),
            tmp_path / "training",
            settings=TrainingSettings(
                max_updates=2, batch_size=16, model_kwargs=_TINY_MODEL, rollout_horizons=(), read_workers=1
            ),
            device=device,
        )
    finally:
        env.close()
    SimulationContext.clear_instance()

    assert replaced_shards == ["shard-000"]
    lengths = [len(NewtonTrajectoryZarrReader(shard)) for shard in sorted((tmp_path / "sharded").iterdir())]
    assert len(lengths) > 2 and max(lengths) == 5 and sum(lengths) == recorder.episodes_written
    trained = len(training.train_trajectory_ids) + len(training.validation_trajectory_ids)
    assert trained == recorder.episodes_written


def test_parallel_and_cached_loading_keep_the_trajectories(
    recording: _Recording, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Worker processes and the dataset cache return the trajectories and metadata of a serial read, in order."""
    monkeypatch.setattr(datasets, "_CHUNK_TRAJECTORIES", 7)
    paths = datasets.expand_dataset_paths(recording.dataset)
    serial = datasets.load_trajectory_datasets(paths)
    cache = tmp_path / "cache.pt"
    parallel = datasets.load_trajectory_datasets(paths, workers=3, cache_path=cache)
    cached = datasets.load_trajectory_datasets(paths, cache_path=cache)

    for reader in (parallel, cached):
        assert len(reader) == len(serial) > 7
        for index in range(len(serial)):
            assert reader.trajectory_metadata(index) == serial.trajectory_metadata(index)
            expected, actual = serial[index][0], reader[index][0]
            assert set(actual.keys()) == set(expected.keys())
            for key in expected.keys():
                torch.testing.assert_close(actual[key], expected[key], atol=0.0, rtol=0.0)


def test_resident_windows_equal_the_window_reader(recording: _Recording, device: str):
    """Windows gathered on the device equal those of the PhysicsNeMo Robotics window reader and loader, in order."""
    reader = datasets.load_trajectory_datasets(datasets.expand_dataset_paths(recording.dataset))
    env = CartpoleEnv(cfg=_make_cfg(device))
    try:
        codec = newton_state_codec(env.sim.physics_manager.get_model(), "joint")
    finally:
        env.close()
    SimulationContext.clear_instance()
    features = ControlFieldEncoder("joint_f", per_world_shape=(codec.layout.dof_qd,), key=CONTROL_KEY)
    adapter = TransitionAdapter(reader.schema, codec, TransitionSpec(history=10), features=features)
    ids = list(recording.training.train_trajectory_ids)
    windows = TransitionWindowReader(reader, history=10, trajectory_ids=ids, include_reset_prefix=True)
    normalizer = fit_transition_normalizer(windows, adapter, device=torch.device(device))
    resident = materialize_windows(
        reader, ids, adapter, normalizer, history=10, device=torch.device(device), input_dtype=torch.float32
    )

    pipeline = build_stock_data_pipeline(windows, device=torch.device(device), batch_size=512, seed=0)
    try:
        expected = [normalize_transition(adapter, normalizer, batch) for batch in pipeline.loader]
    finally:
        pipeline.close()
    torch.testing.assert_close(resident.inputs, torch.cat([item[0].inputs for item in expected]), atol=0.0, rtol=0.0)
    torch.testing.assert_close(resident.targets, torch.cat([item[1] for item in expected]), atol=0.0, rtol=0.0)


def test_half_precision_inference_tracks_full_precision(recording: _Recording, device: str):
    """FP16 projections change the learned Cartpole states only by FP16 rounding."""
    states = {}
    for dtype in ("float32", "float16"):
        cfg = _make_cfg(device)
        apply_learned_physics(cfg, recording.training.artifact_path, inference_dtype=dtype)
        env = CartpoleEnv(cfg=cfg)
        try:
            env.reset(seed=0)
            binding = NewtonBinding(env.sim.physics_manager)
            codec = newton_state_codec(binding.model, "joint")
            actions = SmoothRandomActions(env.num_envs, env.cfg.action_space, device=env.device, seed=0)
            with torch.inference_mode():
                for _ in range(5):
                    env.step(actions(None))
            states[dtype] = codec.read(binding.state).clone()
        finally:
            env.close()
        SimulationContext.clear_instance()
    torch.testing.assert_close(states["float16"], states["float32"], atol=1.0e-2, rtol=1.0e-2)
    assert not torch.equal(states["float16"], states["float32"])


def test_learned_cartpole_round_trip(recording: _Recording, device: str):
    """A dataset recorded with MuJoCo Warp yields an artifact that deploys through ``NewtonCfg``.

    Deployment reuses the recording configuration with a different environment count, resets the
    fixed-base cart through the normal environment API, and must pass every preflight check. A physics
    step that differs from the artifact frame is rejected when the environment is built.
    """
    assert recording.finished_episodes > 0, "the recording should contain asynchronous environment resets"
    assert recording.written_episodes >= recording.finished_episodes
    assert recording.training.parameter_updates == 20
    assert [error.horizon for error in recording.training.rollout_nrmse][:2] == [1, 15]

    cfg = _make_cfg(device)
    cfg.scene.num_envs = _NUM_ENVS // 2
    cfg.sim.dt = 0.5
    artifact = apply_learned_physics(cfg, recording.training.artifact_path)
    assert cfg.sim.dt == pytest.approx(1.0 / 120.0)

    env = CartpoleEnv(cfg=cfg)
    try:
        env.reset(seed=0)
        report = preflight_learned_physics(env.sim.physics_manager, artifact)
        assert report.passed, str(report)
        _step(env, 30)
    finally:
        env.close()
    SimulationContext.clear_instance()

    cfg.sim.dt = 1.0 / 60.0
    with pytest.raises(ValueError, match="artifact frame duration"):
        CartpoleEnv(cfg=cfg)
    SimulationContext.clear_instance()


def test_recorded_transitions_replay_on_the_teacher(recording: _Recording, device: str):
    """Each recorded transition is what MuJoCo Warp produces from the recorded state and joint efforts.

    This pins the time alignment of the recording: the stored effort of a transition must be the one the
    solver applied between its two stored states.
    """
    reader = NewtonTrajectoryZarrReader(recording.dataset)
    samples = [data for data, _ in (reader[index] for index in range(len(reader))) if data["state"].shape[0] >= 4]
    samples = samples[:_NUM_ENVS]
    assert len(samples) == _NUM_ENVS
    frame = 2
    state = torch.stack([data["state"][frame] for data in samples]).to(device)
    effort = torch.stack([data[CONTROL_KEY][frame] for data in samples]).to(device)
    recorded_next = torch.stack([data["state"][frame + 1] for data in samples]).to(device)

    env = CartpoleEnv(cfg=_make_cfg(device))
    try:
        env.reset(seed=0)
        manager = env.sim.physics_manager
        binding = NewtonBinding(manager)
        codec = newton_state_codec(binding.model, "joint")
        codec.write(binding.state, state)
        manager.invalidate_fk()
        manager.forward()
        wp.to_torch(binding.control.joint_f).copy_(effort.reshape(-1))
        # Advance the solver alone; a full step would recompute the efforts from the actuators.
        manager._simulate_physics_only()
        replayed_next = codec.read(binding.state)
    finally:
        env.close()
    SimulationContext.clear_instance()

    torch.testing.assert_close(replayed_next, recorded_next, atol=1.0e-5, rtol=1.0e-5)


def test_deployed_solver_matches_the_offline_operator(recording: _Recording, device: str):
    """Every learned step equals the artifact operator applied offline to the same states, efforts, and resets.

    The comparison covers the cold start of every episode and the asynchronous resets that Isaac Lab
    forwards to the learned solver, so it pins how the deployment feeds states, efforts, and history to
    the model and writes its prediction back.
    """
    cfg = _make_cfg(device)
    cfg.episode_length_s = 0.2
    artifact = apply_learned_physics(cfg, recording.training.artifact_path)
    env = CartpoleEnv(cfg=cfg)
    try:
        env.reset(seed=0)
        binding = NewtonBinding(env.sim.physics_manager)
        codec = newton_state_codec(binding.model, "joint")
        features = ControlFieldEncoder("joint_f", per_world_shape=(codec.layout.dof_qd,), key=CONTROL_KEY)
        operator = artifact.as_operator(codec=codec, features=features, device=device)
        history = operator.initial_state((env.num_envs,), device=device)
        actions = SmoothRandomActions(env.num_envs, env.cfg.action_space, device=env.device, seed=0)
        dones, resets = None, 0
        with torch.inference_mode():
            for _ in range(60):
                state = codec.read(binding.state)
                _, _, terminated, truncated, _ = env.step(actions(dones))
                effort = wp.to_torch(binding.control.joint_f).reshape(env.num_envs, -1).clone()
                offline = operator.step(DynamicsFrame(state, features=effort), history, reset_mask=dones)
                history = offline.state
                dones = terminated | truncated
                resets += int(dones.sum())
                # Finished environments return their reset state instead of the prediction.
                online = codec.read(binding.state)
                torch.testing.assert_close(online[~dones], offline.output[~dones], atol=1.0e-6, rtol=0.0)
    finally:
        env.close()
    SimulationContext.clear_instance()
    assert resets > _NUM_ENVS, "the comparison should cover environments restarted after a reset"


def test_memorized_trajectories_track_the_teacher(device: str, tmp_path: Path):
    """A model fitted to a few recorded episodes reproduces them when deployed with the same actions.

    This end-to-end check covers what the step-wise tests cannot: that training pairs each recorded state
    window with the effort of its own transition. Recording the efforts one step late leaves the training
    episodes about ten times further from the teacher.
    """
    steps = 60

    def make_cfg(learned_artifact: Path | None = None) -> CartpoleTeacherEnvCfg:
        cfg = _make_cfg(device)
        cfg.episode_length_s = steps / 120.0
        if learned_artifact is not None:
            apply_learned_physics(cfg, learned_artifact)
        return cfg

    def run(env: CartpoleEnv, on_step) -> None:
        """Step the reset environment with the same actions in the teacher and the learned run."""
        actions = SmoothRandomActions(env.num_envs, env.cfg.action_space, scale=0.5, device=env.device, seed=3)
        dones = None
        with torch.inference_mode():
            for _ in range(steps - 1):
                _, _, terminated, truncated, _ = env.step(actions(dones))
                dones = terminated | truncated
                on_step(dones)

    env = CartpoleEnv(cfg=make_cfg())
    try:
        env.reset(seed=3)
        with TrajectoryDatasetRecorder(env, tmp_path / "dataset", scenario_id="test-memorize") as recorder:
            run(env, recorder.record)
        training = train_dynamics_artifact(
            tmp_path / "dataset",
            env.sim.physics_manager.get_model(),
            tmp_path / "training",
            settings=TrainingSettings(
                history=4,
                max_updates=4000,
                batch_size=256,
                learning_rate=3.0e-3,
                log_every=1000,
                model_kwargs={"temporal_depth": 2, "num_heads": 4, "hidden_size": 128, "head_hidden": 32},
                rollout_horizons=(),
            ),
            device=device,
        )
    finally:
        env.close()
    SimulationContext.clear_instance()

    reader = NewtonTrajectoryZarrReader(tmp_path / "dataset")
    trained = set(training.train_trajectory_ids)
    teacher = {
        reader.trajectory_metadata(index).world_index: reader[index][0]["state"].to(device)
        for index in range(len(reader))
        if reader.trajectory_metadata(index).trajectory_id in trained
    }

    env = CartpoleEnv(cfg=make_cfg(training.artifact_path))
    try:
        env.reset(seed=3)
        binding = NewtonBinding(env.sim.physics_manager)
        codec = newton_state_codec(binding.model, "joint")
        learned_states = [codec.read(binding.state).clone()]
        run(env, lambda dones: learned_states.append(codec.read(binding.state).clone()))
    finally:
        env.close()
    SimulationContext.clear_instance()

    learned = torch.stack(learned_states, dim=1)
    q_errors, qd_errors = [], []
    for world, recorded in teacher.items():
        length = min(recorded.shape[0], steps)
        error = codec.state_to_delta(recorded[:length], learned[world, :length]).abs()
        q_errors.append(error[:, : codec.layout.dof_q].max())
        qd_errors.append(error[:, codec.layout.dof_q :].max())
    assert len(teacher) >= _NUM_ENVS - 3
    assert torch.stack(q_errors).mean() < 0.04
    assert torch.stack(qd_errors).mean() < 0.25
