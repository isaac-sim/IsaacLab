# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train and package NeRD dynamics artifacts from recorded Isaac Lab trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import newton
import torch
from physicsnemo.distributed import DistributedManager
from physicsnemo_robotics.data import (
    InMemoryTrajectoryReader,
    NewtonTrajectoryZarrReader,
    TransitionWindowReader,
    split_reader_ids,
)
from physicsnemo_robotics.dynamics import (
    ControlFieldEncoder,
    DynamicsArtifact,
    FeatureEncoder,
    StateCodec,
    TransitionAdapter,
    TransitionSpec,
    newton_state_codec,
)
from physicsnemo_robotics.models import NewtonNeRDVectorTransformer, ParallelNeRD
from physicsnemo_robotics.training import (
    Intervals,
    RuntimeOptions,
    StopCondition,
    fit_transition_model,
    fit_transition_normalizer,
    normalize_transition,
    package_dynamics_artifact,
)

from .evaluation import HorizonError, evaluate_rollout_nrmse
from .recording import CONTROL_KEY

PARALLEL_NERD_CONFIG: dict[str, Any] = {
    "num_heads": 5,
    "head_dim": 64,
    "intermediate_size": 640,
    "head_hidden": 128,
    "temporal_attention_spans": [4, 3, 3, 3],
    "rms_norm_eps": 1.0e-6,
    "linear_probe": True,
}
"""ParallelNeRD (FlashNeRD) geometry of the PhysicsNeMo Robotics G1 application, the default for ``"parallel_nerd"``."""


@dataclass(frozen=True)
class TrainingSettings:
    """Optimization settings of :func:`train_dynamics_artifact`."""

    history: int = 10
    """Number of past frames the model observes."""

    max_updates: int = 1000
    """Number of parameter updates."""

    batch_size: int = 512
    """Number of transition windows per update."""

    learning_rate: float = 1.0e-3
    """Initial AdamW learning rate."""

    final_learning_rate: float = 1.0e-4
    """Learning rate reached linearly at the last update."""

    weight_decay: float = 0.0
    """AdamW weight decay. Zero, as in the FlashNeRD and PhysicsNeMo Robotics training recipes."""

    gradient_clip_norm: float = 1.0
    """Maximum total gradient norm."""

    validation_fraction: float = 0.1
    """Fraction of trajectories held out for validation."""

    log_every: int = 100
    """Number of updates between training log entries."""

    seed: int = 7
    """Seed of the trajectory split and the batch order."""

    architecture: Literal["nerd", "parallel_nerd"] = "nerd"
    """Network: :class:`~physicsnemo_robotics.models.NewtonNeRDVectorTransformer` (``"nerd"``) or the streaming
    :class:`~physicsnemo_robotics.models.ParallelNeRD` of FlashNeRD (``"parallel_nerd"``)."""

    model_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword overrides of the network constructor. ``"parallel_nerd"`` starts from :data:`PARALLEL_NERD_CONFIG`."""

    rollout_horizons: tuple[int, ...] = (1, 15, 100, 200)
    """Horizons of the held-out recursive-rollout evaluation after training.

    The evaluation feeds recorded efforts to the rolled-out model, so it runs only for the default
    effort-only features. Inputs that depend on the state, such as contact probes, must be recomputed from
    every predicted state; evaluate those artifacts in a deployed environment instead.
    """


@dataclass(frozen=True)
class TrainingSummary:
    """Result of :func:`train_dynamics_artifact`."""

    artifact_path: Path
    """Directory of the packaged dynamics artifact."""

    parameter_updates: int
    """Completed parameter updates."""

    train_trajectory_ids: tuple[str, ...]
    """Trajectory identifiers of the training split."""

    validation_trajectory_ids: tuple[str, ...]
    """Trajectory identifiers of the held-out validation split."""

    validation: dict[str, float]
    """Final held-out one-step metrics, keyed by metric name."""

    rollout_nrmse: tuple[HorizonError, ...]
    """Held-out recursive-rollout errors, see :func:`~isaaclab_physicsnemo.evaluate_rollout_nrmse`."""


def train_dynamics_artifact(
    dataset_path: str | Path,
    model: newton.Model,
    output_dir: str | Path,
    *,
    remove_root_translation: bool | Literal["planar"] = False,
    codec: StateCodec | None = None,
    features: FeatureEncoder | None = None,
    settings: TrainingSettings | None = None,
    device: str = "cuda:0",
) -> TrainingSummary:
    """Train a control-conditioned vector NeRD on a recorded dataset and package it as an artifact.

    The model predicts the normalized joint-state change of one physics frame from the last
    :attr:`TrainingSettings.history` states and the applied joint efforts. Trajectories, not windows,
    are split between training and validation so that held-out metrics are not leaked. The dataset is
    loaded into host memory once, so it must fit in RAM. An existing artifact in ``output_dir`` is
    replaced.

    Args:
        dataset_path: Dataset written by :class:`~isaaclab_physicsnemo.TrajectoryDatasetRecorder`.
        model: Finalized Newton model of the recording scene, used to build the state codec.
        output_dir: Directory receiving training checkpoints and the ``artifact`` directory.
        remove_root_translation: Same value the dataset was recorded with. Ignored when ``codec`` is given.
        codec: State codec the dataset was recorded with. Defaults to the PhysicsNeMo Robotics joint codec.
        features: Encoder of the model's conditioning inputs, built against ``model``. Defaults to the applied
            joint efforts. Its recorded fields must be in the dataset.
        settings: Optimization settings. Defaults to :class:`TrainingSettings`.
        device: Training device.

    Returns:
        The artifact location and final validation metrics.

    Raises:
        ValueError: If the split leaves the training or validation set empty, or the codec does not
            match the dataset schema.
    """
    settings = settings or TrainingSettings()
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    device = torch.device(device)
    output_dir = Path(output_dir).expanduser().resolve()

    # Window sampling reads many short slices per trajectory, which is far faster from memory than
    # from per-trajectory Zarr groups.
    with NewtonTrajectoryZarrReader(dataset_path) as stored:
        trajectories = [stored[index][0] for index in range(len(stored))]
        reader = InMemoryTrajectoryReader(
            trajectories,
            schema=stored.schema,
            metadata=[stored.trajectory_metadata(index) for index in range(len(stored))],
        )
    split = split_reader_ids(
        reader,
        validation_fraction=settings.validation_fraction,
        test_fraction=0.0,
        seed=settings.seed,
        group_by="trajectory",
    )
    if not split.train or not split.validation:
        raise ValueError(f"The dataset at {dataset_path} has too few trajectories for a train/validation split.")

    if codec is None:
        codec = newton_state_codec(model, "joint", joint_remove_root_translation=remove_root_translation)
    effort_only = features is None
    if effort_only:
        features = ControlFieldEncoder("joint_f", per_world_shape=(codec.layout.dof_qd,), key=CONTROL_KEY)
    transition = TransitionSpec(history=settings.history)
    adapter = TransitionAdapter(reader.schema, codec, transition, features=features)
    # A deployed solver starts every episode with a repeated first frame as history, so training includes
    # those cold-start windows.
    train_windows = TransitionWindowReader(
        reader, history=settings.history, trajectory_ids=split.train, include_reset_prefix=True
    )
    validation_windows = TransitionWindowReader(
        reader, history=settings.history, trajectory_ids=split.validation, include_reset_prefix=True
    )
    normalizer = fit_transition_normalizer(train_windows, adapter, device=device)

    dimensions = {
        "input_dim": codec.state_shape[-1] + features.signature.width,
        "prediction_dim": codec.prediction_shape[-1],
    }
    if settings.architecture == "parallel_nerd":
        network = ParallelNeRD(**dimensions, **{**PARALLEL_NERD_CONFIG, **settings.model_kwargs})
    else:
        network = NewtonNeRDVectorTransformer(
            **dimensions, max_sequence_length=settings.history, **settings.model_kwargs
        )
    network = network.to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=settings.final_learning_rate / settings.learning_rate,
        total_iters=settings.max_updates,
    )

    def normalized_delta_mse(network: torch.nn.Module, batch: Any) -> torch.Tensor:
        transition_batch, target = normalize_transition(adapter, normalizer, batch)
        return (adapter.predict(network, transition_batch).value - target).square().mean()

    result = fit_transition_model(
        network,
        adapter,
        normalizer,
        train_windows=train_windows,
        val_windows=validation_windows,
        loss_fn=normalized_delta_mse,
        loss_name="normalized_delta_mse",
        metrics={
            "delta_rmse": lambda physical, normalized: physical,
            "normalized_delta_rmse": lambda physical, normalized: normalized,
        },
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        batch_size=settings.batch_size,
        seed=settings.seed,
        gradient_clip_norm=settings.gradient_clip_norm,
        runtime=RuntimeOptions(),
        stop=StopCondition(max_updates=settings.max_updates),
        intervals=Intervals(log=settings.log_every, validation=settings.max_updates, checkpoint=settings.max_updates),
        label="isaaclab_physicsnemo",
        checkpoint_dir=output_dir / "training_checkpoints",
    )

    artifact_path = package_dynamics_artifact(
        output_dir / "artifact",
        model=network,
        schema=reader.schema,
        codec=codec,
        features=features,
        transition=transition,
        normalizer=normalizer,
        epoch=result.parameter_updates,
    )
    artifact = DynamicsArtifact.load(artifact_path)
    artifact.assert_compatible(reader.schema)
    indices = {reader.trajectory_metadata(index).trajectory_id: index for index in range(len(reader))}
    rollout_nrmse = evaluate_rollout_nrmse(
        artifact,
        reader,
        codec,
        features,
        evaluation_indices=[indices[trajectory_id] for trajectory_id in split.validation],
        reference_indices=[indices[trajectory_id] for trajectory_id in split.train],
        horizons=settings.rollout_horizons if effort_only else (),
        device=device,
    )
    return TrainingSummary(
        artifact_path=Path(artifact_path),
        parameter_updates=result.parameter_updates,
        train_trajectory_ids=tuple(split.train),
        validation_trajectory_ids=tuple(split.validation),
        validation=dict(result.validation),
        rollout_nrmse=rollout_nrmse,
    )
