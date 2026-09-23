# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for runtime visual domain randomization.

Everything the runtime needs is declared here so a run is reproducible from its
serialized environment config alone: which pixels are protected, how often a
frame is restyled, which style it gets, and what happens when generation fails.
Backends are selected through ``class_type`` the same way actuators and actions
are, so a task can swap Cosmos for a passthrough without touching the runtime.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .backends import DRBackend


@configclass
class PromptBankCfg:
    """Background styles to sample from, given by exactly one of three sources.

    A bank is scene-coupled -- it names a room, a camera layout and sometimes a
    view tiling -- so it belongs to the task rather than to this package. Inline
    ``variants`` suit a demo; ``path`` and ``ref`` suit a deployed task whose
    bank is long enough to deserve its own file.
    """

    variants: tuple[str, ...] = ()
    """Prompts given inline."""

    path: str | None = None
    """A ``.json`` list of prompts, or a ``.txt`` file with one prompt per line."""

    ref: str | None = None
    """An importable ``package.module:NAME`` sequence of prompts."""

    negative_prompt: str | None = None
    """Applied to every variant; ``None`` uses the checkpoint's own default."""

    progression: tuple[str, ...] = ()
    """Phrases appended to the chosen variant in order as a run proceeds, for
    conditions that should drift rather than be drawn independently -- time of day
    being the obvious one. Empty leaves prompts untouched."""

    progression_steps: int = 0
    """Environment steps taken to walk ``progression`` once. Zero holds the first
    phrase. The walk is clamped rather than wrapped, so a run longer than this ends
    on the last phrase instead of snapping back to the first."""

    def __post_init__(self):
        sources = [bool(self.variants), self.path is not None, self.ref is not None]
        if sum(sources) != 1:
            raise ValueError("Set exactly one of PromptBankCfg.variants, .path or .ref")
        if self.progression_steps < 0:
            raise ValueError("PromptBankCfg.progression_steps cannot be negative")


@configclass
class CameraDRCfg:
    """Which pixels of one camera are protected from regeneration.

    Classes name the FOREGROUND. Naming what to keep rather than what to replace
    means an asset nobody remembered to tag stays in the image instead of being
    silently dissolved, which is the failure that is hard to see in a reward curve.
    """

    preserve_classes: tuple[str, ...] = ()
    """Semantic classes to keep exactly, e.g. ``("robot", "cube_1", "table")``.

    Required while ``composite_foreground`` is set. Still honoured when it is
    cleared, because a backend may send the mask to the model as a guidance signal
    even though the runtime will not paste it back; leave it empty only if nothing
    in view should be protected by either mechanism. Naming a class that never
    appears raises rather than silently regenerating everything."""

    composite_foreground: bool = True
    """Whether the runtime pastes the preserved pixels back over the generated frame.

    Set, the foreground is bit-identical to the render: what the policy sees of the
    robot and the objects it manipulates is exactly what the simulator produced. The
    cost is that the generator never saw the foreground, so the two do not share
    lighting, reflections or contact shadows, and the mask boundary is a visible seam.

    Clear it to let the model render the whole frame, foreground included. The result
    is coherently lit and has no seam, but nothing constrains the appearance of the
    objects: in this task's own scene the cubes come back cyan and the robot black,
    which is fatal for a policy that must pick a cube out by colour. Worth it when
    nothing in view is task-relevant, or with a model conditioned tightly enough to
    keep the foreground faithful -- not as a default."""

    unknown_policy: Literal["preserve", "randomize"] = "preserve"
    """What to do with a semantic ID absent from ``preserve_classes``."""

    boundary_px: int = 0
    """Dilate the preserved region by this many pixels, protecting object edges."""

    def __post_init__(self):
        if self.boundary_px < 0:
            raise ValueError("CameraDRCfg.boundary_px cannot be negative")
        if self.composite_foreground and not self.preserve_classes:
            raise ValueError(
                "CameraDRCfg.preserve_classes must name the foreground while composite_foreground is set; "
                "clear composite_foreground to let the generated frame stand as-is"
            )


@configclass
class DRBackendCfg:
    """Base class for image-generation backends."""

    class_type: type[DRBackend] = MISSING
    """The backend implementation this config constructs."""

    device: str = "cuda:0"
    """The backend must reside on the device the cameras render to."""

    max_batch: int = 8
    """Frames per backend call. The runtime chunks anything larger, so this bounds
    peak generation memory independently of how many environments are randomized."""

    def __post_init__(self):
        if self.max_batch < 1:
            raise ValueError("DRBackendCfg.max_batch must be positive")


@configclass
class CosmosBackendCfg(DRBackendCfg):
    """Depth-guided transfer against a public Cosmos3 checkpoint.

    Foreground preservation is a composite performed by the runtime, not masked
    denoising: the background is generated without knowledge of what will be
    pasted over it. That is the cost of running a stock checkpoint and a stock
    ``cosmos-framework``; revisit it if boundary artifacts show up in rollouts.
    """

    checkpoint: str = "Cosmos3-Nano"
    """A name registered in ``cosmos_framework.inference.args._CHECKPOINTS``, an
    ``s3://`` URI, or a local checkpoint directory. Registered names download from
    the Hugging Face Hub on first use."""

    control_kind: Literal["depth", "seg"] = "depth"
    """Which control hint carries the scene into the generated background. Depth
    pins exact geometry; segmentation names regions instead, leaving the model
    freer to invent what fills them while keeping layout and perspective. Cosmos
    also accepts edge, blur and wsm hints, which would each need their own signal
    derived here."""

    control_guidance: float = 1.5
    """Strength of the control signal, defaulting to what Cosmos tunes for the
    depth hint. Its own per-hint values are depth 1.5, seg 2.0, edge/blur 1.5 and
    wsm 3.0; these apply only when the request omits the field, and this backend
    always sends it, so a task using a non-depth hint should set the matching
    value. Below 1.0 the control is progressively ignored, which widens the visual
    distribution at the cost of any agreement with the scene."""

    control_weight: float = 1.0
    """Weight of the control hint within the transfer spec."""

    depth_range_m: tuple[float, float] = (0.1, 6.0)
    """Metres of depth the control map spans. Normalizing against the frame's own
    min and max instead makes the map depend on the far plane: with a distant
    horizon in view the near geometry collapses into a couple of dark values and
    the control signal is effectively lost. Clamp to the range the scene actually
    occupies."""

    num_steps: int = 16
    """Sampler steps, and the main latency control: measured on an H100 at 200x200,
    cost is about 0.205 s per step plus 0.085 s fixed, so 2/6/16/35 steps take
    0.49/1.32/3.36/7.26 s per frame.

    Sixteen is the point where this checkpoint resolves a detailed scene under the
    default guidance; below about six, high guidance blows out into hard contrast
    instead of detail, and thirty-five (Cosmos' own default) is sharper again for
    a bit over twice the cost. If throughput matters more than background detail,
    a distilled checkpoint is the right answer rather than fewer steps here --
    ``Cosmos3-Nano`` is a base model and is being pushed below its comfortable
    range well before it gets fast."""

    resolution: str = "480"
    """Generation resolution bucket."""

    aspect_ratio: str = "1,1"
    """Generation aspect-ratio bucket. Square suits policies that consume square
    crops; the generated grid must not fall below the policy's input size."""

    mask_guidance: bool = False
    """Send the preserved-foreground mask to Cosmos so it denoises around it.

    Stock ``cosmos-framework`` has no guided-generation support, so this requires a
    checkout that adds it and raises on load otherwise rather than silently
    generating without a mask. Where it is available it is strictly better than
    compositing alone: the model knows what occupies the foreground, so the
    background it produces can agree with it on lighting and contact shadows
    instead of being pasted over blind. Compositing stays useful alongside it --
    guidance preserves approximately, in latent space, while the paste is exact."""

    mask_step_threshold: int = 3
    """Sampler steps over which the masked region is held to the source. Higher
    preserves harder and leaves less room for the background to settle around it."""

    mask_downsample_mode: Literal["max", "area", "trilinear"] = "max"
    """How the pixel mask is reduced to latent resolution. ``max`` keeps a latent
    cell whenever any of its pixels are foreground, which is what protects thin
    structures such as fingers; ``area`` averages and tends to drop them."""

    guidance: float = 3.0
    """Classifier-free guidance on the prompt, matching what Cosmos tunes for every
    transfer hint. At 1.0 guidance is effectively off and the prompt barely
    influences the result -- backgrounds come out washed out and generic. The
    ``image2image`` default of 6.0 is for the mode with no control hint at all, and
    is too strong here. A distilled checkpoint is the exception: those are trained
    to run without guidance."""

    compile: bool = False
    """Compile the transfer network. Costs a warmup per process."""

    fp8: bool = False
    """Quantize the reasoner to fp8, trading load-time memory for setup cost."""

    max_batch: int = 1
    """One frame per call. Stock ``cosmos-framework`` rejects batched transfer
    inference outright (``Batching is not supported for transfer inference``), so
    N randomized environments cost N sequential calls. Raising this requires
    multi-sample control packing to land upstream in Cosmos first."""

    prompts: PromptBankCfg = MISSING
    """Background styles to sample from."""

    def __post_init__(self):
        super().__post_init__()
        if self.num_steps < 1:
            raise ValueError("CosmosBackendCfg.num_steps must be positive")
        if not 0.0 <= self.control_guidance <= 10.0:
            raise ValueError("CosmosBackendCfg.control_guidance must be within [0, 10]")


@configclass
class RemoteCosmosBackendCfg(CosmosBackendCfg):
    """Cosmos generation in worker processes on their own GPUs.

    Two problems at once: the simulator stops competing with a diffusion model for
    a GPU, and several workers generate the environments of one step in parallel --
    the only parallelism available while Cosmos rejects batched transfer inference.

    Same-node only. Image payloads move by CUDA IPC and peer copy, never through
    host memory; crossing machines needs a real transport behind the same class.
    """

    devices: tuple[int, ...] = MISSING
    """GPU indices to run workers on, one process each. These are indices into the
    devices visible to this process, so ``CUDA_VISIBLE_DEVICES`` must cover the
    union of the workers' GPUs and the simulator's rather than a single device."""

    startup_timeout_s: float = 900.0
    """Budget for a worker to import Cosmos and load the checkpoint, which is slow
    the first time and involves a download."""

    request_timeout_s: float = 300.0
    """Budget for one generation. Exceeding it raises rather than hanging the
    rollout, and ``VisualDRCfg.on_error`` decides whether that stops the run."""

    def __post_init__(self):
        super().__post_init__()
        if not self.devices:
            raise ValueError("RemoteCosmosBackendCfg.devices must name at least one GPU")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError(f"RemoteCosmosBackendCfg.devices must be distinct, got {self.devices}")
        if self.max_batch > len(self.devices):
            raise ValueError(
                f"max_batch={self.max_batch} exceeds {len(self.devices)} workers; each worker takes one "
                "frame per call, so the runtime must chunk to the worker count"
            )


@configclass
class VisualDRCfg:
    """Runtime visual DR for one environment.

    Attach to an environment config; the runtime reads it once at construction.
    Leaving ``enabled`` false must cost nothing -- no model load, no extra render
    products, no change to what the policy sees.
    """

    enabled: bool = False

    cameras: dict[str, CameraDRCfg] = {}
    """Per-camera preservation rules, keyed by sensor name. A camera absent from
    this mapping is never randomized."""

    backend: DRBackendCfg = MISSING

    probability: float = 0.5
    """Fraction of consumed observations that get restyled. The remainder pass
    through untouched, so the policy sees both distributions."""

    scope: Literal["per_env", "per_batch"] = "per_env"
    """Whether the restyle decision is drawn per environment or once per batch."""

    style: Literal["per_episode", "per_observation"] = "per_episode"
    """Whether an episode keeps one background style throughout, or redraws every
    observation. Per-episode is the safer default: a style that changes under a
    stationary camera is a cue no real deployment provides."""

    decision_period: int = 1
    """Environment actions between consumed observations, i.e. the action chunk
    length. Frames the policy never reads are not generated."""

    seed: int = 0
    """Base seed. Style selection and the restyle decision derive from this, the
    episode id and the camera name, so they survive replica reassignment."""

    on_error: Literal["raise", "passthrough"] = "raise"
    """Whether a generation failure stops the run or yields the raw frame with a
    recorded reason. ``raise`` while bringing a task up; ``passthrough`` once a
    long run matters more than any single observation."""

    def __post_init__(self):
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("VisualDRCfg.probability must be within [0, 1]")
        if self.decision_period < 1:
            raise ValueError("VisualDRCfg.decision_period must be positive")
        if self.enabled and not self.cameras:
            raise ValueError("Visual DR is enabled but no cameras are configured")
