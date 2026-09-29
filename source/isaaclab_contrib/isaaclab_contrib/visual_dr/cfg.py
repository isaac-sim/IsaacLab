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

    composite_foreground: bool = False
    """Paste source pixels over the generated foreground when explicitly enabled.

    Disabled by default so the generated foreground is returned unchanged.

    Enabling it guarantees pixel identity but also restores the source lighting.
    Mask guidance is a separate latent-space constraint and is not pixel-exact.
    """

    unknown_policy: Literal["preserve", "randomize"] = "preserve"
    """What to do with a semantic ID absent from ``preserve_classes``."""

    boundary_px: int = 0
    """Dilate the preserved region by this many pixels, protecting object edges."""

    boundary_erosion_px: int | dict[str, int] = {}
    """Release an inner boundary band of the preserve mask.

    An integer, e.g. ``8``, erodes the union of all preserved regions, including
    preserved unknown IDs. Erosion occurs after union, so touching classes do not
    acquire gaps. The radius uses a square pixel neighborhood.

    For example, ``{"table": 8}`` allows Cosmos to regenerate eight pixels inside
    the table's outer silhouette, reducing background contamination in its VAE
    boundary cells. Other preserved classes, such as thin robot fingers and small
    objects, keep their full masks. Shared edges between preserved objects are
    untouched. Released pixels are also excluded from foreground compositing.
    Cannot be combined with positive ``boundary_px`` dilation.
    """

    def __post_init__(self):
        if self.boundary_px < 0:
            raise ValueError("CameraDRCfg.boundary_px cannot be negative")
        if isinstance(self.boundary_erosion_px, int):
            radii = [self.boundary_erosion_px]
        elif isinstance(self.boundary_erosion_px, dict):
            for name in self.boundary_erosion_px:
                if name not in self.preserve_classes:
                    raise ValueError(f"boundary_erosion_px class {name!r} must appear in preserve_classes")
            radii = list(self.boundary_erosion_px.values())
        else:
            raise ValueError("boundary_erosion_px must be an integer or a class-to-radius mapping")
        for radius in radii:
            if not isinstance(radius, int) or radius < 0:
                raise ValueError("boundary_erosion_px values must be non-negative integers")
        if self.boundary_px and any(radii):
            raise ValueError("boundary_erosion_px cannot be combined with boundary_px dilation")
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
    """Cosmos image transfer with optional mask-guided denoising.

    Both public and GitLab Cosmos Framework checkouts support this backend. The
    GitLab mask API is used when present; public samplers use an IsaacLab adapter.
    Foreground pixel compositing is controlled separately by ``CameraDRCfg``.
    """

    checkpoint: str = "nvidia/Cosmos3-Nano"
    """Public Nano Hub ID, registered Cosmos name, S3 URI, or local export directory."""

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
    value. These are base-model settings; start with 1.0 for a distilled transfer
    checkpoint, avoiding an extra control-CFG branch. Below 1.0 the control is
    progressively ignored, which widens the visual distribution at the cost of
    agreement with the scene."""

    control_weight: float = 1.0
    """Weight of the control hint within the transfer spec."""

    depth_range_m: tuple[float, float] = (0.1, 6.0)
    """Metres of depth the control map spans. Normalizing against the frame's own
    min and max instead makes the map depend on the far plane: with a distant
    horizon in view the near geometry collapses into a couple of dark values and
    the control signal is effectively lost. Clamp to the range the scene actually
    occupies."""

    num_steps: int = 50
    """Denoising steps. A distilled export must use its fixed schedule (typically 4)."""

    resolution: str = "720"
    """Stock Cosmos resolution tier; 720/4:3 generates 1104x832 pixels."""

    shift: float = 10.0
    """Flow-matching shift: 10 for public 720, 5 for 480; ignored by fixed-step samplers."""

    native_resolution: bool = False
    """Override the bucket with camera dimensions, for a validated custom checkpoint.

    This means camera-sized generation, not the public checkpoint's native tier.
    Leave disabled for public Nano. For the tested 640x480 distilled checkpoint,
    enable with resolution=480. Dimensions must be positive multiples of 32;
    the override is restored after each request, including failed requests.
    """

    aspect_ratio: str = "1,1"
    """Generation aspect-ratio bucket. Square suits policies that consume square
    crops; the generated grid must not fall below the policy's input size."""

    mask_guidance: bool = False
    """Preserve source regions during denoising, independently of final compositing."""

    mask_strength: float = 1.0
    """Per-update latent blend weight in [0, 1].

    Higher strength preserves source appearance, including its illumination.
    Lower strength permits relighting and geometry changes. The same weight over
    50 updates is not equivalent to four updates. Even 1.0 is not pixel-exact.
    """

    mask_step_threshold: int | None = None
    """Last guided update index (inclusive); None guides all denoising updates.

    Thus four-step exports use 3 and the public 50-step recipe uses 49. Explicit
    earlier release permits unconstrained late updates and can alter geometry.
    """

    mask_downsample_mode: Literal["max", "area", "trilinear"] = "area"
    """Reduce the pixel mask to latent cells. Area retains fractional boundaries;
    max protects thin structures but can retain a fringe of source background.
    """

    guidance: float = 3.0
    """Text classifier-free guidance. Use 1.0 with a distilled checkpoint."""

    compile: bool = False
    """Compile the transfer network. Costs a warmup per process."""

    fp8: bool = False
    """Quantize the reasoner to fp8, trading load-time memory for setup cost."""

    max_batch: int = 1
    """Frames per request. Public transfer runs them sequentially; the GitLab
    checkout can pack compatible samples into a model batch.
    """

    prompts: PromptBankCfg = MISSING
    """Background styles to sample from."""

    def __post_init__(self):
        super().__post_init__()
        if self.shift <= 0:
            raise ValueError("CosmosBackendCfg.shift must be positive")
        if self.mask_step_threshold is not None and self.mask_step_threshold < 0:
            raise ValueError("CosmosBackendCfg.mask_step_threshold must be non-negative or None")
        if self.num_steps < 1:
            raise ValueError("CosmosBackendCfg.num_steps must be positive")
        if not 0.0 <= self.control_guidance <= 10.0:
            raise ValueError("CosmosBackendCfg.control_guidance must be within [0, 10]")
        if not 0.0 <= self.mask_strength <= 1.0:
            raise ValueError("CosmosBackendCfg.mask_strength must be within [0, 1]")


@configclass
class RemoteCosmosBackendCfg(CosmosBackendCfg):
    """Cosmos generation in worker processes on their own GPUs.

    Two problems at once: the simulator stops competing with a diffusion model for
    a GPU, and several workers generate the environments of one step in parallel --
    including public Cosmos, which processes transfer samples individually.

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
