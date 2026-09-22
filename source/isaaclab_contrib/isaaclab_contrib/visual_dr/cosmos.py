# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Depth-guided transfer backend, against a stock ``cosmos-framework`` checkout.

No patched checkout and no private checkpoint: the default is the public
``Cosmos3-Nano``. Foreground preservation is the runtime's composite, which is
what lets this run on stock code -- mask-guided denoising lives in files that
would have to be forked.

``cosmos-framework``'s inference API is file-oriented, so this module patches its
two media choke points to serve tensors from memory. Image payloads therefore
never reach disk, but they do make a CPU round trip inside Cosmos itself; a
tensor-native entry point upstream would remove it.
"""

from __future__ import annotations

import gc
import json
import shutil
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .backends import DRFrame, DRRequest

if TYPE_CHECKING:
    from .cfg import CosmosBackendCfg

_FRAMES: dict[str, torch.Tensor] = {}
_OUTPUTS: dict[str, torch.Tensor] = {}
_WANTED: set[str] = set()
_LOCK = threading.Lock()
_PATCHED = False
_ATTENTION_PATCHED = False


def _cudnn_attention_is_usable(device: torch.device) -> bool:
    """Probe whether cuDNN can still serve an attention shape it has not seen.

    Cosmos' only 128-wide-head attention backends are cuDNN and FlashAttention,
    and the Flash wheels are built against a different torch than Isaac Lab pins.
    That leaves cuDNN, which for 128-wide heads answers through a runtime-compiled
    engine -- and once Omniverse Kit has started, compiling a *new* shape fails.
    Shapes here depend on prompt length, so warming a fixed set does not help.

    The probe deliberately uses an odd, unequal-length pair no real call will have
    warmed, so it measures compilation rather than cache hits.
    """
    try:
        query = torch.randn(1, 137, 32, 128, device=device, dtype=torch.bfloat16).transpose(1, 2)
        key = torch.randn(1, 32, 251, 128, device=device, dtype=torch.bfloat16)
        value = torch.randn(1, 32, 251, 128, device=device, dtype=torch.bfloat16)
        torch.ops.aten._scaled_dot_product_cudnn_attention(query, key, value, None, True, 0.0, False, False, scale=None)
        return True
    except Exception:  # noqa: BLE001 - any failure means the backend is unusable
        return False


def _install_attention_fallback(device: torch.device) -> bool:
    """Route Cosmos' attention through torch's Flash kernels when cuDNN cannot cope.

    Torch's own ``_scaled_dot_product_flash_attention`` needs no runtime
    compilation, returns the same logsumexp statistics Cosmos consumes, and
    produces identically shaped outputs. Returns whether the fallback was needed.
    """
    global _ATTENTION_PATCHED
    if _ATTENTION_PATCHED or _cudnn_attention_is_usable(device):
        return _ATTENTION_PATCHED

    from cosmos_framework.model.attention.cudnn import functions

    def flash_sdpa_with_lse(query, key, value, is_causal: bool, scale):
        output, logsumexp = torch.ops.aten._scaled_dot_product_flash_attention(
            query, key, value, 0.0, is_causal, False, scale=scale
        )[:2]
        return output, logsumexp

    functions._cudnn_sdpa_with_lse = flash_sdpa_with_lse
    _ATTENTION_PATCHED = True
    return True


def _release_retained_weights(pointers: set[int]) -> int:
    """Drop the loader's copy of the weights so offload actually frees the GPU.

    Cosmos' checkpoint loader leaves a state dict alive in a frame that outlives
    loading. Its tensors share storage with the model's parameters, so moving the
    parameters to host frees nothing -- the storages stay referenced -- and the
    next activate() allocates a second copy on top. Residency then grows by the
    size of the model on every offload/activate cycle, which defeats the point of
    offloading at all.

    Matching is by storage address rather than key name, because the retained
    dict is keyed by the submodule's own prefix rather than the top-level model's.
    Only dicts holding CUDA tensors that share storage with the weights we just
    moved are cleared, so an unrelated mapping cannot be caught by accident.
    Returns the number of dicts released.
    """
    released = 0
    for obj in gc.get_objects():
        if not isinstance(obj, dict) or not obj:
            continue
        try:
            hits = [
                value
                for value in obj.values()
                if isinstance(value, torch.Tensor) and value.is_cuda and value.data_ptr() in pointers
            ]
        except Exception:  # noqa: BLE001 - exotic mappings must not break offload
            continue
        if hits:
            obj.clear()
            released += 1
    gc.collect()
    return released


def _abs(path) -> str:
    """Match the absolute-path normalization Cosmos' validators apply."""
    return str(Path(path).expanduser().absolute())


def _as_media_frames(tensor: torch.Tensor) -> torch.Tensor:
    """Convert ``[C,H,W]`` float in [0,1] to the ``uint8 [C,T,H,W]`` decode result."""
    frame = tensor[:3].detach().float().clamp(0.0, 1.0)
    if frame.shape[0] == 1:
        frame = frame.repeat(3, 1, 1)
    return (frame * 255.0).round().clamp(0, 255).to(torch.uint8).unsqueeze(1).contiguous()


def _install_patches() -> None:
    """Serve registered paths from memory instead of decoding files.

    Idempotent, and it defers to the original for any path we did not register,
    so an unrelated Cosmos call in the same process still reads from disk.
    """
    global _PATCHED
    if _PATCHED:
        return

    from cosmos_framework.inference import inference as _inference
    from cosmos_framework.inference import transfer as _transfer
    from cosmos_framework.inference import vision as _vision

    original_read = _vision.read_media_frames

    def read_media_frames(path, max_frames):
        key = _abs(path)
        with _LOCK:
            registered = _FRAMES.get(key)
        if registered is None:
            return original_read(path, max_frames)
        return (registered[:, :max_frames] if registered.shape[1] > max_frames else registered), 1.0

    # transfer.py reads the control hint through its own binding, so both need it.
    _vision.read_media_frames = read_media_frames
    _transfer.read_media_frames = read_media_frames

    original_save = _inference.save_img_or_video

    def save_img_or_video(sample, save_fp_wo_ext, *args, **kwargs):
        if isinstance(save_fp_wo_ext, str):
            with _LOCK:
                wanted = save_fp_wo_ext in _WANTED
            if wanted:
                with _LOCK:
                    _OUTPUTS[save_fp_wo_ext] = sample.detach().float().clamp(0.0, 1.0)
                # Cosmos asserts the vision file exists right after this call.
                return original_save(sample, save_fp_wo_ext, *args, **kwargs)
            # Every other save is a control-hint dump: a debug artifact nothing
            # reads back, and one JPEG encode per frame of rollout if left on.
            return None
        return original_save(sample, save_fp_wo_ext, *args, **kwargs)

    _inference.save_img_or_video = save_img_or_video
    _PATCHED = True


class CosmosBackend:
    """Owns one Cosmos pipeline and its GPU residency."""

    def __init__(self, cfg: CosmosBackendCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if self.device.type != "cuda":
            raise ValueError("Cosmos visual DR requires a CUDA device")
        self._pipe = None
        self._setup_args = None
        self._work_dir: Path | None = None
        self._slots: list[dict[str, Path]] = []
        self._active = False

    # -- residency ---------------------------------------------------------

    def activate(self) -> None:
        if self._pipe is None:
            self._build()
        elif not self._active:
            self._pipe.model.to(self.device)
        self._active = True

    def _build(self) -> None:
        from cosmos_framework.inference.args import OmniSetupOverrides
        from PIL import Image

        _install_patches()
        if _install_attention_fallback(self.device):
            print(
                "[visual_dr] cuDNN cannot compile new attention shapes here; "
                "routing Cosmos attention through torch Flash kernels",
                flush=True,
            )
        self._work_dir = Path(tempfile.mkdtemp(prefix="isaaclab_visual_dr_"))

        # One sentinel per concurrent sample: the registry is keyed by path, so a
        # batch sharing one path would collide.
        sentinels = self._work_dir / "sentinels"
        sentinels.mkdir(parents=True, exist_ok=True)
        placeholder = Image.new("RGB", (1, 1))
        for slot in range(self.cfg.max_batch):
            paths = {}
            for name in ("vision", "control"):
                path = sentinels / f"{name}_{slot}.png"
                if not path.exists():
                    placeholder.save(path)
                paths[name] = path
            self._slots.append(paths)

        setup = OmniSetupOverrides.model_construct(
            checkpoint_path=str(self.cfg.checkpoint),
            output_dir=self._work_dir / "out",
            guardrails=False,
            benchmark=False,
            use_torch_compile=bool(self.cfg.compile),
            compiled_region="all",
            # Prompt variants tokenize to different lengths, so static shapes
            # recompile per prompt and eventually hit the recompile limit.
            compile_dynamic=True,
            parallelism_preset="latency",
            dp_shard_size=1,
            dp_replicate_size=1,
            tp_size=1,
            cp_size=1,
            cfgp_size=1,
            max_num_seqs=self.cfg.max_batch,
        )
        self._setup_args = setup.build_setup()
        self._pipe = self._setup_args.get_inference_cls().create(self._setup_args)
        if self.cfg.fp8:
            self._quantize()

    def _quantize(self) -> None:
        """Quantize the reasoner's linear layers, which hold most of the weights."""
        import re

        import torch.nn as nn
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_

        pattern = re.compile("language_model.model.layers")

        def matches(module, fqn: str) -> bool:
            return isinstance(module, nn.Linear) and bool(pattern.search(fqn))

        model = self._pipe.model
        if not any(matches(m, n) for n, m in model.named_modules()):
            raise ValueError("fp8 requested but no reasoner linear layers matched")
        quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()), filter_fn=matches)
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    def offload(self) -> None:
        if not self._active:
            return
        self._active = False
        torch.cuda.synchronize(self.device)
        model = self._pipe.model
        # Record where the weights live before moving them, so the loader's
        # retained copy can be recognised by storage rather than by name.
        pointers = {tensor.data_ptr() for tensor in model.parameters() if tensor.is_cuda}
        pointers |= {tensor.data_ptr() for tensor in model.buffers() if tensor.is_cuda}
        model.to("cpu")
        _release_retained_weights(pointers)
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

    def close(self) -> None:
        self._active = False
        if self._pipe is not None:
            torch.cuda.synchronize(self.device)
            self._pipe = None
            self._setup_args = None
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        if self._work_dir is not None:
            shutil.rmtree(self._work_dir, ignore_errors=True)
            self._work_dir = None
        self._slots = []

    # -- generation --------------------------------------------------------

    @torch.no_grad()
    def generate(self, frame: DRFrame, request: DRRequest) -> torch.Tensor:
        """Restyle a batch of frames in a single Cosmos generate call."""
        if not self._active:
            raise RuntimeError("Activate the Cosmos backend before generating")
        if frame.rgb.device != self.device:
            raise ValueError(f"Frames are on {frame.rgb.device} but the backend owns {self.device}")
        count = frame.num_envs
        if count > len(self._slots):
            raise ValueError(f"Batch of {count} exceeds max_batch={self.cfg.max_batch}; the runtime must chunk")

        # NHWC uint8 -> the [C,H,W] float layout Cosmos' decode path yields.
        rgb = frame.rgb.permute(0, 3, 1, 2).float().div(255.0)
        control = self._control_map(frame)

        registered: list[str] = []
        stubs: list[str] = []
        manifests: list[Path] = []
        out_dirs: list[Path] = []
        try:
            for i in range(count):
                slot = self._slots[i]
                seed = int(request.seeds[i]) & 0x7FFFFFFF
                request_dir = self._work_dir / f"req_{request.camera}_{i}"
                out_dir = request_dir / "out"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_dirs.append(out_dir)

                def register(path, tensor: torch.Tensor) -> str:
                    key = _abs(path)
                    with _LOCK:
                        _FRAMES[key] = _as_media_frames(tensor)
                    registered.append(key)
                    return str(path)

                manifest = {
                    "name": "restyle",
                    "model_mode": "image2image",
                    "prompt": request.prompts[i],
                    "vision_path": register(slot["vision"], rgb[i]),
                    "resolution": str(self.cfg.resolution),
                    "aspect_ratio": str(self.cfg.aspect_ratio),
                    "num_frames": 1,
                    "num_steps": int(self.cfg.num_steps),
                    "seed": seed,
                    "num_outputs": 1,
                    "guidance": float(self.cfg.guidance),
                    "control_guidance": float(self.cfg.control_guidance),
                    self.cfg.control_kind: {
                        "control_path": register(slot["control"], control[i]),
                        "weight": float(self.cfg.control_weight),
                    },
                }
                negative = getattr(self.cfg.prompts, "negative_prompt", None)
                if negative:
                    manifest["negative_prompt"] = negative
                path = request_dir / "manifest.json"
                path.write_text(json.dumps(manifest))
                manifests.append(path)

            overrides = self._setup_args.get_sample_overrides_cls().from_files(
                manifests, overrides=self._setup_args.sample_overrides
            )
            for override, out_dir in zip(overrides, out_dirs):
                override.output_dir = out_dir
                override.download(out_dir / "inputs")
            samples = [o.build_sample(model_config=self._pipe.model_config) for o in overrides]

            for sample, out_dir in zip(samples, out_dirs):
                stub = str((out_dir / f"vision{sample.vision_extension}").with_suffix(""))
                stubs.append(stub)
            with _LOCK:
                for stub in stubs:
                    _OUTPUTS.pop(stub, None)
                    _WANTED.add(stub)

            self._pipe.generate(samples)
            torch.cuda.synchronize(self.device)

            height, width = frame.rgb.shape[1:3]
            restyled = []
            for stub in stubs:
                with _LOCK:
                    captured = _OUTPUTS.pop(stub, None)
                if captured is None:
                    raise RuntimeError(f"Cosmos produced no output for {stub}")
                image = captured[:, 0] if captured.ndim == 4 else captured
                image = image[:3].to(self.device)
                if image.shape[-2:] != (height, width):
                    image = torch.nn.functional.interpolate(
                        image[None], size=(height, width), mode="bilinear", align_corners=False
                    )[0]
                restyled.append(image)
            stacked = torch.stack(restyled)
            return stacked.mul(255.0).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        finally:
            with _LOCK:
                for key in registered:
                    _FRAMES.pop(key, None)
                for stub in stubs:
                    _WANTED.discard(stub)
                    _OUTPUTS.pop(stub, None)
            # Cosmos writes a manifest, an output JSON and a vision file per call;
            # left alone these accumulate one set per generated frame.
            for out_dir in out_dirs:
                shutil.rmtree(out_dir.parent, ignore_errors=True)

    def _control_map(self, frame: DRFrame) -> torch.Tensor:
        """Build the control hint named by ``control_kind`` as a 3-channel image."""
        if self.cfg.control_kind == "depth":
            return self._depth_map(frame)
        if self.cfg.control_kind == "seg":
            return self._segmentation_map(frame)
        raise NotImplementedError(
            f"control_kind={self.cfg.control_kind!r} has no control map here. Cosmos supports edge, blur "
            "and wsm hints, but each needs its own signal; only depth and segmentation are derived today."
        )

    def _depth_map(self, frame: DRFrame) -> torch.Tensor:
        """Turn metric depth into the normalized 3-channel map Cosmos expects."""
        near, far = self.cfg.depth_range_m
        depth = frame.depth.permute(0, 3, 1, 2)
        # Sky and anything past the far plane come back non-finite; treat them as
        # maximally distant rather than letting them poison the normalization.
        depth = torch.nan_to_num(depth, nan=far, posinf=far, neginf=near).clamp(near, far)
        # Near surfaces read as bright, which is the convention the depth hint was
        # trained on.
        normalized = 1.0 - (depth - near) / max(far - near, 1e-6)
        return normalized.repeat(1, 3, 1, 1)

    def _segmentation_map(self, frame: DRFrame) -> torch.Tensor:
        """Colorize semantic IDs into the map a segmentation hint expects.

        The renderer stores each ID as the little-endian bytes of the RGBA colour
        it would have drawn with colorization enabled, so unpacking recovers that
        exact palette -- no arbitrary hashing, and two runs of the same scene agree.
        """
        if frame.segmentation is None:
            raise ValueError(
                "control_kind='seg' needs the segmentation buffer; the camera must emit "
                "'semantic_segmentation' and the observation term must pass it through"
            )
        ids = frame.segmentation.permute(0, 3, 1, 2).to(torch.int64) & 0xFFFFFFFF
        channels = [(ids >> shift) & 0xFF for shift in (0, 8, 16)]
        return torch.cat(channels, dim=1).float().div(255.0)
