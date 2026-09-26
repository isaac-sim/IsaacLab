# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera-image processing shared by observation terms and direct environments.

All camera-observation math lives in this module:

- per-modality normalizers :func:`normalize_rgb`, :func:`normalize_depth`, :func:`normalize_normals`
  and :func:`normalize_segmentation`, which share the ``(images, channel_dim, output_channel_dim)``
  layout arguments;
- :func:`normalize_camera_image`, which picks the normalizer from a camera data-type string;
- :class:`CameraFrameStack`, which turns raw camera frames into policy observations with optional
  channel-first layout and frame stacking;
- the fused Warp kernels behind the uint8 fast path of :func:`normalize_rgb`;
- display helpers used by frame-capture tooling.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import warp as wp

from .buffers import CircularBuffer

_RGB_LIKE_PREFIXES: tuple[str, ...] = ("rgb", "albedo", "simple_shading")
_DEPTH_LIKE_PATTERNS: tuple[str, ...] = ("depth", "distance_to")
_NORMALS_PREFIXES: tuple[str, ...] = ("normals",)


"""
Data-type predicates.
"""


def is_rgb_like(data_type: str) -> bool:
    """Whether ``data_type`` is one of the RGB-like camera outputs (rgb, albedo, simple_shading_*).

    Args:
        data_type: The camera data-type string from ``sensor.data.output`` / ``CameraCfg``.

    Returns:
        True if the data type is normalized by :func:`normalize_rgb`.
    """
    return data_type.startswith(_RGB_LIKE_PREFIXES)


def is_depth_like(data_type: str) -> bool:
    """Whether ``data_type`` is one of the depth-like camera outputs (depth, distance_to_*)."""
    return any(p in data_type for p in _DEPTH_LIKE_PATTERNS)


def is_normals_like(data_type: str) -> bool:
    """Whether ``data_type`` is one of the surface-normals camera outputs."""
    return data_type.startswith(_NORMALS_PREFIXES)


"""
Normalizers.
"""


def normalize_rgb(
    images: torch.Tensor,
    channel_dim: int = -1,
    output_channel_dim: int | None = None,
    mean: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scale a color image to ``[0, 1]`` and subtract a mean.

    Computes ``images / 255 - mean``. 4D ``uint8`` input, including strided views such as the RGB
    channels of an RGBA buffer, runs through a fused Warp kernel that can also convert the layout.
    Other input runs through PyTorch with the same math.

    Args:
        images: Color image. Shape is ``(B, H, W, C)`` or ``(B, C, H, W)``.
        channel_dim: Position of the channel axis in ``images``. Defaults to ``-1``.
        output_channel_dim: Position of the channel axis in the result. Defaults to None,
            which keeps the input layout.
        mean: Constant subtracted after scaling. Defaults to None, which subtracts the per-image,
            per-channel mean over the spatial axes.
        out: Optional pre-allocated, contiguous float32 output for the uint8 fast path. Ignored
            otherwise. Defaults to None.

            .. warning::

                Reusing ``out`` across an environment step aliases consecutive observations: the
                trainer still holds the previous result when the next call overwrites it. Use
                two alternating buffers, or omit ``out``, when results cross ``env.step()``.

    Returns:
        The normalized float32 image. Same object as ``out`` when it is used.

    Raises:
        ValueError: On the uint8 fast path, if a channel axis does not resolve to 1 or 3, or
            ``out`` does not match the result's shape, dtype or device.
    """
    if images.dtype == torch.uint8 and images.ndim == 4:
        return _normalize_uint8(images, channel_dim, output_channel_dim, mean, out)
    resolved_channel_dim = channel_dim % images.ndim
    spatial_dims = tuple(d for d in range(1, images.ndim) if d != resolved_channel_dim)
    images = images.float() / 255.0
    images -= torch.mean(images, dim=spatial_dims, keepdim=True) if mean is None else mean
    return _move_channels(images, channel_dim, output_channel_dim)


def normalize_depth(
    images: torch.Tensor,
    channel_dim: int = -1,
    output_channel_dim: int | None = None,
    invalid_value: float = 0.0,
    max_depth: float | None = None,
    tanh_scale: float | None = None,
) -> torch.Tensor:
    """Replace invalid depth values and optionally rescale depth.

    NaN and infinite values, which renderers emit for pixels without a hit, are replaced with
    ``invalid_value``. Depth is then kept metric, or rescaled with one of:

    - ``max_depth``: clip to ``max_depth`` and divide by it, giving ``[0, 1]`` for non-negative depth.
    - ``tanh_scale``: ``tanh(depth / tanh_scale) - 0.5``, giving ``[-0.5, 0.5)`` for non-negative
      depth without a hard range limit.

    Args:
        images: Depth image [m]. Shape is ``(B, H, W, C)`` or ``(B, C, H, W)``.
        channel_dim: Position of the channel axis in ``images``. Defaults to ``-1``.
        output_channel_dim: Position of the channel axis in the result. Defaults to None,
            which keeps the input layout.
        invalid_value: Value for NaN and infinite pixels [m]. Defaults to 0.
        max_depth: Clipping range for linear rescaling [m]. Defaults to None.
        tanh_scale: Depth scale of the ``tanh`` rescaling [m]. Defaults to None.

    Returns:
        A new float image; metric [m] unless a rescaling is selected.

    Raises:
        ValueError: If both ``max_depth`` and ``tanh_scale`` are set.
    """
    if max_depth is not None and tanh_scale is not None:
        raise ValueError("Set at most one of max_depth and tanh_scale.")
    images = torch.nan_to_num(images, nan=invalid_value, posinf=invalid_value, neginf=invalid_value)
    if max_depth is not None:
        images.clamp_(max=max_depth).div_(max_depth)
    elif tanh_scale is not None:
        images.div_(tanh_scale).tanh_().sub_(0.5)
    return _move_channels(images, channel_dim, output_channel_dim)


def normalize_normals(
    images: torch.Tensor, channel_dim: int = -1, output_channel_dim: int | None = None
) -> torch.Tensor:
    """Map unit surface normals from ``[-1, 1]`` to ``[0, 1]``.

    Args:
        images: Surface normals. Shape is ``(B, H, W, C)`` or ``(B, C, H, W)``.
        channel_dim: Position of the channel axis in ``images``. Defaults to ``-1``.
        output_channel_dim: Position of the channel axis in the result. Defaults to None,
            which keeps the input layout.

    Returns:
        A new float image in ``[0, 1]``.
    """
    return _move_channels((images + 1.0) * 0.5, channel_dim, output_channel_dim)


def normalize_segmentation(
    images: torch.Tensor, channel_dim: int = -1, output_channel_dim: int | None = None
) -> torch.Tensor:
    """Convert a segmentation image to a float observation.

    Colorized (``uint8`` RGBA) segmentation is normalized like color with :func:`normalize_rgb`.
    Label-id segmentation (``int32``) is cast to float32 without rescaling, since ids carry no scale.

    Args:
        images: Segmentation image. Shape is ``(B, H, W, C)`` or ``(B, C, H, W)``.
        channel_dim: Position of the channel axis in ``images``. Defaults to ``-1``.
        output_channel_dim: Position of the channel axis in the result. Defaults to None,
            which keeps the input layout.

    Returns:
        A new float32 image.
    """
    if images.dtype == torch.uint8:
        return normalize_rgb(images, channel_dim, output_channel_dim)
    return _move_channels(images.to(torch.float32, copy=True), channel_dim, output_channel_dim)


def normalize_camera_image(
    images: torch.Tensor,
    data_type: str,
    out: torch.Tensor | None = None,
    channel_dim: int = -1,
    output_channel_dim: int | None = None,
) -> torch.Tensor:
    """Normalize a camera image with the default normalizer for its ``data_type``.

    - :func:`is_rgb_like`: :func:`normalize_rgb` with the per-image mean.
    - ``"semantic_segmentation"``: :func:`normalize_segmentation`.
    - :func:`is_depth_like`: :func:`normalize_depth`, replacing invalid values with zero.
    - :func:`is_normals_like`: :func:`normalize_normals`.
    - Otherwise: ``images`` is returned unchanged, apart from the layout.

    Args:
        images: Camera image. Shape and dtype depend on ``data_type``.
        data_type: The camera data-type string.
        out: Optional pre-allocated float32 output, forwarded to :func:`normalize_rgb` for RGB-like
            and colorized segmentation input. Defaults to None.
        channel_dim: Position of the channel axis in ``images``. Defaults to ``-1``.
        output_channel_dim: Position of the channel axis in the result. Defaults to None,
            which keeps the input layout.

    Returns:
        The normalized image.
    """
    if is_rgb_like(data_type) or (data_type == "semantic_segmentation" and images.dtype == torch.uint8):
        return normalize_rgb(images, channel_dim, output_channel_dim, out=out)
    if data_type == "semantic_segmentation":
        return normalize_segmentation(images, channel_dim, output_channel_dim)
    if is_depth_like(data_type):
        return normalize_depth(images, channel_dim, output_channel_dim)
    if is_normals_like(data_type):
        return normalize_normals(images, channel_dim, output_channel_dim)
    return _move_channels(images, channel_dim, output_channel_dim)


def _move_channels(images: torch.Tensor, channel_dim: int, output_channel_dim: int | None) -> torch.Tensor:
    """Move the channel axis to ``output_channel_dim`` as a contiguous tensor."""
    if output_channel_dim is None:
        return images
    return images.movedim(channel_dim, output_channel_dim).contiguous()


"""
Frame stacking.
"""


class CameraFrameStack:
    """Turn raw camera frames into policy observations, optionally stacking recent frames.

    Each call normalizes one ``(B, H, W, C)`` camera frame, lays it out channel-last or channel-first,
    and, when ``frame_stack > 1``, concatenates the last ``frame_stack`` frames oldest-to-newest along
    the channel axis. After construction or a reset, the history is filled with the current frame.

    ``uint8`` frames are buffered raw and normalized after stacking. This keeps the ring buffer four
    times smaller than float32 and is exact because each frame occupies its own channel slice; it
    requires the normalizer to treat channels independently, as :func:`normalize_rgb` does.

    The returned tensor never aliases the camera or ring buffer, so it outlives the next step.
    """

    def __init__(self, num_envs: int, device: str, frame_stack: int = 1, channel_first: bool = False):
        """Initialize the frame stack.

        Args:
            num_envs: Number of environments.
            device: Device of the ring buffer.
            frame_stack: Number of frames to stack along the channel axis. Defaults to 1.
            channel_first: Whether to return ``(B, C, H, W)`` instead of ``(B, H, W, C)``.
                Defaults to False.

        Raises:
            ValueError: If ``frame_stack`` is less than 1.
        """
        if frame_stack < 1:
            raise ValueError(f"frame_stack must be >= 1, got {frame_stack}.")
        self._channel_dim = 1 if channel_first else -1
        self._buffer: CircularBuffer | None = None
        if frame_stack > 1:
            self._buffer = CircularBuffer(
                max_len=frame_stack, batch_size=num_envs, device=device, stack_dim=self._channel_dim
            )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Clear the frame history of the given environments.

        Args:
            env_ids: Environments to reset. Defaults to None, which resets all environments.
        """
        if self._buffer is not None:
            self._buffer.reset(env_ids)

    def __call__(self, images: torch.Tensor, normalize: Callable[..., torch.Tensor] | None = None) -> torch.Tensor:
        """Process the latest camera frame.

        Args:
            images: Raw camera frame. Shape is ``(B, H, W, C)``; may be a view of the camera buffer.
            normalize: Normalizer called as ``normalize(images, channel_dim=..., output_channel_dim=...)``
                that returns a new tensor, e.g. :func:`normalize_rgb` or a :func:`functools.partial`
                of it. Defaults to None, which keeps the raw values.

        Returns:
            The observation. Shape is ``(B, H, W, K * C)`` or ``(B, K * C, H, W)`` for ``K`` stacked frames.
        """
        if self._buffer is None:
            if normalize is None:
                return images.movedim(-1, self._channel_dim).clone(memory_format=torch.contiguous_format)
            return normalize(images, channel_dim=-1, output_channel_dim=self._channel_dim)

        defer_normalize = normalize is not None and images.dtype == torch.uint8
        if normalize is None or defer_normalize:
            frame = images.movedim(-1, self._channel_dim)
        else:
            frame = normalize(images, channel_dim=-1, output_channel_dim=self._channel_dim)
        self._buffer.append(frame)
        stacked = self._buffer.stacked
        if defer_normalize:
            # no ``out=``: each call allocates, so the trainer's previous observation stays intact
            return normalize(stacked, channel_dim=self._channel_dim, output_channel_dim=None)
        # ``stacked`` views the ring buffer, which the next step overwrites
        return stacked.clone()


"""
Fused uint8 normalization.
"""

# Rows reduced per thread in :func:`_uint8_spatial_mean`. Tuned on L40; robust across
# Ampere/Ada/Hopper at R256.
_UINT8_SUM_TILE_HW: int = 32

# int32 partial-sum scratch keyed by (src.shape, device, channel_dim); typically one entry per run.
_uint8_sum_partials_cache: dict[tuple[tuple[int, ...], str, int], torch.Tensor] = {}


@wp.kernel(enable_backward=False)
def _normalize_uint8_kernel(
    src: wp.array4d(dtype=wp.uint8),
    mean: wp.array2d(dtype=wp.float32),
    out: wp.array4d(dtype=wp.float32),
    src_channel_dim: wp.int32,
    out_channel_dim: wp.int32,
):
    """Compute ``out = src / 255 - mean[b, c]``; launch with ``dim=out.shape`` for coalesced writes.

    Args:
        src: Input image. Shape is ``(B, H, W, C)`` or ``(B, C, H, W)``; may be strided.
        mean: Value subtracted per (batch, channel). Shape is ``(B, C)``.
        out: Output image with the channel axis at ``out_channel_dim``.
        src_channel_dim: Channel axis of ``src``, ``1`` or ``3``.
        out_channel_dim: Channel axis of ``out``, ``1`` or ``3``.
    """
    b, d1, d2, d3 = wp.tid()
    if out_channel_dim == 1:
        c = d1
        h = d2
        w = d3
    else:
        c = d3
        h = d1
        w = d2
    if src_channel_dim == 1:
        value = wp.float32(src[b, c, h, w]) / 255.0
    else:
        value = wp.float32(src[b, h, w, c]) / 255.0
    out[b, d1, d2, d3] = value - mean[b, c]


@wp.kernel(enable_backward=False)
def _spatial_sum_uint8_kernel(
    src: wp.array4d(dtype=wp.uint8),
    partials: wp.array3d(dtype=wp.int32),
    tile_size: wp.int32,
    channel_dim: wp.int32,
):
    """Tiled int32 partial sums over the spatial axes; launch with ``dim=(B, ceil(H / tile_size), C)``.

    Args:
        src: Input image. Shape is ``(B, H, W, C)`` or ``(B, C, H, W)``.
        partials: Output partial sums. Shape is ``(B, NUM_TILES, C)``.
        tile_size: Number of H rows reduced per thread.
        channel_dim: Channel axis of ``src``, ``1`` or ``3``.
    """
    b, tile, c = wp.tid()
    h_start = tile * tile_size
    s = wp.int32(0)
    if channel_dim == 1:
        h_end = wp.min(h_start + tile_size, src.shape[2])
        for i in range(h_start, h_end):
            for j in range(src.shape[3]):
                s += wp.int32(src[b, c, i, j])
    else:
        h_end = wp.min(h_start + tile_size, src.shape[1])
        for i in range(h_start, h_end):
            for j in range(src.shape[2]):
                s += wp.int32(src[b, i, j, c])
    partials[b, tile, c] = s


def _normalize_uint8(
    src: torch.Tensor,
    channel_dim: int,
    output_channel_dim: int | None,
    mean: float | None,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Fused-kernel implementation of :func:`normalize_rgb` for 4D uint8 input."""
    resolved_channel_dim, resolved_output_channel_dim = _resolve_image_channel_dims(channel_dim, output_channel_dim)
    out = _image_output(src, resolved_channel_dim, resolved_output_channel_dim, out)

    if mean is None:
        spatial_dims = tuple(d for d in (1, 2, 3) if d != resolved_channel_dim)
        spatial_size = src.shape[spatial_dims[0]] * src.shape[spatial_dims[1]]
        mean_bc = _uint8_spatial_mean(src, spatial_size * 255.0, resolved_channel_dim)
    else:
        mean_bc = torch.full((src.shape[0], src.shape[resolved_channel_dim]), mean, device=src.device)

    wp.launch(
        kernel=_normalize_uint8_kernel,
        dim=out.shape,
        inputs=[
            wp.from_torch(src, dtype=wp.uint8),
            wp.from_torch(mean_bc, dtype=wp.float32),
            wp.from_torch(out, dtype=wp.float32),
            resolved_channel_dim,
            resolved_output_channel_dim,
        ],
        device=str(src.device),
    )
    return out


def _uint8_spatial_mean(src: torch.Tensor, scale: float, channel_dim: int) -> torch.Tensor:
    """Per-(batch, channel) spatial sum of a uint8 image divided by ``scale``. Shape is ``(B, C)``.

    Per-tile sums stay int32 (overflow-safe up to ~16M values per tile); the final sum is int64.
    """
    if channel_dim == 1:
        b, c, h, _ = src.shape
    else:
        b, h, _, c = src.shape
    device_str = str(src.device)
    cache_key = (src.shape, device_str, channel_dim)
    partials = _uint8_sum_partials_cache.get(cache_key)
    if partials is None:
        num_tiles = (h + _UINT8_SUM_TILE_HW - 1) // _UINT8_SUM_TILE_HW
        # C innermost: adjacent threads read stride-1 along src's trailing dim for BHWC input
        partials = torch.empty((b, num_tiles, c), dtype=torch.int32, device=src.device)
        _uint8_sum_partials_cache[cache_key] = partials

    wp.launch(
        kernel=_spatial_sum_uint8_kernel,
        dim=partials.shape,
        inputs=[
            wp.from_torch(src, dtype=wp.uint8),
            wp.from_torch(partials, dtype=wp.int32),
            _UINT8_SUM_TILE_HW,
            channel_dim,
        ],
        device=device_str,
    )
    return partials.sum(dim=1, dtype=torch.int64).float() / scale


def _resolve_image_channel_dims(channel_dim: int, output_channel_dim: int | None) -> tuple[int, int]:
    """Resolve the input and output channel axes of a 4D image to ``1`` (BCHW) or ``3`` (BHWC)."""
    if output_channel_dim is None:
        output_channel_dim = channel_dim
    resolved_channel_dim = channel_dim + 4 if channel_dim < 0 else channel_dim
    resolved_output_channel_dim = output_channel_dim + 4 if output_channel_dim < 0 else output_channel_dim
    if resolved_channel_dim not in (1, 3) or resolved_output_channel_dim not in (1, 3):
        raise ValueError(
            f"channel_dim and output_channel_dim must resolve to 1 (BCHW) or 3 (BHWC) for 4D input;"
            f" got channel_dim={channel_dim} -> {resolved_channel_dim},"
            f" output_channel_dim={output_channel_dim} -> {resolved_output_channel_dim}"
        )
    return resolved_channel_dim, resolved_output_channel_dim


def _image_output(
    src: torch.Tensor, channel_dim: int, output_channel_dim: int, out: torch.Tensor | None
) -> torch.Tensor:
    """Allocate, or validate, the contiguous float32 output of an image kernel."""
    output_shape = src.movedim(channel_dim, output_channel_dim).shape
    if out is None:
        return torch.empty(output_shape, dtype=torch.float32, device=src.device)
    if out.shape != output_shape or out.dtype != torch.float32 or out.device != src.device:
        raise ValueError(
            f"out shape/dtype/device mismatch: expected {tuple(output_shape)}/float32/{src.device},"
            f" got {tuple(out.shape)}/{out.dtype}/{out.device}"
        )
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    return out


"""
Display helpers.
"""


def normalize_camera_output_for_display(tensor: torch.Tensor, data_type: str) -> torch.Tensor:
    """Convert camera output tensor to [0, 1] float32 for conversion to image."""
    normalized = tensor.float()

    if data_type in ["depth", "distance_to_camera", "distance_to_image_plane"]:
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = normalized.max()
        if max_val > 0:
            normalized = normalized / max_val
    elif data_type in {"albedo"}:
        normalized = normalized[..., :3] / 255.0
    elif data_type in {"normals"}:
        normalized = (normalized + 1.0) * 0.5
    elif data_type in {"motion_vectors"}:
        # Motion vectors are per-pixel (u, v) offsets that can be positive or negative. Clamp to [-1, 1],
        # remap to [0, 1], and pack the two channels into an RGB image (u -> R, v -> G, unused B -> 0)
        # so the result can be composed into a grid and saved as an image.
        uv = normalized[..., :2].clamp(-1.0, 1.0)
        uv = (uv + 1.0) * 0.5
        normalized = torch.cat([uv, torch.zeros_like(uv[..., :1])], dim=-1)
    else:
        normalized = normalized / 255.0

    return normalized


def make_camera_output_grid(images: torch.Tensor) -> torch.Tensor:
    """Make a grid of images from a tensor of shape (B, H, W, C)."""
    from torchvision.utils import make_grid

    return make_grid(torch.swapaxes(images.unsqueeze(1), 1, -1).squeeze(-1), nrow=round(images.shape[0] ** 0.5))
