# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small tensor utilities shared by Franka Pour reset orchestration."""

import math

import torch


def polar_workspace_cells(
    nominal_position: tuple[float, float, float] | torch.Tensor,
    *,
    radius_range: tuple[float, float] | torch.Tensor,
    azimuth_half_range: float,
    grid_size: int,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return deterministic source-workspace positions on a polar grid [m].

    The polar origin is the fixed robot-base origin in the environment XY plane. When the nominal
    radius is strictly inside the configured range, radial samples are piecewise linear from the
    lower bound through the nominal radius to the upper bound. When the nominal radius is either
    endpoint, a single linear grid spans the range without duplicating a radial sample. Azimuth
    samples are symmetric about the nominal XY bearing. The result is flattened radius-major with
    azimuth varying fastest; sampling rows uniformly therefore samples discrete rings uniformly
    rather than sampling the sector uniformly by area.

    Args:
        nominal_position: Authored nominal source position ``(x, y, z)`` [m]. Its XY bearing
            centers the sector, its radius is an exact grid sample, and its z coordinate is used
            by every returned cell.
        radius_range: Strictly increasing positive lower and upper workspace radii [m]. The
            nominal radius must lie within their closed interval.
        azimuth_half_range: Positive angular half-range about the nominal bearing [rad]. Must be
            less than pi radians.
        grid_size: Odd number of radial samples and azimuth samples. Must be at least three.
        device: Optional output device. Tensor inputs are converted to this device. When omitted,
            the device of ``nominal_position`` is preserved, or CPU is used for tuple input.
        dtype: Optional floating-point output dtype. When omitted, the dtype of a tensor
            ``nominal_position`` is preserved, or the default floating-point dtype is used.

    Returns:
        Source positions [m], shape ``(grid_size**2, 3)``. One row is bit-for-bit equal to
        ``nominal_position`` after any requested device or dtype conversion. It is the center row
        when the nominal radius is strictly interior and lies on the first or last ring when the
        nominal radius anchors the corresponding range endpoint.
    """
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 3 or grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd integer of at least three.")
    if not math.isfinite(azimuth_half_range) or not 0.0 < azimuth_half_range < math.pi:
        raise ValueError("azimuth_half_range must be finite and lie in (0, pi).")

    if isinstance(nominal_position, torch.Tensor):
        resolved_device = nominal_position.device if device is None else torch.device(device)
        resolved_dtype = nominal_position.dtype if dtype is None else dtype
    else:
        resolved_device = torch.device("cpu") if device is None else torch.device(device)
        resolved_dtype = torch.get_default_dtype() if dtype is None else dtype
    if not torch.empty((), dtype=resolved_dtype).is_floating_point():
        raise ValueError("nominal_position and the output must use a floating-point dtype.")

    nominal = torch.as_tensor(nominal_position, device=resolved_device, dtype=resolved_dtype)
    radii = torch.as_tensor(radius_range, device=resolved_device, dtype=resolved_dtype)
    if nominal.shape != (3,):
        raise ValueError(f"nominal_position must contain three coordinates, got shape {tuple(nominal.shape)}.")
    if radii.shape != (2,):
        raise ValueError(f"radius_range must contain two values, got shape {tuple(radii.shape)}.")
    if not bool(torch.isfinite(nominal).all()):
        raise ValueError("nominal_position must contain finite values.")
    if not bool(torch.isfinite(radii).all()):
        raise ValueError("radius_range must contain finite values.")

    nominal_radius = torch.linalg.vector_norm(nominal[:2])
    radius_lower, radius_upper = radii.unbind()
    if not bool((radius_lower > 0.0) & (radius_lower < radius_upper)):
        raise ValueError("radius_range must contain positive values in strictly increasing order.")
    if not bool((radius_lower <= nominal_radius) & (nominal_radius <= radius_upper)):
        raise ValueError("radius_range must contain the nominal XY radius within its closed interval.")

    half_size = grid_size // 2
    nominal_at_lower = bool(nominal_radius == radius_lower)
    nominal_at_upper = bool(nominal_radius == radius_upper)
    if nominal_at_lower or nominal_at_upper:
        radial_samples = torch.linspace(
            float(radius_lower),
            float(radius_upper),
            grid_size,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        nominal_radius_index = 0 if nominal_at_lower else grid_size - 1
    else:
        lower_radii = torch.linspace(
            float(radius_lower),
            float(nominal_radius),
            half_size + 1,
            device=resolved_device,
            dtype=resolved_dtype,
        )[:-1]
        upper_radii = torch.linspace(
            float(nominal_radius),
            float(radius_upper),
            half_size + 1,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        radial_samples = torch.cat((lower_radii, upper_radii))
        nominal_radius_index = half_size
    nominal_azimuth = torch.atan2(nominal[1], nominal[0])
    azimuth_offsets = torch.linspace(
        -azimuth_half_range,
        azimuth_half_range,
        grid_size,
        device=resolved_device,
        dtype=resolved_dtype,
    )
    radius_grid, azimuth_grid = torch.meshgrid(
        radial_samples,
        nominal_azimuth + azimuth_offsets,
        indexing="ij",
    )
    cells = torch.stack(
        (
            radius_grid * torch.cos(azimuth_grid),
            radius_grid * torch.sin(azimuth_grid),
            torch.full_like(radius_grid, nominal[2]),
        ),
        dim=-1,
    ).reshape(-1, 3)
    cells[nominal_radius_index * grid_size + half_size] = nominal
    return cells


def asymmetric_reset_offset_samples(
    lower_bound: tuple[float, float, float] | torch.Tensor,
    upper_bound: tuple[float, float, float] | torch.Tensor,
    sample_count: int,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return deterministic, balanced reset-TCP offset samples [m].

    The first three rows are the exact zero, lower-bound, and upper-bound offsets. Remaining rows
    are low-discrepancy interior samples in complementary pairs. In normalized box coordinates,
    every interior pair sums to one; in physical coordinates, it sums to the element-wise sum of
    ``lower_bound`` and ``upper_bound``. This is balance within an asymmetric box, not a zero-mean
    guarantee.

    Args:
        lower_bound: Inclusive lower offset bound ``(x, y, z)`` [m].
        upper_bound: Inclusive upper offset bound ``(x, y, z)`` [m].
        sample_count: Odd number of samples. Must be at least three.
        device: Optional output device. When omitted, a tensor lower bound supplies the device,
            followed by a tensor upper bound; otherwise CPU is used.
        dtype: Optional floating-point output dtype. When omitted, a tensor lower bound supplies
            the dtype, followed by a tensor upper bound; otherwise the default dtype is used.

    Returns:
        Reset-TCP offsets [m], shape ``(sample_count, 3)``. Row zero is exactly zero, rows one and
        two exactly span the configured bounds, and later rows are complementary interior pairs.
    """
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 3 or sample_count % 2 == 0:
        raise ValueError("sample_count must be an odd integer of at least three.")

    tensor_inputs = tuple(value for value in (lower_bound, upper_bound) if isinstance(value, torch.Tensor))
    if device is None:
        if len(tensor_inputs) == 2 and tensor_inputs[0].device != tensor_inputs[1].device:
            raise ValueError("Tensor offset bounds must use the same device unless device is specified.")
        resolved_device = tensor_inputs[0].device if tensor_inputs else torch.device("cpu")
    else:
        resolved_device = torch.device(device)
    if dtype is None:
        if len(tensor_inputs) == 2 and tensor_inputs[0].dtype != tensor_inputs[1].dtype:
            raise ValueError("Tensor offset bounds must use the same dtype unless dtype is specified.")
        resolved_dtype = tensor_inputs[0].dtype if tensor_inputs else torch.get_default_dtype()
    else:
        resolved_dtype = dtype
    if not torch.empty((), dtype=resolved_dtype).is_floating_point():
        raise ValueError("Offset bounds and the output must use a floating-point dtype.")

    lower = torch.as_tensor(lower_bound, device=resolved_device, dtype=resolved_dtype)
    upper = torch.as_tensor(upper_bound, device=resolved_device, dtype=resolved_dtype)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError(
            "lower_bound and upper_bound must each contain three coordinates; "
            f"got {tuple(lower.shape)} and {tuple(upper.shape)}."
        )
    if not bool(torch.isfinite(lower).all()) or not bool(torch.isfinite(upper).all()):
        raise ValueError("Offset bounds must contain finite values.")
    if bool(torch.any(lower > 0.0)) or bool(torch.any(upper < 0.0)) or bool(torch.any(lower > upper)):
        raise ValueError("Offset bounds must be ordered coordinate-wise and contain zero.")
    if not bool(torch.any(lower < upper)):
        raise ValueError("Offset bounds must have positive width on at least one coordinate.")

    samples = [torch.zeros((1, 3), device=resolved_device, dtype=resolved_dtype), lower[None], upper[None]]
    interior_pair_count = (sample_count - 3) // 2
    if interior_pair_count:
        pair_index = torch.arange(interior_pair_count, device=resolved_device, dtype=resolved_dtype) + 0.5
        # Three-dimensional Kronecker sequence used by the legacy reset bank, restricted to the
        # open unit cube so endpoint coverage remains the responsibility of the explicit bounds.
        multipliers = lower.new_tensor((0.754877666, 0.569840296, 0.438447187))
        unit = torch.frac(pair_index[:, None] * multipliers[None, :])
        epsilon = torch.finfo(resolved_dtype).eps
        unit = torch.clamp(unit, min=epsilon, max=1.0 - epsilon)
        span = upper - lower
        interior = lower + unit * span
        complement = lower + (1.0 - unit) * span
        samples.append(torch.stack((interior, complement), dim=1).reshape(-1, 3))
    return torch.cat(samples, dim=0)


def reset_rotation_vector_samples(
    angle_range: tuple[float, float] | torch.Tensor,
    sample_count: int,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return deterministic reset-orientation perturbations as rotation vectors [rad].

    Rotation axes follow a Fibonacci sphere, avoiding a preferred roll, pitch, or yaw direction.
    Magnitudes cover the configured interval and remain nonzero when its lower bound is positive.
    Curriculum code can therefore scale the vectors continuously from an exactly aligned pose at
    zero extent to a directionally diverse, deliberately misaligned pose at full extent.

    Args:
        angle_range: Inclusive lower and upper rotation magnitudes [rad]. Values must be finite,
            nonnegative, ordered, and no greater than pi radians.
        sample_count: Number of deterministic samples. Must be positive.
        device: Optional output device. A tensor range supplies the default; otherwise CPU is used.
        dtype: Optional floating-point output dtype. A tensor range supplies the default; otherwise
            the default floating-point dtype is used.

    Returns:
        Rotation vectors [rad], shape ``(sample_count, 3)``. For more than one sample, the first and
        last rows attain the lower and upper configured magnitudes exactly.
    """
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer.")

    if isinstance(angle_range, torch.Tensor):
        resolved_device = angle_range.device if device is None else torch.device(device)
        resolved_dtype = angle_range.dtype if dtype is None else dtype
    else:
        resolved_device = torch.device("cpu") if device is None else torch.device(device)
        resolved_dtype = torch.get_default_dtype() if dtype is None else dtype
    if not torch.empty((), dtype=resolved_dtype).is_floating_point():
        raise ValueError("angle_range and the output must use a floating-point dtype.")

    bounds = torch.as_tensor(angle_range, device=resolved_device, dtype=resolved_dtype)
    if bounds.shape != (2,):
        raise ValueError(f"angle_range must contain two values, got shape {tuple(bounds.shape)}.")
    if not bool(torch.isfinite(bounds).all()):
        raise ValueError("angle_range must contain finite values.")
    lower, upper = bounds.unbind()
    if not bool((lower >= 0.0) & (lower <= upper) & (upper <= math.pi)):
        raise ValueError("angle_range must be ordered within [0, pi].")

    sample_ids = torch.arange(sample_count, device=resolved_device, dtype=resolved_dtype)
    unit_height = (sample_ids + 0.5) / sample_count
    axis_z = 1.0 - 2.0 * unit_height
    azimuth = 2.0 * math.pi * torch.frac((sample_ids + 0.5) * 0.6180339887498949)
    axis_radius = torch.sqrt(torch.clamp(1.0 - axis_z.square(), min=0.0))
    axes = torch.stack(
        (axis_radius * torch.cos(azimuth), axis_radius * torch.sin(azimuth), axis_z),
        dim=-1,
    )

    if sample_count == 1:
        angles = ((lower + upper) * 0.5).reshape(1)
    else:
        angle_unit = torch.frac((sample_ids + 0.5) * 0.7548776662466927)
        angles = lower + angle_unit * (upper - lower)
        angles[0] = lower
        angles[-1] = upper
    return axes * angles.unsqueeze(-1)


def boolean_selection_mask(count: int, selected: torch.Tensor) -> torch.Tensor:
    """Return a fixed-size boolean mask selecting the supplied indices."""
    if count < 0:
        raise ValueError(f"Mask length must be non-negative, got {count}.")
    mask = torch.zeros(count, dtype=torch.bool, device=selected.device)
    mask[selected.reshape(-1).long()] = True
    return mask


def balanced_cyclic_permutations(values: torch.Tensor, group_count: int) -> torch.Tensor:
    """Return deterministic cyclic permutations with balanced column-wise value counts."""
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError(f"values must be a nonempty one-dimensional tensor, got shape {tuple(values.shape)}.")
    if group_count < 0:
        raise ValueError(f"group_count must be nonnegative, got {group_count}.")
    group_ids = torch.arange(group_count, device=values.device).unsqueeze(-1)
    value_ids = torch.arange(values.numel(), device=values.device).unsqueeze(0)
    return values[(group_ids + value_ids) % values.numel()]


def scale_randomization_rows_by_extent(
    rows: torch.Tensor,
    extent_levels: tuple[float, ...],
) -> torch.Tensor:
    """Scale one balanced offset bank independently at every curriculum extent.

    Args:
        rows: Full-amplitude offsets whose first dimension is the balanced bank-row dimension.
        extent_levels: Strictly increasing amplitude multipliers in ``[0, 1]`` ending at ``1``.
            A leading zero provides an exact continuity anchor before randomization is introduced.

    Returns:
        Scaled rows with shape ``(len(extent_levels), *rows.shape)``. Every level therefore
        preserves the complete row marginal, and the final level exactly preserves ``rows``.
    """
    if rows.ndim == 0 or rows.shape[0] == 0:
        raise ValueError(f"rows must have a nonempty bank-row dimension, got shape {tuple(rows.shape)}.")
    if not torch.is_floating_point(rows) or not bool(torch.isfinite(rows).all()):
        raise ValueError("rows must be a finite floating-point tensor.")
    if not extent_levels:
        raise ValueError("extent_levels must not be empty.")
    levels = tuple(float(extent) for extent in extent_levels)
    if any(not math.isfinite(extent) or extent < 0.0 or extent > 1.0 for extent in levels):
        raise ValueError("extent_levels must contain finite values in [0, 1].")
    if any(right <= left for left, right in zip(levels, levels[1:], strict=False)):
        raise ValueError("extent_levels must be strictly increasing.")
    if not math.isclose(levels[-1], 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("extent_levels must end at 1.0 to preserve the full randomization domain.")
    levels = (*levels[:-1], 1.0)

    extent = rows.new_tensor(levels).reshape((len(levels),) + (1,) * rows.ndim)
    return rows.unsqueeze(0) * extent


def randomization_extent_index_pools(
    source_positions: torch.Tensor,
    source_yaws: torch.Tensor,
    target_positions: torch.Tensor,
    tcp_jitter: torch.Tensor,
    *,
    source_center: tuple[float, float] | torch.Tensor,
    source_half_range: tuple[float, float] | torch.Tensor,
    source_yaw_half_range: float,
    target_center: tuple[float, float] | torch.Tensor,
    target_half_range: tuple[float, float] | torch.Tensor,
    tcp_jitter_half_range: tuple[float, float, float] | torch.Tensor,
    extent_levels: tuple[float, ...],
    tolerance: float = 1.0e-6,
) -> tuple[torch.Tensor, ...]:
    """Return nested bank indices within combined normalized reset extents.

    Each extent is a Chebyshev radius over source XY position [m], source yaw [rad], target XY
    position [m], and TCP jitter [m], normalized by their configured half-ranges. A zero-range axis
    contributes zero difficulty at its center and excludes rows displaced from that center.
    """
    if source_positions.ndim != 2 or source_positions.shape[-1] < 2:
        raise ValueError(f"source_positions must have shape (N, D) with D >= 2, got {tuple(source_positions.shape)}.")
    if source_yaws.ndim != 1:
        raise ValueError(f"source_yaws must have shape (N,), got {tuple(source_yaws.shape)}.")
    if target_positions.ndim != 2 or target_positions.shape[-1] < 2:
        raise ValueError(f"target_positions must have shape (N, D) with D >= 2, got {tuple(target_positions.shape)}.")
    if tcp_jitter.ndim != 2 or tcp_jitter.shape[-1] != 3:
        raise ValueError(f"tcp_jitter must have shape (N, 3), got {tuple(tcp_jitter.shape)}.")
    row_count = source_positions.shape[0]
    if source_yaws.shape[0] != row_count or target_positions.shape[0] != row_count or tcp_jitter.shape[0] != row_count:
        raise ValueError(
            "source_positions, source_yaws, target_positions, and tcp_jitter must have the same row count."
        )
    if (
        source_yaws.device != source_positions.device
        or target_positions.device != source_positions.device
        or tcp_jitter.device != source_positions.device
    ):
        raise ValueError("source_positions, source_yaws, target_positions, and tcp_jitter must be on the same device.")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    if not extent_levels:
        raise ValueError("extent_levels must not be empty.")
    levels = tuple(float(extent) for extent in extent_levels)
    if any(not math.isfinite(extent) or extent < 0.0 for extent in levels):
        raise ValueError("extent_levels must contain finite nonnegative values.")
    if any(right <= left for left, right in zip(levels, levels[1:], strict=False)):
        raise ValueError("extent_levels must be strictly increasing.")

    def normalized_offsets(
        values: torch.Tensor,
        center_values: tuple[float, ...] | torch.Tensor,
        half_range_values: tuple[float, ...] | torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        center = torch.as_tensor(center_values, device=values.device, dtype=values.dtype)
        half_range = torch.as_tensor(half_range_values, device=values.device, dtype=values.dtype)
        if center.shape != (values.shape[1],) or half_range.shape != (values.shape[1],):
            raise ValueError(f"{name}_center and {name}_half_range must each contain {values.shape[1]} coordinates.")
        if bool(torch.any(~torch.isfinite(values))) or bool(torch.any(~torch.isfinite(center))):
            raise ValueError(f"{name} values and center must be finite.")
        if bool(torch.any(~torch.isfinite(half_range))) or bool(torch.any(half_range < 0.0)):
            raise ValueError(f"{name}_half_range must contain finite nonnegative values.")

        offsets = torch.abs(values - center)
        positive_range = half_range > 0.0
        result = torch.zeros_like(offsets)
        result[:, positive_range] = offsets[:, positive_range] / half_range[positive_range]
        if bool(torch.any(~positive_range)):
            result[:, ~positive_range] = torch.where(
                offsets[:, ~positive_range] <= tolerance,
                torch.zeros_like(offsets[:, ~positive_range]),
                torch.full_like(offsets[:, ~positive_range], torch.inf),
            )
        return result

    normalized_source = normalized_offsets(source_positions[:, :2], source_center, source_half_range, "source")
    normalized_source_yaw = normalized_offsets(
        source_yaws.unsqueeze(-1),
        torch.zeros(1, device=source_yaws.device, dtype=source_yaws.dtype),
        torch.as_tensor((source_yaw_half_range,), device=source_yaws.device, dtype=source_yaws.dtype),
        "source_yaw",
    )
    normalized_target = normalized_offsets(target_positions[:, :2], target_center, target_half_range, "target")
    normalized_tcp_jitter = normalized_offsets(
        tcp_jitter,
        torch.zeros(3, device=tcp_jitter.device, dtype=tcp_jitter.dtype),
        tcp_jitter_half_range,
        "tcp_jitter",
    )
    difficulty = torch.cat(
        (normalized_source, normalized_source_yaw, normalized_target, normalized_tcp_jitter), dim=-1
    ).amax(dim=-1)

    pools = tuple(torch.nonzero(difficulty <= extent + tolerance, as_tuple=False).flatten() for extent in levels)
    if any(pool.numel() == 0 for pool in pools):
        raise ValueError("Every randomization extent level must select at least one bank row.")
    return pools


def sample_index_pools(
    index_pools: tuple[torch.Tensor, ...],
    pool_ids: torch.Tensor,
    *,
    weights: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """Sample one global bank index per row from its selected device-resident pool.

    Args:
        index_pools: Global bank indices for each pool.
        pool_ids: Pool selected by each output row.
        weights: Optional nonnegative row weights aligned with :paramref:`index_pools`. Each pool
            is sampled proportionally to its weights. If omitted, rows are sampled uniformly.

    Returns:
        One sampled global bank index per input row.
    """
    if pool_ids.ndim != 1:
        raise ValueError(f"pool_ids must be one-dimensional, got shape {tuple(pool_ids.shape)}.")
    if weights is not None and len(weights) != len(index_pools):
        raise ValueError("weights must contain one tensor per index pool.")
    result = torch.empty_like(pool_ids, dtype=torch.long)
    for pool_id, index_pool in enumerate(index_pools):
        if index_pool.ndim != 1 or index_pool.numel() == 0:
            raise ValueError("Every index pool must be a nonempty one-dimensional tensor.")
        if index_pool.device != pool_ids.device:
            raise ValueError("index pools and pool_ids must be on the same device.")
        rows = torch.nonzero(pool_ids == pool_id, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        if weights is None:
            slots = torch.randint(index_pool.numel(), (rows.numel(),), device=pool_ids.device)
        else:
            pool_weights = weights[pool_id]
            if pool_weights.shape != index_pool.shape or pool_weights.device != pool_ids.device:
                raise ValueError("Every weight tensor must match its index pool's shape and device.")
            if (
                not torch.is_floating_point(pool_weights)
                or not bool(torch.isfinite(pool_weights).all())
                or bool(torch.any(pool_weights < 0.0))
                or not bool(torch.any(pool_weights > 0.0))
            ):
                raise ValueError("Pool weights must be finite, nonnegative floating-point values with positive sum.")
            slots = torch.multinomial(pool_weights, rows.numel(), replacement=True)
        result[rows] = index_pool[slots]
    return result


def target_xy_behind_source(
    source_xy: torch.Tensor,
    *,
    target_center: tuple[float, float] | torch.Tensor,
    target_half_range: tuple[float, float] | torch.Tensor,
    minimum_y_separation: float | torch.Tensor,
    unit_samples: torch.Tensor,
) -> torch.Tensor:
    """Map unit-square samples to target positions safely behind each source cup [m]."""
    if source_xy.ndim != 2 or source_xy.shape[-1] != 2:
        raise ValueError(f"source_xy must have shape (N, 2), got {tuple(source_xy.shape)}.")
    if unit_samples.shape != source_xy.shape:
        raise ValueError(
            f"unit_samples must match source_xy shape {tuple(source_xy.shape)}, got {tuple(unit_samples.shape)}."
        )
    separation = torch.as_tensor(minimum_y_separation, device=source_xy.device, dtype=source_xy.dtype)
    if separation.ndim == 0:
        separation = separation.expand(source_xy.shape[0])
    elif separation.shape != (source_xy.shape[0],):
        raise ValueError("minimum_y_separation must be a scalar or contain one value per source row.")
    if bool(torch.any(~torch.isfinite(separation))) or bool(torch.any(separation < 0.0)):
        raise ValueError("minimum_y_separation must be finite and nonnegative.")
    if bool(torch.any((unit_samples < 0.0) | (unit_samples > 1.0))):
        raise ValueError("unit_samples must lie in [0, 1].")

    center = torch.as_tensor(target_center, device=source_xy.device, dtype=source_xy.dtype)
    half_range = torch.as_tensor(target_half_range, device=source_xy.device, dtype=source_xy.dtype)
    if center.shape != (2,) or half_range.shape != (2,):
        raise ValueError("target_center and target_half_range must each contain two coordinates.")
    if bool(torch.any(~torch.isfinite(center))) or bool(torch.any(~torch.isfinite(half_range))):
        raise ValueError("Target randomization bounds must be finite.")
    if bool(torch.any(half_range < 0.0)):
        raise ValueError("target_half_range must be nonnegative.")

    lower = center - half_range
    upper = center + half_range
    allowed_y_upper = torch.minimum(
        torch.full_like(source_xy[:, 1], upper[1]),
        source_xy[:, 1] - separation,
    )
    if bool(torch.any(allowed_y_upper < lower[1])):
        raise ValueError("No target y-position satisfies the configured range and source-cup separation.")

    target_x = lower[0] + unit_samples[:, 0] * (upper[0] - lower[0])
    target_y = lower[1] + unit_samples[:, 1] * (allowed_y_upper - lower[1])
    return torch.stack((target_x, target_y), dim=-1)
