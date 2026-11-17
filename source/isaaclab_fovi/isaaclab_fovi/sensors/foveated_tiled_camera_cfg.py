# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the FoveatedTiledCamera sensor."""

from __future__ import annotations

from fovi.sensing.projection import CameraCalibration

from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_fovi.sensors.native_grid import validate_radius_norm


@configclass
class FoveatedTiledCameraCfg(CameraCfg):
    """Configuration for a foveated tiled camera sensor.

    Extends :class:`CameraCfg` with CMF-based foveation parameters.
    The sensor generates non-uniform foveated rays from
    :class:`fovi.sensing.coords.SamplingCoords`.  ``height`` and ``width``
    are auto-computed from the number of sampling coordinates.

    The actual camera class (Newton or RTX) is resolved at runtime by the
    :func:`isaaclab_fovi.sensors.FoveatedTiledCamera` factory based on
    ``renderer_cfg.renderer_type``.
    """

    class_type: type | str = "isaaclab_fovi.sensors:FoveatedTiledCamera"

    # -- Foveation parameters (no img_height/img_width/fixation_size) --------

    cmf_a: float = 0.1
    """Normalized CMF parameter relative to the selected field diameter.
    Smaller values give stronger foveation. The public fovi API instead takes
    an angular cmf_a; sampling_geometry performs that conversion once."""

    cmf_a_frac: float | None = None
    """Normalized CMF parameter (= cmf_a / fov).  If set, ``cmf_a`` is
    ignored. Planar sampling uses fov=1; spherical sampling resolves degrees
    from the calibrated retinal window before scaling this fraction."""

    field_geometry: str = "planar"
    camera_calibration: CameraCalibration | None = None
    """CameraModel fields for the physical source image; required for spherical geometry."""

    foveated_res: int = 60
    """Resolution parameter passed to :class:`SamplingCoords`.  For isotropic
    sampling this is the number of radial rings; the total sample count *N*
    depends on ``(fov, cmf_a, foveated_res)``."""

    foveated_style: str = "isotropic"
    """Sampling style passed to :class:`SamplingCoords`.  Supported values
    include ``"isotropic"``, ``"logpolar"``, ``"uniform"``, etc."""

    fov_type: str = "circular"
    """Visual footprint; circular is the default for native warped grids."""

    radius_norm: float = 2.0
    """Warped-Cartesian shell norm: 2.0 for circular shells, infinity for square shells."""

    isotropic_plotting_type: str = "v1like"
    """Plotting-space mapping passed to :class:`SamplingCoords` for isotropic
    sampling. Must match the resolved retinal transform so the camera rays
    and policy-side coordinate grids agree on cortical-space geometry. Supported values:
    ``"v1like"`` (default in :class:`SamplingCoords`), ``"schwartz"``, ``"warp"``.
    Ignored for non-isotropic ``foveated_style``."""

    image_rotation_deg: float = 0.0
    """Rotation (degrees, counter-clockwise) applied to the foveated ray
    directions around the optical axis BEFORE rendering.  Used to undo a
    physical camera mount rotation (e.g. an upside-down right-eye sensor)
    so that the per-ray samples land in the canonical
    :class:`SamplingCoords` order — i.e. sample ``i`` always corresponds to
    visual-field position ``rel_cart_coords[i]`` regardless of how the
    camera prim is oriented in world.  General-purpose alternative to
    post-render ``torch.flip`` (which would shuffle the sample order and
    misalign with retinal scatter and KNN processing).

    Common values:
        * ``0.0``    — no rotation (default; right-side-up sensor).
        * ``180.0``  — flip vertically + horizontally (upside-down sensor).
        * Other angles work in principle but are untested."""

    projection: str = "pinhole"
    """Projection model for mapping normalized coords to ray directions.
    ``"pinhole"`` (rectilinear) or ``"fisheye_equidistant"`` (linear angle
    mapping, suitable for wide FOV)."""

    camera_fov_deg: float | None = None
    """If set, use this as the camera HFOV (in degrees) instead of deriving
    from ``spawn.focal_length`` and ``spawn.horizontal_aperture``.  When this
    is not None, ``spawn.focal_length`` and ``spawn.horizontal_aperture``
    must be left at their defaults (they will be back-computed automatically
    for USD prim compatibility)."""

    fixed_software_fov: bool = False
    """Mask pose-steered rays against a fixed dense pinhole camera frustum."""

    software_fov_image_height: int | None = None
    """Dense camera image height used by the deployment software sampler."""

    software_fov_image_width: int | None = None
    """Dense camera image width used by the deployment software sampler."""

    retinal_window_short_side_fraction: float = 1.0
    """Diameter scale applied to the selected image reference side."""

    retinal_window_reference_side: str = "short"
    """Image side defining the circular retinal diameter: ``short`` or ``long``."""

    # -- Optional retinal pooling (photoreceptor -> RGC two-layer sampling) --

    pool_enabled: bool = False
    """If True, the sensor fires a *highres* (photoreceptor) ray grid and
    pools it down to the regular foveated grid. Default False keeps the single-
    layer behaviour unchanged."""

    pool_res_mult: float = 2
    """Resolution multiplier for the highres photoreceptor ray grid.  The
    highres :class:`SamplingCoords` uses
    ``res = round(pool_res_mult * matched_res)`` with the **same** ``cmf_a``
    as the lowres (output) grid so both layers share the same cortical
    magnification profile.  Fractional values are allowed — e.g.
    ``pool_res_mult = sqrt(2)`` gives ~2× input-to-output sample ratio, since
    total sample count scales with ``res²`` in isotropic sampling."""

    pool_k: int | None = None
    """Pooling neighborhood size. ``_as_grid`` sensors use 2D pooling, where
    ``pool_k`` is the squared kernel side (a perfect square) and None uses the
    squared high/low resolution ratio. Sample-set sensors use KNN, where None
    uses the rounded sample-count ratio."""

    pool_mode: str = "avg"
    """Retinal pooling mode. ``"avg"`` or ``"gaussian"``."""

    pool_sample_cortex: bool | str = "geodesic"
    """Space in which KNN neighbours are selected.  ``True`` = Euclidean on
    cortical manifold, ``False`` = visual field, ``"geodesic"`` = exact
    geodesic cortical distances (slower at init, most faithful)."""

    pool_gauss_sigma: float = 10.0
    """Gaussian sigma for ``pool_mode="gaussian"``.  Unused otherwise."""

    pool_int_mode: str = "max"
    """Pool mode used for outputs whose key name contains ``"segmentation"``
    (``semantic_segmentation``, ``instance_segmentation_fast``,
    ``instance_id_segmentation_fast``).  Averaging segmentation values blurs
    class labels and breaks downstream integer/equality checks, so these
    outputs get a separate pooler with this mode.  Set equal to ``pool_mode``
    to disable this behaviour."""

    # -- RTX-specific parameters (ignored by Newton backend) ----------------

    tiled: bool = True
    """Use a shared tiled RTX render product across environments.

    Non-tiled foveated RTX rendering is not implemented and selecting it
    raises :class:`NotImplementedError`. Ignored by Newton backend.
    """

    ndc_layout: str = "cortical_disc"
    """RTX texture layout: ``cortical_disc`` or warped Cartesian ``native_grid``.

    Row-major packing breaks spatial filtering and is no longer accepted.
    Newton does not use a texture layout.
    """

    exit_texture_size: int | None = None
    """Dimension of the exit (inverse) LUT texture (EW = EH) for RTX rendering.
    If ``None`` (default), computed as ``max(128, 4 * tw)`` where ``tw`` is the
    direction texture dimension — this ensures the octahedral inverse map has
    sufficient angular resolution to distinguish individual foveated samples.
    Ignored by Newton backend."""

    # height / width are set automatically in _initialize_impl before the
    # parent allocates buffers.  We default them to 1 here so the MISSING
    # sentinel from CameraCfg is satisfied at config-validation time.
    width: int = 1
    height: int = 1

    # Enable per-frame pose updates so fixation rotation is applied each step.
    update_latest_camera_pose: bool = True

    def __post_init__(self):
        if self.ndc_layout not in ("cortical_disc", "native_grid"):
            raise ValueError(
                f"Unsupported ndc_layout={self.ndc_layout!r}; use 'cortical_disc' or 'native_grid'. "
                "Row-major packing is no longer supported because it breaks spatial filtering."
            )
        if self.ndc_layout == "native_grid" and self.foveated_style != "warped_cartesian_as_grid":
            raise ValueError("native_grid requires a Cartesian grid foveated_style.")
        if self.foveated_style == "warped_cartesian_as_grid" and self.ndc_layout != "native_grid":
            raise ValueError(f"{self.foveated_style} requires ndc_layout='native_grid'.")
        validate_radius_norm(self.radius_norm, self.foveated_style, self.fov_type)
        # ``spawn`` is a ``PinholeCameraCfg`` / ``FisheyeCameraCfg`` whose
        # ``lock_camera`` field defaults to True.  That writes
        # ``omni:kit:cameraLock=True`` on every spawned prim, which the RTX
        # renderer can interpret as "static camera" and use to skip motion-
        # vector synthesis — breaking DLSS-RR temporal reprojection on dynamic
        # geometry.  Foveated cameras always have a moving fixation pose, so
        # we force the lock off here.
        if self.spawn is not None and getattr(self.spawn, "lock_camera", None):
            self.spawn = self.spawn.replace(lock_camera=False)
