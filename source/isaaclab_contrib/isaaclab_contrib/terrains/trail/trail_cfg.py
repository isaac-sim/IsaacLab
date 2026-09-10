# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Literal

import numpy as np
import trimesh
from shapely.geometry import Polygon

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.utils.configclass import configclass

from . import trail_terrains
from .elements import decoration_functions as decoration
from .elements import object_profiles as profiles
from .elements import roll_functions as roll
from .elements import sweeping_paths as paths
from .elements import terrain_functions as terrain
from .utils import colors, transformations
from .utils.math import sample, sample_sign
from .utils.numpy_arrays import mirror_and_join


@configclass
class ObjectParameters:
    """Parameters in this class are used to parametrize the trail objects, and are varied linearly depending on the
    current terrain level."""

    length: float | tuple[float, float] = 1.0
    """The length of each object, measured in heading direction of the trail.

    [m]
    """

    width: float | tuple[float, float] = 1.0
    """The width of the trail.

    [m]
    """

    params: dict[str, float | tuple[float, float]] = {}
    """Collection of optional parameters which are used to parametrize the trail objects."""


@configclass
class TerrainParameters:
    """Parameters in this class are used to parametrize the trail terrain, and are varied linearly depending on the
    current terrain level."""

    params: dict[str, float | tuple[float, float]] = {}
    """Collection of optional parameters which are used to parametrize the trail terrain."""


@configclass
class WallParameters:
    """Parameters in this class are used to parametrize the walls of terrain.

    Parameters do not change as a function of the terrain level.
    """

    wall_functions: dict[str, float] = {
        "no_wall": 0.2,
        "linear_wall": 0.5,
        "half_cos_wall": 1.0,
        "cos_wall": 1.5,
        "circular_wall": 0.0,
        "gaussian_wall": 0.5,
    }
    """Mapping from wall-shape function name to sampling probability weight.

    A wall function is sampled randomly according to its weight.
    """

    wall_dim: dict[str, float | tuple[float, float]] = {
        "width": (0.2, 0.7),
        "height": (0.1, 0.4),
    }
    """The wall dimensions are specified by these values [m]."""

    wall_direction: dict[str, float] = {"up": 1.0, "down": 0.1, "up-down": 0.1}
    """Mapping from wall direction to sampling probability weight.

    Supported directions are: "up", "down", "up-down".
    A direction is sampled randomly according to its weight.
    """

    num_segments: int | tuple[int, int] = 8
    """Number of segments used to define each side of the walls.

    Default is 8.
    """

    num_segments_floor: int = 8
    """Number of segments used to define the ground floor.

    Default is 4. The minimum is 1, indicating a linear evolution of the ground along the lateral direction of the
    trail.
    """

    @property
    def max_wall_width(self) -> float:
        """Helper function that returns an upper estimate of the trail width."""
        return self.wall_dim["width"][1] if isinstance(self.wall_dim["width"], tuple) else self.wall_dim["width"]


@configclass
class ColorParameters:
    """Parameters in this class are used to color the trail environment.

    Colors are defined in HSV convention and per object/segment.
    """

    color_mesh: bool = True
    """If true, the mesh is colored according to the color plate specified below."""

    hsv: dict[str, float | tuple[float, float]] = colors.TRAIL
    """HSV (or rgb) color range."""

    uniform: bool = False
    """If true, the object/segment will be colored uniformly."""


@configclass
class TrailBaseCfg(SubTerrainBaseCfg):
    function = trail_terrains.mesh_trail_segment
    """Generic function to generate the trail terrains.

    This function does not need to be changed nor configured. Find all configuration parameters below.
    """
    # --- Surroundings ---
    # The following parameters describe the local surroundings of the trail.
    platform_length: float = 10.0
    """Length of the starting and ending platform [m]."""
    thickness: float = 1.0
    """The thickness of the ground [m].

    Make sure this value is not too small, otherwise artifacts may arise as vertices below the ground are extended
    beyond the surface.
    """
    border_width: float = 2.0
    """Border added to both sides of the terrain patch.

    This is useful for visual separation of the terrain patches [m].
    """
    ride_direction: Literal["downhill", "uphill"] = "downhill"
    """The direction in which the trail is traversed.

    Can be downhill (default), or uphill.
    """
    floor_width: float | None = None
    """Width of the floor added to the left and the right of the trail.

    If larger than the terrain width, the values are clipped. None means the floor is extended to the borders. Default
    is None.
    """
    path_res: float = 0.5
    """Any two adjacent points on the trail path are at least this far away.

    [m]
    """
    wp: WallParameters = WallParameters()
    """Parameters defining the walls, i.e., the connection between the trail and the floor."""
    # --- Trail objects ---
    # Trail objects are objects that can be typically encountered on a trail or pump track, such as waves or drops.
    object_function: Callable[[float, float, dict[str, float], TrailBaseCfg], np.ndarray | trimesh.Trimesh] | None = (
        None
    )
    """Function that defines objects placed on the trail.

    It returns one of the following: * ``NumPy array``: a 2D profile which is swept into a 3D mesh
    * ``trimesh.Trimesh``: a 3D mesh which does not require a sweeping path
    * ``None``: indicating that no objects are created along the trail path.
    """
    extrude_trail_objects: bool = False
    """If False, trail objects are represented as additional meshes merged with the trail mesh.

    If True, the trail mesh is re-meshed to approximate the objects. This can reduce the total number of vertices when
    an object is large but contains relatively few vertices.
    """
    sweeping_path: Callable[[float, float, dict[str, float], TrailBaseCfg], np.ndarray] | None = None
    """Function that defines how 2D objects are swept into 3D objects.

    This function is only used if object_function is not None.
    """
    object_transformation: Callable[[float, float, dict[str, float], TrailBaseCfg], np.ndarray] | None = None
    """This function can be used to define a certain pose with respect to the trail path.

    ``None`` means identity.
    """
    trail_under_object: Callable[[float, float, dict[str, float], TrailBaseCfg], np.ndarray] | None = None
    """This sweeping function can be used to define the trail under the object.

    None means the local terrain is flat.
    """
    num_segments: int = 8
    """Curved objects are approximated by this many segments.

    Depending on the object, this number may have a different meaning.
    """
    length_between_objects: float | tuple[float, float] = 0.0
    """Spacing between any two objects in x direction, measured from end to start [m]."""
    length_between_platform_and_object: float | tuple[float, float] = 0.5
    """Spacing between an object and a platform in x direction, measured from end to start [m]."""
    cp0: ObjectParameters = ObjectParameters()
    """Object parameters at curriculum start."""
    cp1: ObjectParameters = ObjectParameters()
    """Object parameters at curriculum end."""
    # --- Trail shape ---
    # The trail shape defines the overall curvature and inclination of the trail.
    terrain_functions: list[Callable[[np.ndarray, dict[str, float]], np.ndarray]] | None = [
        partial(terrain.delta_i_sin_x, dim=0, A=(0.0, 0.1), T=(6.0, 16.0), N=(1, 2)),
        partial(terrain.delta_i_sin_x, dim=1, A=(0.1, 0.3), T=(8.0, 16.0), N=(1, 2)),
        partial(terrain.delta_i_sin_x, dim=2, A=(0.1, 0.1), T=(6.0, 16.0), N=(1, 2)),
        partial(terrain.delta_z_sin_xy, A=(0.0, 0.1), Tx=(6.0, 30.0), Ty=(2.0, 6.0), N=1),
        partial(terrain.delta_z_slope_x, S=(0.0, 0.15)),
        partial(terrain.delta_z_noise, U=(0.0, 0.02), N=1),
    ]
    """Set of functions that specify the raw terrain.

    The functions are applied in the order that they are added to the list. The parameters for these functions are place
    in TerrainParameters. If set to None, the terrain is flat.
    """
    skip_terrain_functions: list[str] | None = None
    """List of terrain functions to skip.

    This can be used to disable certain terrain functions without changing the curriculum parameters.
    """
    # --- Trail roll ---
    # Trail roll is an additional DOF of the trail shape.
    roll_functions: list[Callable[[np.ndarray, dict[str, float]], np.ndarray]] | None = [roll.const]
    """Set of functions that specify the roll (i.e., the angle around the heading direction).

    The functions are applied in the order that they are added to the list. If set to None, the terrain is flat.
    """
    roll0: TerrainParameters = TerrainParameters(params={"A": 0.0})
    """Roll parameters at curriculum start."""
    roll1: TerrainParameters = TerrainParameters(params={"A": (0.0, 0.1)})
    """Roll parameters at curriculum end."""
    # --- Decorations ---
    # decorative elements are trees, stones, roots, etc, which are placed next to the trail.
    distance_between_decorations: float | tuple[float, float] = (1.5, 5.0)
    """Distance between any two decorative objects, sampled per terrain patch.

    [m]
    """
    trail_border_clearance: float | tuple[float, float] = (-0.1, 0.2)
    """Distance between the trail border and the center of the decoration object.

    [m]
    """
    decoration_functions: list[Callable[[dict(str, list[trimesh.Trimesh]), TrailBaseCfg], trimesh.Trimesh]] | None = [
        partial(
            decoration.load_object_mesh,
            object_dist={"evergreen": 0.90, "roots": 0.05, "rocks": 0.05},
        ),
        partial(
            decoration.load_object_mesh,
            object_dist={"summer": 0.9, "roots": 0.05, "rocks": 0.05},
        ),
        partial(
            decoration.load_object_mesh,
            object_dist={"evergreen": 0.2, "winter": 0.7, "roots": 0.05, "rocks": 0.05},
        ),
        partial(
            decoration.load_object_mesh,
            object_dist={
                "evergreen": 0.45,
                "summer": 0.45,
                "roots": 0.05,
                "rocks": 0.05,
            },
        ),
        partial(
            decoration.load_object_mesh,
            object_dist={"evergreen": 0.05, "rocks": 0.95},
        ),
    ]
    """List of functions used to generate decorative objects.

    Sampled per terrain patch according to
    ``decoration_function_weights``. If ``None``, no decorative elements are placed next to the trail.
    """
    decoration_function_weights: list[float] | None = [4.0, 1.0, 1.0, 1.0, 1.0]
    """Sampling weights for ``decoration_functions``.

    Must have the same length. If ``None``, uniform weights are used.
    """
    cut_objects_above: float | None = 2.0
    """Decorative objects are cut above this distance to reduce the number of vertices.

    If ``None``, no cutting is performed. [m]
    """
    convex_approx: bool = False
    """If true, decoration objects are approximated using convex hull to reduce the total number of vertices."""
    rel_decorated_terrains: float = 1.0
    """If decorative objects are used, this fraction of terrain patches will include them.

    Must be a value in [0, 1].
    """
    num_decoration_layers: int = 1
    """Number of layers of added decorative objects.

    The first layer is left/right to the trail, the second layer extends the first layer, etc.
    """
    overhanging_clearance: float = 2.5
    """The trail is guaranteed to have no overhanging objects below this distance.

    [m]
    """
    # --- Coloring ---
    col_trail = ColorParameters(hsv=colors.TRAIL)
    col_trail_object = ColorParameters(hsv=colors.TRAIL)
    col_trail_under_object = ColorParameters(hsv=colors.TRAIL)
    col_floor = ColorParameters(hsv=colors.FLOOR)
    col_start = ColorParameters(hsv=colors.START)
    col_goal = ColorParameters(hsv=colors.GOAL)
    # --- Others ----
    training: bool = True
    """Indicates if terrain is used for training or sim deployment.

    Depending on this flag, objects will be more or less simplified.
    """
    distance_start_to_trail: float = 2.5
    """The starting location of the robots is this far away from the trail start.

    [m]
    """


@configclass
class WavesCfg(TrailBaseCfg):
    """This terrain consists of two platforms on each end, which are connected through a sequence of waves.

    * Each wave consists of a ramp up and ramp down.
    * The curvature of the ramps depends on its length and height.
    """

    def object_function(length: float, width: float, params: dict[str, float], cfg: WavesCfg) -> np.ndarray:
        """Create the 2D profile for a wave object.

        Args:
            length: total object length along the trail [m].
            width: lateral width of the trail [m].
            params: sampled parameters for the object.
            cfg: configuration instance for this terrain.

        Returns:
            A NumPy array describing the 2D profile to be swept into 3D.
        """
        return profiles.wave_profile(
            length=0.5 * length,
            height=params["height"],
            num_segments=cfg.num_segments,
            platform_length=0.0,
            gap=False,
            type="cos",
        )

    def sweeping_path(length: float, width: float, params: dict[str, float], cfg: WavesCfg) -> np.ndarray:
        """Provide the sweeping path used to extrude a 2D profile into 3D.

        Args:
            length: object length [m].
            width: trail width [m].
            params: sampled object parameters.
            cfg: configuration instance for this terrain.

        Returns:
            Path array used for sweeping the profile.
        """
        return paths.linear_path_y(width=width, num_segments=cfg.wp.num_segments_floor)

    # overwrite
    extrude_trail_objects = True
    num_segments = 14


@configclass
class JumpsCfg(TrailBaseCfg):
    """This terrain consists of two platforms on each end, which are connected through a sequence of jumps.

    * Each jump consists of a ramp up, a plateau platform, and ramp down.
    * The curvature of the jump depends on its length, height and the plateau length.
    """

    def object_function(length: float, width: float, params: dict[str, float], cfg: WavesCfg) -> np.ndarray:
        """Create the 2D profile for a jump object.

        Args:
            length: total object length along the trail [m].
            width: lateral width of the trail [m].
            params: sampled parameters for the object.
            cfg: configuration instance for this terrain.

        Returns:
            A NumPy array describing the 2D profile to be swept into 3D.
        """
        plateau_length = sample(params["plateau_proportion"]) * length
        ramp_length = 0.5 * (length - plateau_length)
        return profiles.wave_profile(
            length=ramp_length,
            height=params["height"],
            num_segments=cfg.num_segments,
            platform_length=plateau_length,
            gap=cfg.gap,
            exponent=cfg.exponent,
            type="half-cos",
        )

    def sweeping_path(length: float, width: float, params: dict[str, float], cfg: WavesCfg) -> np.ndarray:
        """Provide the sweeping path for jump objects.

        Args:
            length: object length [m].
            width: trail width [m].
            params: sampled object parameters.
            cfg: configuration instance for this terrain.

        Returns:
            Path array used for sweeping the profile.
        """
        return paths.linear_path_y(width=width, num_segments=cfg.wp.num_segments_floor)

    gap: bool | tuple[bool, bool] = True
    """If true, the plateau is embedded into the ground, forming a gap between the two ramps.

    The gap is parametrized by a cosine shaped profile.
    """
    exponent: int | tuple[int, int] = (4, 10)
    """The larger the exponent is chosen, the sharper the gap becomes.

    Only used if gap is True.
    """
    # overwrite
    extrude_trail_objects = True
    num_segments = 50


@configclass
class StonesCfg(TrailBaseCfg):
    """This terrain consists of two platforms on each end, which are connected through patches of gravel roads."""

    def object_function(length: float, width: float, params: dict[str, float], cfg: StonesCfg) -> np.ndarray:
        """Create the 2D profile for a stone/rock object.

        Args:
            length: object length [m].
            width: trail width [m].
            params: sampled parameters for the object.
            cfg: configuration instance for this terrain.

        Returns:
            A NumPy array describing the 2D profile to be swept into 3D.
        """
        return profiles.root_profile(
            length=length,
            height=params["height"],
            num_segments=cfg.num_segments,
            exponent=cfg.exponent,
        )

    def sweeping_path(length: float, width: float, params: dict[str, float], cfg: StonesCfg) -> np.ndarray:
        """Provide the sweeping path for stone objects.

        Args:
            length: object length [m].
            width: trail width [m].
            params: sampled parameters for the object.
            cfg: configuration instance for this terrain.

        Returns:
            Path array used for sweeping the profile.
        """
        return paths.sinusoidal_exp_z_curve_y(
            width=width,
            stone_length=length,
            stone_height=params["height"],
            distance_between_stones=cfg.length_between_objects,
            exponent=cfg.exponent,
            num_segments=cfg.num_segments,
        )

    exponent: int | tuple[int, int] = (1, 5)
    """The larger the exponent is chosen, the more the stone profile approaches a square.

    A value of 1 means a sinusoidal shape.
    """
    # color trail objects uniformly (overwrite)
    col_trail_object = ColorParameters(hsv=colors.TRAIL, uniform=True)


@configclass
class RootsCfg(TrailBaseCfg):
    """This terrain consists of two platforms on each end, which are connected through a road of roots."""

    def object_function(length: float, width: float, params: dict[str, float], cfg: RootsCfg) -> np.ndarray:
        """Create the 2D profile for a root object.

        Args:
            length: object length [m].
            width: trail width [m].
            params: sampled parameters for the object.
            cfg: configuration instance for this terrain.

        Returns:
            A NumPy array describing the 2D profile to be swept into 3D.
        """
        return profiles.root_profile(
            length=length,
            height=params["height"],
            num_segments=cfg.num_segments,
            exponent=cfg.exponent,
        )

    def sweeping_path(length: float, width: float, params: dict[str, float], cfg: RootsCfg) -> np.ndarray:
        """Provide the sweeping path for root objects.

        Args:
            length: object length [m].
            width: trail width [m].
            params: sampled parameters for the object.
            cfg: configuration instance for this terrain.

        Returns:
            Path array used for sweeping the profile.
        """
        total_width = cfg.wp.max_wall_width + width
        return paths.sinusoidal_xz_curve_y(
            width=2.0 * cfg.wp.max_wall_width + width,
            amplitude_x=params["amplitude_x"],
            amplitude_z=params["amplitude_z"],
            T_xz=params["T_xz"],
            num_segments=int(total_width * cfg.num_segments * 2.0),
        )

    exponent: int | tuple[int, int] = (1, 4)
    """The larger the exponent is chosen, the more the root profile approaches a square.

    A value of 1 means a sinusoidal shape.
    """
    # color trail objects uniformly (overwrite)
    col_trail_object = ColorParameters(hsv=colors.TRAIL, uniform=True)


@configclass
class RampsCfg(TrailBaseCfg):
    """This terrain consists of two platforms on each end, which are connected through a sequence of ramps.

    * The height specifies the height above ground for the drop.
    """

    def object_function(length: float, width: float, params: dict[str, float], cfg: RampsCfg) -> np.ndarray:
        """Create the 2D profile for a ramp object.

        Args:
            length: object length [m].
            width: trail width [m].
            params: sampled parameters for the object.
            cfg: configuration instance for this terrain.

        Returns:
            A NumPy array describing the 2D profile to be swept into 3D.
        """
        return profiles.ramp_profile(
            length=length,
            height=params["height"],
            num_segments=cfg.num_segments,
            elevation=sample(cfg.box_height),
        )

    def sweeping_path(length: float, width: float, params: dict[str, float], cfg: RampsCfg) -> np.ndarray:
        """Provide the sweeping path for ramp objects.

        Args:
            length: object length [m].
            width: trail width [m].
            params: sampled parameters for the object.
            cfg: configuration instance for this terrain.

        Returns:
            Path array used for sweeping the profile.
        """
        return paths.linear_path_y(
            width=sample(cfg.rel_ramp_width) * width,
            num_segments=cfg.wp.num_segments_floor,
        )

    rel_ramp_width: float | tuple[float, float] = (0.7, 0.9)
    """Relative width of the ramp with respect to the trail width."""
    box_height: float | tuple[float, float] = (0.0, 0.1)
    """The slope is placed on a box of this height."""


@configclass
class MultipleRampsCfg(TrailBaseCfg):
    """This terrain consists of two platforms on each end, which are connected through a sequence of parallel ramps.

    * The height specifies the height above ground for the drop.
    """

    def object_function(
        length: float, width: float, params: dict[str, float], cfg: MultipleRampsCfg
    ) -> trimesh.Trimesh:
        num_ramps = sample(cfg.num_ramps)
        lane_width = width / num_ramps
        ramps = []

        rel_heights = np.array([sample(cfg.rel_height) for _ in range(num_ramps)], dtype=float)
        max_idx = int(np.argmax(rel_heights))
        rel_heights[max_idx] = 1.0  # largest ramp

        for i in range(num_ramps):
            ramp_height = rel_heights[i] * params["height"]
            ramp_profile = profiles.ramp_profile(
                length=length,
                height=ramp_height,
                num_segments=cfg.num_segments,
                elevation=sample(cfg.box_height),
            )
            sweeping_path = paths.linear_path_y(width=lane_width, num_segments=cfg.wp.num_segments_floor)
            sweeping_path[:, 1] += -0.5 * width + (i + 0.5) * lane_width
            ramps.append(
                trimesh.creation.sweep_polygon(
                    Polygon(ramp_profile),
                    path=sweeping_path,
                    engine=trail_terrains.ENGINE,
                )
            )

        return ramps[0] if len(ramps) == 1 else trimesh.util.concatenate(ramps)

    def object_transformation(length: float, width: float, params: dict[str, float], cfg: RampsCfg) -> np.ndarray:
        return transformations.translation(vec=[0.0, 0.0, 0.0])

    rel_height: float | tuple[float, float] = (0.5, 1.0)
    """Relative height scaling for each ramp with respect to height."""
    num_ramps: int | tuple[int, int] = (2, 3)
    """Number of parallel ramps placed across the trail width."""
    box_height: float | tuple[float, float] = (0.0, 0.1)
    """The slope is placed on a box of this height."""


@configclass
class SinusoidalCurvesCfg(TrailBaseCfg):
    """This terrain consists of two platforms on each end, which are connected through a sequence of sinusoidal curves
    in xy plane.

    * The length specifies the curve length in x direction of the trail.
    * The sign of the amplitude is randomly sampled.
    """

    def trail_under_object(
        length: float, width: float, params: dict[str, float], cfg: SinusoidalCurvesCfg
    ) -> np.ndarray:
        """Provide a local trail profile beneath objects for sinusoidal-curve terrains.

        Args:
            length: object length [m].
            width: trail width [m].
            params: sampled parameters for the object.
            cfg: configuration instance for this terrain.

        Returns:
            Path array representing the trail under the object.
        """
        return paths.sinusoidal_y_curve_x(
            length=length,
            amplitude=sample_sign() * params["amplitude"],
            num_curves=cfg.num_curves,
            num_segments=cfg.num_segments,
        )

    num_curves: int | tuple[int, int] = 1
    """Number of sinusoidal curves put in sequence.

    1 means sinusoidal curve in (0,2*pi).
    """
    # overwrite
    num_segments = 20


@configclass
class DropCurvesCfg(TrailBaseCfg):
    """This terrain consists of two platforms on each end, which are connected through a sequence of sinusoidal curves
    in xz plane.

    * The length specifies the curve length in x direction of the trail.
    * The sign is set by the sign of height. Positive value means wave up.
    """

    def trail_under_object(length: float, width: float, params: dict[str, float], cfg: DropCurvesCfg) -> np.ndarray:
        return paths.sinusoidal_z_curve_x(
            length=length,
            amplitude=params["amplitude"],
            num_curves=cfg.num_curves,
            num_segments=cfg.num_segments,
        )

    num_curves: int | tuple[int, int] = 1
    """Number of sinusoidal curves put in sequence.

    1 means sinusoidal curve in (0,pi).
    """


@configclass
class WingCurvesCfg(TrailBaseCfg):
    """This terrain consists of two platforms on each end, which are connected through a sequence of three circular
    curves in xy plane.

    * The length specifies the curve length in x direction of the trail. Attention: the requested length may be smaller.
    * The direction of the curve is randomly chosen.
    * Construction of the curve is only possible if (0.5*length - 2*height > 0) and (2*height>max_wing_length).
    """

    def trail_under_object(length: float, width: float, params: dict[str, float], cfg: WingCurvesCfg) -> np.ndarray:
        return paths.circular_xy_curve_x(
            length=length,
            radius=params["radius"] + cfg.wp.max_wall_width + 0.5 * width,
            rel_angle=sample_sign() * params["rel_angle"],
            slope=params["slope"],
            max_wing_length=cfg.max_wing_length,
            num_segments=cfg.num_segments,
        )

    max_wing_length: float | tuple[float, float] = 10.0
    """Largest allowed overshoot in y direction of the trail, measured from center to the most outer part of the wing.

    Make sure to choose this value s.t. the curve does not penetrate into neighboring terrains.
    """


@configclass
class SkinnyCfg(TrailBaseCfg):
    """This terrain consists of two platforms on each end, which are connected through a sequence of bridges.

    * Each bridge is modeled as a beam, which has two turns.
    * The bridge is spawned above a sinusoidal valley.
    """

    def object_function(length: float, width: float, params: dict[str, float], cfg: SkinnyCfg) -> np.ndarray:
        return profiles.box_profile(length=params["beam_width"], height=params["beam_thickness"])

    def sweeping_path(length: float, width: float, params: dict[str, float], cfg: SkinnyCfg) -> np.ndarray:
        return paths.zig_zag_yz_path_x(
            length=length,
            amplitude_y=params["amplitude_y"],
            amplitude_z=params["amplitude_z"],
            rel_knot_point=params["rel_knot_point"],
        )

    def object_transformation(length: float, width: float, params: dict[str, float], cfg: SkinnyCfg) -> np.ndarray:
        offset_from_center_y = sample((-0.5, 0.5)) * abs(width - params["beam_width"])
        vec = [
            0.0,
            0.5 * params["beam_width"] + offset_from_center_y,
            cfg.rel_distance_beam_above_ground * params["beam_thickness"],
        ]
        return transformations.translation(vec=vec)

    def trail_under_object(length: float, width: float, params: dict[str, float], cfg: SkinnyCfg) -> np.ndarray:
        rel_flat_distance = sample(cfg.rel_flat_distance)
        curve = paths.sinusoidal_z_curve_x(
            length=length * (1.0 - rel_flat_distance) * 0.5,
            amplitude=-params["height"],
            num_curves=1,
            num_segments=cfg.num_segments,
        )
        return mirror_and_join(object=curve, dim=0, offset=length * rel_flat_distance)

    rel_flat_distance: float | tuple[float, float] = (0.0, 0.4)
    """This parameter is useful to alter the curvature of the terrain underneath the beam.

    1 indicates 90 degrees edges, 0 indicates a pure cosine wave.
    """
    rel_distance_beam_above_ground: float = 0.0
    """Relative distance of the beam above the ground with respect to the beam thickness.

    For example, 0.5 means the beam is placed half of its thickness above the ground.
    """


@configclass
class Loops360Cfg(TrailBaseCfg):
    """This terrain consists of two platforms on each end, which are connected through a sequence of loops in the xz
    plane."""

    def object_function(length: float, width: float, params: dict[str, float], cfg: Loops360Cfg) -> np.ndarray:
        return profiles.box_profile(length=cfg.loop_thickness, height=params["loop_width"])

    def sweeping_path(length: float, width: float, params: dict[str, float], cfg: Loops360Cfg) -> np.ndarray:
        return paths.loop_curve_x(
            displacement_z=sample_sign() * params["loop_width"],
            radius=0.5 * length,
            angle=2.0 * np.pi,
            num_segments=cfg.num_segments,
        )

    def object_transformation(length: float, width: float, params: dict[str, float], cfg: Loops360Cfg) -> np.ndarray:
        vec = [0.0, 0.5 * params["loop_width"], -cfg.loop_thickness]
        return transformations.translate_and_roll(vec=vec, angle=0.5 * np.pi)

    loop_thickness: float = 0.2
    """The thickness of the loop [m]."""
    # color trail objects uniformly (overwrite)
    col_trail_object = ColorParameters(hsv=colors.TRAIL, uniform=True)


@configclass
class StairsCfg(TrailBaseCfg):
    """This terrain consists of two platforms on each end, which are connected through a sequence of stair cases."""

    def object_function(length: float, width: float, params: dict[str, float], cfg: StairsCfg) -> np.ndarray:
        return profiles.stair_profile(length=length, height=params["height"], step_width=params["step_width"])

    def sweeping_path(length: float, width: float, params: dict[str, float], cfg: StairsCfg) -> np.ndarray:
        return paths.linear_path_y(width=width, num_segments=cfg.wp.num_segments_floor)

    def trail_under_object(length: float, width: float, params: dict[str, float], cfg: StairsCfg) -> np.ndarray:
        return np.vstack([[0.0, 0.0, 0.0], [length, 0.0, params["height"]]])


@configclass
class SlalomCfg(TrailBaseCfg):
    """This terrain consists of two platforms on each end.

    The trail contains cylinders standing upright, forming a slalom like course.
    """

    def object_function(length: float, width: float, params: dict[str, float], cfg: SlalomCfg) -> trimesh.Trimesh:
        # create object
        object = trimesh.creation.cylinder(radius=length, height=1.2 * params["height"], sections=sample(cfg.num_edges))
        # add noise to randomize the pole
        delta_xy = cfg.noise_level * length
        delta_h = cfg.noise_level * params["height"]
        object.vertices[:, 0] += np.random.uniform(-delta_xy, delta_xy, size=object.vertices.shape[0])
        object.vertices[:, 1] += np.random.uniform(-delta_xy, delta_xy, size=object.vertices.shape[0])
        object.vertices[:, 2] += np.random.uniform(-delta_h, delta_h, size=object.vertices.shape[0])
        return object

    def object_transformation(length: float, width: float, params: dict[str, float], cfg: SlalomCfg) -> np.ndarray:
        delta_y = (width + cfg.wp.max_wall_width) * sample(params["rel_dist_from_center"]) * sample_sign()
        return transformations.translation(vec=[0.0, 0.5 * delta_y, 0.4 * params["height"]])

    # redefine the pillar color to be part of the floor, and color it uniformly (overwrite)
    col_trail_object = ColorParameters(hsv=colors.FLOOR, uniform=True)

    num_edges: int | tuple[int, int] = (4, 8)
    """Number of edges of the pose."""
    noise_level: float = 0.2
    """The noise level added to the pole geometry to randomize the shape.

    This is a fraction of the length and height of the pole.
    """
