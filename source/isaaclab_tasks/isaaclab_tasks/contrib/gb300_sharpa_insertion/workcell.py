# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured GB300 port registration and one lightweight, physically active cabinet."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import newton
    import warp as wp

# The payload has its own +90-degree root yaw and 108 mm root lift. These
# composed bounds and the additional -90-degree yaw are measured together.
GB300_PRESENTATION_TRANSLATION_E = (0.5800, 0.6833, -1.89475)
GB300_PRESENTATION_ROTATION_XYZW = (0.0, 0.0, -0.7071067811865476, 0.7071067811865476)
GB300_SIMREADY_COMPOSED_BOUNDS_MIN_M = (-0.9940365173959482, -0.35440000280737877, -0.032516271868500105)
GB300_SIMREADY_COMPOSED_BOUNDS_MAX_M = (0.6283000076413155, 0.30250001051441067, 2.41371728817683)
FLOOR_OFFSET_M = -(GB300_PRESENTATION_TRANSLATION_E[2] + GB300_SIMREADY_COMPOSED_BOUNDS_MIN_M[2])
GB300_CABINET_COLLISION_X_MIN_E = GB300_PRESENTATION_TRANSLATION_E[0] - 0.30012
GB300_CABINET_COLLISION_X_MAX_E = GB300_PRESENTATION_TRANSLATION_E[0] + 0.30000

# SN2201 port face at asset X=354.6743 mm, on two levels 44.45 mm apart.
# The 145.96 mm column separation keeps the already seated end out of the grasp.
SOCKET_POSITION_E, UPPER_SOCKET_POSITION_E = tuple(
    (
        GB300_PRESENTATION_TRANSLATION_E[0] + asset_y,
        GB300_PRESENTATION_TRANSLATION_E[1] - (0.3546743 - 0.01275),
        GB300_PRESENTATION_TRANSLATION_E[2] + asset_z + FLOOR_OFFSET_M,
    )
    for asset_y, asset_z in ((-0.0568248, 2.04887), (-0.2027848, 2.04887 + 0.04445))
)


def head_goal_for_socket(socket_position_e: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return the connector-head center at a registered socket origin [m]."""
    # Preserve floor clearance while the clip passes the keeper at its -0.2 rad limit.
    return (socket_position_e[0] - 0.00018, socket_position_e[1] - 0.0372, socket_position_e[2] - 0.0022525)


def _cabinet_shell_boxes() -> tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...]:
    """Approximate the measured cabinet side posts, rear, roof, base, and service backstop [m]."""
    x_min, x_max = GB300_CABINET_COLLISION_X_MIN_E, GB300_CABINET_COLLISION_X_MAX_E
    front_y = GB300_PRESENTATION_TRANSLATION_E[1] - 0.54530
    rear_y = GB300_PRESENTATION_TRANSLATION_E[1] - GB300_SIMREADY_COMPOSED_BOUNDS_MIN_M[0]
    width, depth = x_max - x_min, rear_y - front_y
    height = GB300_SIMREADY_COMPOSED_BOUNDS_MAX_M[2] - GB300_SIMREADY_COMPOSED_BOUNDS_MIN_M[2]
    x_center, y_center = (x_min + x_max) / 2, (front_y + rear_y) / 2
    post_width, sheet = 0.02968, 0.008
    return (
        ("Left", (x_min + post_width / 2, y_center, height / 2), (post_width, depth, height)),
        ("Right", (x_max - post_width / 2, y_center, height / 2), (post_width, depth, height)),
        ("Rear", (x_center, rear_y - sheet / 2, height / 2), (width - 2 * sheet, sheet, height)),
        ("Bottom", (x_center, y_center, sheet / 2), (width, depth, sheet)),
        ("Top", (x_center, y_center, height - sheet / 2), (width, depth, sheet)),
        ("ServiceBayBackstop", (0.470, 0.430, 0.170 + FLOOR_OFFSET_M), (0.240, sheet, 0.120)),
    )


def _service_face_boxes() -> tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...]:
    """Cover the cabinet front except the two measured RJ45 openings [m]."""
    from .socket_geometry import SOCKET_CONTACT_BOUNDS_M

    x_min, x_max = GB300_CABINET_COLLISION_X_MIN_E, GB300_CABINET_COLLISION_X_MAX_E
    z_min = 0.0
    z_max = GB300_PRESENTATION_TRANSLATION_E[2] + GB300_SIMREADY_COMPOSED_BOUNDS_MAX_M[2] + FLOOR_OFFSET_M
    sockets = sorted((SOCKET_POSITION_E, UPPER_SOCKET_POSITION_E), key=lambda socket: socket[2])
    y_front = sockets[0][1] - 0.01275
    thickness = 0.006
    socket_lower, socket_upper = SOCKET_CONTACT_BOUNDS_M
    boxes = []

    def add(name: str, left: float, right: float, bottom: float, top: float) -> None:
        if right > left and top > bottom:
            boxes.append(
                (
                    name,
                    ((left + right) / 2, y_front + thickness / 2, (bottom + top) / 2),
                    (right - left, thickness, top - bottom),
                )
            )

    lower = z_min
    for index, socket in enumerate(sockets):
        x, _, z = socket
        bottom, top = z + socket_lower[2], z + socket_upper[2]
        add(f"Band{index}", x_min, x_max, lower, bottom)
        add(f"Left{index}", x_min, x + socket_lower[0], bottom, top)
        add(f"Right{index}", x + socket_upper[0], x_max, bottom, top)
        lower = top
    add("BandTop", x_min, x_max, lower, z_max)
    return tuple(boxes)


def _front_header_boxes() -> tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...]:
    """Approximate the CAD header's folded 1.5 mm sheet without filling its rear recess [m]."""
    return (
        ("GB300/Collision/FrontHeader/Face", (0.58, 0.13875, 0.37840 + FLOOR_OFFSET_M), (0.60, 0.0015, 0.04570)),
        ("GB300/Collision/FrontHeader/Bottom", (0.58, 0.14440, 0.35630 + FLOOR_OFFSET_M), (0.60, 0.0098, 0.0015)),
        ("GB300/Collision/FrontHeader/Top", (0.58, 0.14440, 0.40050 + FLOOR_OFFSET_M), (0.60, 0.0098, 0.0015)),
    )


def _tor_bezel_boxes() -> tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...]:
    """Follow the curved bezel's measured leading face with five conservative slices [m]."""
    x_edges = (0.326420, 0.333576, 0.340732, 0.347888, 0.355044, 0.362200)
    front_y = (0.308684, 0.311477, 0.315595, 0.318469, 0.320826)
    rear_y = 0.338506
    return tuple(
        (
            f"GB300/Collision/TorBezel/{index}",
            ((left + right) / 2, (front + rear_y) / 2, 0.170875 + FLOOR_OFFSET_M),
            (right - left, rear_y - front, 0.0885),
        )
        for index, (left, right, front) in enumerate(zip(x_edges[:-1], x_edges[1:], front_y, strict=True))
    )


def add_rack_colliders(builder: newton.ModelBuilder, body: int, env_position: wp.vec3, env_rotation: wp.quat) -> None:
    """Add one simple cabinet shell and two CAD-measured socket SDFs to a rigid fixture."""
    import newton
    import warp as wp

    from .socket_geometry import local_socket_mesh

    env_tf = wp.transform(env_position, env_rotation)
    contact_cfg = newton.ModelBuilder.ShapeConfig(density=0, mu=0.5, ke=1.0e5, gap=0.002, is_visible=False)
    socket_cfg = newton.ModelBuilder.ShapeConfig(density=0, mu=0.8, ke=1.0e5, gap=0.0005, is_visible=False)
    mesh = local_socket_mesh()
    for name, position in (("Lower", SOCKET_POSITION_E), ("Upper", UPPER_SOCKET_POSITION_E)):
        builder.add_shape_mesh(
            body=body,
            xform=wp.transform_multiply(env_tf, wp.transform(position, wp.quat_identity())),
            mesh=mesh,
            cfg=socket_cfg,
            color=(0.08, 0.75, 0.9),
            label=f"{builder.body_label[body]}/GB300/Sockets/{name}",
        )
    boxes = [(f"GB300/Collision/{name}", center, size) for name, center, size in _cabinet_shell_boxes()]
    boxes.extend((*_front_header_boxes(), *_tor_bezel_boxes()))
    boxes.extend((f"GB300/Housing/{name}", center, size) for name, center, size in _service_face_boxes())
    for name, center, size in boxes:
        builder.add_shape_box(
            body=body,
            xform=wp.transform_multiply(env_tf, wp.transform(center, wp.quat_identity())),
            hx=size[0] / 2,
            hy=size[1] / 2,
            hz=size[2] / 2,
            cfg=contact_cfg,
            label=f"{builder.body_label[body]}/{name}",
        )
