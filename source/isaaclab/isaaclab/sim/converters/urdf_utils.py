# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for pre-processing URDF files before USD conversion."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np


def merge_fixed_joints(urdf_path: str, output_path: str) -> str:
    """Pre-process a URDF file to merge links connected by fixed joints.

    For each fixed joint, the child link's ``<visual>``, ``<collision>``, and ``<inertial>``
    elements are merged into the parent link with proper transform composition. Any
    downstream joints whose parent was the child link are re-parented to the surviving
    parent link (with their origin transforms composed accordingly).

    Chains of consecutive fixed joints are handled by iterating until no fixed joints
    remain.

    Args:
        urdf_path: Path to the input URDF file.
        output_path: Path to write the modified URDF file.

    Returns:
        The *output_path* that was written to.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # iterate until no fixed joints remain (handles chains)
    while True:
        fixed_joints = [j for j in root.findall("joint") if j.get("type") == "fixed"]
        if not fixed_joints:
            break

        # process the first fixed joint found (order matters for chains)
        joint = fixed_joints[0]
        parent_link_name = joint.find("parent").get("link")
        child_link_name = joint.find("child").get("link")

        T_joint = _parse_origin(joint.find("origin"))

        # locate the corresponding `<link>` elements
        parent_link_elem = _find_link(root, parent_link_name)
        child_link_elem = _find_link(root, child_link_name)

        if parent_link_elem is None or child_link_elem is None:
            # safety guard: drop the joint and continue
            root.remove(joint)
            continue

        # move `<visual>` elements from child to parent (with composed transforms)
        for visual in child_link_elem.findall("visual"):
            _compose_origin(visual, T_joint)
            parent_link_elem.append(visual)

        # move `<collision>` elements from child to parent (with composed transforms)
        for collision in child_link_elem.findall("collision"):
            _compose_origin(collision, T_joint)
            parent_link_elem.append(collision)

        # merge `<inertial>` properties (mass, CoM, inertia tensor)
        _merge_inertial(parent_link_elem, child_link_elem, T_joint)

        # re-parent any joints that reference the child link as their parent
        for other_joint in root.findall("joint"):
            if other_joint is joint:
                continue
            parent_elem = other_joint.find("parent")
            if parent_elem is not None and parent_elem.get("link") == child_link_name:
                parent_elem.set("link", parent_link_name)
                # compose transforms: new_origin = T_joint @ T_other
                T_other = _parse_origin(other_joint.find("origin"))
                _set_origin(other_joint, T_joint @ T_other)

        # remove the fixed joint and the now-empty child link
        root.remove(joint)
        root.remove(child_link_elem)

    tree.write(output_path, xml_declaration=True, encoding="UTF-8")
    return output_path


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------


def _parse_origin(origin_elem: ET.Element | None) -> np.ndarray:
    """Parse an ``<origin>`` element into a 4x4 homogeneous transform matrix.

    Args:
        origin_elem: The ``<origin>`` XML element (may be ``None``).

    Returns:
        A 4x4 numpy array representing the transform.
    """
    if origin_elem is None:
        return np.eye(4)
    xyz = [float(v) for v in origin_elem.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in origin_elem.get("rpy", "0 0 0").split()]
    return _make_transform(xyz, rpy)


def _make_transform(xyz: list[float], rpy: list[float]) -> np.ndarray:
    """Create a 4x4 homogeneous transform from *xyz* translation and *rpy* rotation.

    Args:
        xyz: Translation ``[x, y, z]``.
        rpy: Euler angles ``[roll, pitch, yaw]`` in radians (URDF convention: ``Rz @ Ry @ Rx``).

    Returns:
        A 4x4 numpy array.
    """
    T = np.eye(4)
    T[:3, :3] = _rpy_to_rotation_matrix(rpy)
    T[:3, 3] = xyz
    return T


def _rpy_to_rotation_matrix(rpy: list[float]) -> np.ndarray:
    """Convert roll-pitch-yaw to a 3x3 rotation matrix (``Rz @ Ry @ Rx``).

    Args:
        rpy: Euler angles ``[roll, pitch, yaw]`` in radians.

    Returns:
        A 3x3 rotation matrix.
    """
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _rotation_matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to roll-pitch-yaw.

    Args:
        R: A 3x3 rotation matrix.

    Returns:
        Tuple ``(roll, pitch, yaw)`` in radians.
    """
    sy = -R[2, 0]
    if abs(sy) >= 1.0 - 1e-12:
        # gimbal lock
        pitch = math.copysign(math.pi / 2, sy)
        roll = math.atan2(R[0, 1], R[0, 2])
        yaw = 0.0
    else:
        pitch = math.asin(np.clip(sy, -1.0, 1.0))
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    return (roll, pitch, yaw)


def _set_origin(element: ET.Element, T: np.ndarray) -> None:
    """Set or create the ``<origin>`` sub-element of *element* from a 4x4 transform.

    Args:
        element: The parent XML element (e.g. ``<joint>``, ``<visual>``).
        T: The 4x4 homogeneous transform.
    """
    xyz = T[:3, 3]
    rpy = _rotation_matrix_to_rpy(T[:3, :3])

    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")

    origin.set("xyz", f"{_fmt(xyz[0])} {_fmt(xyz[1])} {_fmt(xyz[2])}")
    origin.set("rpy", f"{_fmt(rpy[0])} {_fmt(rpy[1])} {_fmt(rpy[2])}")


def _compose_origin(element: ET.Element, T_parent: np.ndarray) -> None:
    """Compose *element*'s ``<origin>`` with *T_parent* (``T_parent @ T_element``).

    The composed transform replaces the element's existing origin.

    Args:
        element: An XML element that may contain an ``<origin>`` child.
        T_parent: The parent transform to prepend.
    """
    T_elem = _parse_origin(element.find("origin"))
    _set_origin(element, T_parent @ T_elem)


def _find_link(root: ET.Element, name: str) -> ET.Element | None:
    """Find a ``<link>`` element by its ``name`` attribute.

    Args:
        root: The ``<robot>`` root element.
        name: Link name to search for.

    Returns:
        The matching ``<link>`` element, or ``None``.
    """
    for link in root.findall("link"):
        if link.get("name") == name:
            return link
    return None


def _fmt(v: float) -> str:
    """Format a float for URDF output, suppressing near-zero noise.

    Args:
        v: The value to format.

    Returns:
        A string representation.
    """
    if abs(v) < 1e-12:
        return "0"
    return f"{v:.10g}"


# ---------------------------------------------------------------------------
# Inertial merge
# ---------------------------------------------------------------------------


def _parse_inertia_matrix(inertia_elem: ET.Element | None) -> np.ndarray:
    """Parse an ``<inertia>`` element into a 3x3 symmetric inertia matrix.

    Args:
        inertia_elem: The ``<inertia>`` XML element (may be ``None``).

    Returns:
        A 3x3 numpy array.
    """
    if inertia_elem is None:
        return np.zeros((3, 3))
    ixx = float(inertia_elem.get("ixx", "0"))
    ixy = float(inertia_elem.get("ixy", "0"))
    ixz = float(inertia_elem.get("ixz", "0"))
    iyy = float(inertia_elem.get("iyy", "0"))
    iyz = float(inertia_elem.get("iyz", "0"))
    izz = float(inertia_elem.get("izz", "0"))
    return np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ]
    )


def _merge_inertial(parent_link: ET.Element, child_link: ET.Element, T_joint: np.ndarray) -> None:
    """Merge the child link's inertial properties into the parent link.

    Uses the parallel axis theorem to correctly combine mass, center of mass, and
    inertia tensors when the two bodies are rigidly attached.

    Args:
        parent_link: The parent ``<link>`` element that will absorb the child.
        child_link: The child ``<link>`` element being merged.
        T_joint: The 4x4 transform from parent link frame to child link frame.
    """
    child_inertial = child_link.find("inertial")
    if child_inertial is None:
        return  # nothing to merge

    child_mass_elem = child_inertial.find("mass")
    child_mass = float(child_mass_elem.get("value", "0")) if child_mass_elem is not None else 0.0
    if child_mass == 0.0:
        return  # zero mass — nothing to merge

    # -- child inertial in parent link frame --
    T_child_inertial = _parse_origin(child_inertial.find("origin"))
    T_child_in_parent = T_joint @ T_child_inertial
    R_child = T_child_in_parent[:3, :3]
    child_com_in_parent = T_child_in_parent[:3, 3]

    child_I_local = _parse_inertia_matrix(child_inertial.find("inertia"))
    child_I_in_parent = R_child @ child_I_local @ R_child.T

    # -- parent inertial --
    parent_inertial = parent_link.find("inertial")
    if parent_inertial is not None:
        parent_mass_elem = parent_inertial.find("mass")
        parent_mass = float(parent_mass_elem.get("value", "0")) if parent_mass_elem is not None else 0.0
        T_parent_inertial = _parse_origin(parent_inertial.find("origin"))
        R_parent = T_parent_inertial[:3, :3]
        parent_com = T_parent_inertial[:3, 3]
        parent_I_local = _parse_inertia_matrix(parent_inertial.find("inertia"))
        parent_I_in_link = R_parent @ parent_I_local @ R_parent.T
    else:
        parent_inertial = ET.SubElement(parent_link, "inertial")
        parent_mass = 0.0
        parent_com = np.zeros(3)
        parent_I_in_link = np.zeros((3, 3))

    # -- combined mass and center of mass --
    total_mass = parent_mass + child_mass
    if total_mass == 0.0:
        return
    combined_com = (parent_mass * parent_com + child_mass * child_com_in_parent) / total_mass

    # -- parallel axis theorem: shift each inertia tensor to the combined CoM --
    def _shift_inertia(I_at_com: np.ndarray, mass: float, com: np.ndarray, ref: np.ndarray) -> np.ndarray:
        d = ref - com
        return I_at_com + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    parent_I_shifted = (
        _shift_inertia(parent_I_in_link, parent_mass, parent_com, combined_com) if parent_mass > 0 else parent_I_in_link
    )
    child_I_shifted = _shift_inertia(child_I_in_parent, child_mass, child_com_in_parent, combined_com)

    combined_I = parent_I_shifted + child_I_shifted

    # -- write back to parent <inertial> --
    # origin: combined CoM with identity rotation (tensor is already in link frame)
    origin = parent_inertial.find("origin")
    if origin is None:
        origin = ET.SubElement(parent_inertial, "origin")
    origin.set("xyz", f"{_fmt(combined_com[0])} {_fmt(combined_com[1])} {_fmt(combined_com[2])}")
    origin.set("rpy", "0 0 0")

    # mass
    mass_elem = parent_inertial.find("mass")
    if mass_elem is None:
        mass_elem = ET.SubElement(parent_inertial, "mass")
    mass_elem.set("value", f"{_fmt(total_mass)}")

    # inertia tensor
    inertia_elem = parent_inertial.find("inertia")
    if inertia_elem is None:
        inertia_elem = ET.SubElement(parent_inertial, "inertia")
    inertia_elem.set("ixx", _fmt(combined_I[0, 0]))
    inertia_elem.set("ixy", _fmt(combined_I[0, 1]))
    inertia_elem.set("ixz", _fmt(combined_I[0, 2]))
    inertia_elem.set("iyy", _fmt(combined_I[1, 1]))
    inertia_elem.set("iyz", _fmt(combined_I[1, 2]))
    inertia_elem.set("izz", _fmt(combined_I[2, 2]))
