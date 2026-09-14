# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton provenance and fixed initialization extensions; no model reconstruction."""

from __future__ import annotations

import newton
import numpy as np

from pxr import Sdf, Usd, UsdPhysics, UsdShade

from isaaclab.assets.physics_properties import UsdProperty
from isaaclab.sim.usd_export import AssetPaths, write_properties

JOINT_PROPERTIES = {
    name: UsdProperty(source, (("", "newton:" + name),), power)
    for name, source, power in (
        ("armature", "joint_armature", 0),
        ("friction", "joint_friction", 0),
        ("limitStiffness", "joint_limit_ke", -1),
        ("limitDamping", "joint_limit_kd", -1),
        ("velocityLimit", "joint_velocity_limit", 1),
    )
}
SHAPE_PROPERTIES = {"newton:contactGap": "shape_gap", "newton:contactMargin": "shape_margin"}
MATERIAL_PROPERTIES = {
    "newton:contactStiffness": "shape_material_ke",
    "newton:contactDamping": "shape_material_kd",
    "newton:contactFrictionGain": "shape_material_kf",
    "newton:contactAdhesion": "shape_material_ka",
    "newton:torsionalFriction": "shape_material_mu_torsional",
    "newton:rollingFriction": "shape_material_mu_rolling",
}


def write_shape_properties(prim: Usd.Prim, values: dict[str, np.ndarray], index: int) -> None:
    """Preserve per-shape contact parameters without mutating a shared material.

    Each collider receives a private physics binding, retaining any original material
    schemas and connections. Missing bindings get explicit effective friction/restitution.
    """
    for target, source in SHAPE_PROPERTIES.items():
        prim.CreateAttribute(target, Sdf.ValueTypeNames.Float).Set(float(values[source][index]))
    stage = prim.GetStage()
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    original, _ = binding.ComputeBoundMaterial("physics")
    path = prim.GetPath().AppendChild("ExportPhysicsMaterial")
    if stage.GetPrimAtPath(path):
        raise RuntimeError(f"Export material path already exists: {path}")
    if original:
        if not Sdf.CopySpec(stage.GetRootLayer(), original.GetPath(), stage.GetRootLayer(), path):
            raise RuntimeError(f"Cannot preserve bound material {original.GetPath()}")
        material = UsdShade.Material(stage.GetPrimAtPath(path))
    else:
        material = UsdShade.Material.Define(stage, path)
    if not material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr().Set(float(values["shape_material_mu"][index]))
        physics.CreateDynamicFrictionAttr().Set(float(values["shape_material_mu"][index]))
        physics.CreateRestitutionAttr().Set(float(values["shape_material_restitution"][index]))
    for target, source in MATERIAL_PROPERTIES.items():
        material.GetPrim().CreateAttribute(target, Sdf.ValueTypeNames.Float).Set(float(values[source][index]))
    binding.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")


class SceneAdapter:
    """Map native model rows and preserve Newton-only fixed physical semantics."""

    def __init__(self, scene):
        self.scene = scene
        self.model = scene.sim.physics_manager.get_model()
        sources = (
            {rule.source for rule in JOINT_PROPERTIES.values()}
            | set(SHAPE_PROPERTIES.values())
            | set(MATERIAL_PROPERTIES.values())
            | {
                "shape_material_mu",
                "shape_material_restitution",
                "joint_target_ke",
                "joint_target_kd",
                "joint_target_mode",
                "joint_qd_start",
                "joint_type",
            }
        )
        self.values = {name: getattr(self.model, name).numpy().copy() for name in sources}

    def _labels(self, view, field, labels):
        layout = view.frequency_layouts[self.model.get_attribute_frequency(field)]
        selected = (
            layout.indices.numpy().tolist()
            if layout.indices is not None
            else range(layout.slice.start, layout.slice.stop)
        )
        return [
            labels[
                layout.offset + world * layout.stride_between_worlds + instance * layout.stride_within_worlds + index
            ]
            for world in range(view.world_count)
            for instance in range(view.count_per_world)
            for index in selected
        ]

    def paths(self, asset) -> AssetPaths:
        bodies = self._labels(asset.root_view, "body_mass", self.model.body_label)
        if asset in self.scene.articulations.values():
            joints = self._labels(asset.root_view, "joint_type", self.model.joint_label)
            dofs = [path for path, count in zip(joints, asset.root_view.joint_dof_counts) for _ in range(count)]
            return AssetPaths(
                [(path, asset.body_names.index(name)) for path, name in zip(bodies, asset.backend_body_names)],
                [(path, asset.joint_names.index(name)) for path, name in zip(dofs, asset.backend_joint_names)],
            )
        return AssetPaths([(path, row) for row, path in enumerate(bodies)], [])

    def write_extensions(self, stage: Usd.Stage) -> None:
        from isaaclab_newton.physics import XPBDSolverCfg

        model = self.model
        cfg = self.scene.sim.cfg.physics
        solver = cfg.solver_cfg
        if not isinstance(solver, XPBDSolverCfg):
            raise NotImplementedError("Fixed Newton scene export currently supports XPBD solver settings only.")
        defaults = XPBDSolverCfg().to_dict()
        unsupported = [
            name
            for name, value in solver.to_dict().items()
            if name not in {"class_type", "solver_type", "iterations"} and value != defaults.get(name)
        ]
        if cfg.num_substeps != 1 or cfg.collision_decimation != 0 or unsupported:
            raise NotImplementedError(
                f"No USD representation for Newton substeps/decimation or solver fields: {unsupported}"
            )
        gains = zip(*(self.values[name] for name in ("joint_target_ke", "joint_target_kd", "joint_target_mode")))
        modes = [(float(kp), float(kd), int(mode)) for kp, kd, mode in gains]
        supported = []
        for force_both in (False, True):
            if all(
                int(newton.JointTargetMode.from_gains(kp, kd, force_both, has_drive=mode != 0)) == mode
                for kp, kd, mode in modes
            ):
                supported.append(force_both)
        if not supported:
            raise NotImplementedError("Newton's USD importer cannot represent mixed per-joint actuator modes.")
        stage.GetRootLayer().customLayerData = {
            **stage.GetRootLayer().customLayerData,
            "isaaclab:newtonImportOptions": {"force_position_velocity_actuation": supported[0]},
        }
        stage.GetRootLayer().customLayerData = {
            **stage.GetRootLayer().customLayerData,
            "isaaclab:newtonDriver": {"solver": "xpbd", "iterations": solver.iterations},
        }
        values = self.values
        starts, kinds = self.values["joint_qd_start"], self.values["joint_type"]
        for index, path in enumerate(model.joint_label):
            joint = stage.GetPrimAtPath(path)
            if joint and kinds[index] in (int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC)):
                write_properties(joint, values, int(starts[index]), JOINT_PROPERTIES)
                axis = "angular" if kinds[index] == int(newton.JointType.REVOLUTE) else "linear"
                # Pinned Newton reads angular initial velocity in rad/s; standard USD uses deg/s.
                for name in ("position", "velocity"):
                    value = joint.GetAttribute(f"state:{axis}:physics:{name}").Get()
                    if value is not None:
                        if name == "velocity" and axis == "angular":
                            value = np.deg2rad(value)
                        joint.CreateAttribute(f"newton:{axis}:{name}", Sdf.ValueTypeNames.Float).Set(float(value))
        for index, path in enumerate(model.shape_label):
            shape = stage.GetPrimAtPath(path)
            if shape and shape.HasAPI(UsdPhysics.CollisionAPI):
                write_shape_properties(shape, values, index)
