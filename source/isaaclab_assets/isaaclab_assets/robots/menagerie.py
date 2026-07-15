# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""File-based patcher for Mujoco Menagerie asset conversions.

The Menagerie source models (https://github.com/google-deepmind/mujoco_menagerie) are
correct and complete for MuJoCo. The USD conversions, however, drop guarantees that
MuJoCo provides implicitly and that other engines need authored explicitly. This module
is the single bridge for those conversion defects.

Rather than editing prims on the composed stage at spawn time, every fix is written into
a **patched copy** of the asset's USD layers on disk, so the change is inspectable and
diff-able against the stock conversion. The workflow is:

1. Resolve the asset directory (``payloads/`` + entry ``.usda``) from
   :obj:`MENAGERIE_ASSET_ROOT` -- a local mirror is copied, the public S3 release is
   downloaded via an anonymous ``ListObjectsV2`` enumeration.
2. Materialize it once under ``~/.cache/isaaclab/menagerie_patched/<asset-relpath>/``.
3. Run :func:`patch_menagerie_asset`, which applies one function per conversion defect.
   Each fix is **detection-first** (it inspects the layer and skips when the fix is
   already present) and tagged with the upstream asset/converter change that makes it
   deletable. After all fixes land, the entry layer is stamped with a
   ``isaaclabMenageriePatchVersion`` marker so subsequent spawns short-circuit.
4. Spawn the stock :class:`~isaaclab.sim.spawners.from_files.from_files_cfg.UsdFileCfg`
   path against the patched entry layer -- no stage authoring happens at spawn.

Faithful source content (base frames, masses, joint limits, naming) is intentionally
NOT touched here -- task and robot configurations adapt to it instead.
"""

import os
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from filelock import FileLock

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from pxr import Usd

MENAGERIE_ASSET_ROOT = os.environ.get("MENAGERIE_ASSET_ROOT", f"{ISAAC_NUCLEUS_DIR}/Samples/Mujoco_Menagerie")
"""Root of the Mujoco Menagerie asset conversions.

Resolves through the Isaac asset root to the public production release (S3), so no
Nucleus authentication is required. Override with the ``MENAGERIE_ASSET_ROOT``
environment variable to point at a local mirror.
"""

_PATCH_CACHE_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "isaaclab", "menagerie_patched")
"""Root of the on-disk cache of patched Menagerie assets."""

_PATCH_MARKER_KEY = "isaaclabMenageriePatchVersion"
"""``customLayerData`` key stamped on a fully patched entry layer."""

_PATCH_VERSION = 4
"""Bump when the set or content of the fixes below changes, to invalidate stale caches."""


"""
Conversion-defect fixes.

Each function operates on one USD layer file of a patched asset copy, is detection-first
(skips when the fix is already present), and prints a ``[menagerie-patch]`` line so a
patch run reads as a checklist of what changed.
"""


def _log_applied(fix: str) -> None:
    print(f"[menagerie-patch] applied: {fix}")


def _log_skipped(fix: str) -> None:
    print(f"[menagerie-patch] skipped: {fix} (already present)")


def _strip_drive_deletes(mujoco_layer: str) -> None:
    """Keep the shared layer's UsdPhysics drives alive in the ``mujoco`` variant.

    UPSTREAM(asset): the converter authors the actuation as ``MjcActuator`` prims and, in
    the mujoco variant, deletes the per-joint drive APIs with
    ``delete apiSchemas = ["PhysicsDriveAPI:angular", "PhysicsJointStateAPI:angular"]``.
    IsaacLab's implicit actuators require those drive APIs to exist. Removing the delete
    entries lets the drives composed from the shared physics layer survive. Delete once
    the converter stops dropping them.
    """
    from pxr import Sdf

    fix = "strip mujoco drive deletes"
    layer = Sdf.Layer.FindOrOpen(mujoco_layer)
    count = 0

    def visit(prim_spec):
        nonlocal count
        if prim_spec.HasInfo("apiSchemas"):
            list_op = prim_spec.GetInfo("apiSchemas")
            if list_op.deletedItems:
                new_op = Sdf.TokenListOp()
                new_op.explicitItems = list_op.explicitItems
                new_op.prependedItems = list_op.prependedItems
                new_op.appendedItems = list_op.appendedItems
                prim_spec.SetInfo("apiSchemas", new_op)
                count += 1
        for child in prim_spec.nameChildren:
            visit(child)

    for root in layer.rootPrims:
        visit(root)
    if count == 0:
        _log_skipped(fix)
        return
    layer.Save()
    _log_applied(f"{fix} ({count} joints)")


def _remove_mjc_actuators(mujoco_layer: str) -> None:
    """Remove the converter's ``MjcActuator`` prims so the drive APIs own actuation.

    UPSTREAM(importer): with :meth:`_strip_drive_deletes` re-enabling the per-joint
    UsdPhysics drives, the ``mujoco`` variant declares actuation TWICE — the native
    ``MjcActuator`` prims (MJCF servo, kp=1) and the drives (task actuator config).
    Newton's USD importer builds MuJoCo actuators from both, so every joint carries a
    second servo whose control target IsaacLab never writes; it idles at ctrl=0 and
    drags the joint toward zero with kp/(kp+kp_drive) of the command authority
    (measured: 32 actuators for 16 joints, uniform 0.75 tracking = 3/(3+1)).
    Removing the ``MjcActuator`` prims leaves exactly one actuation source per joint,
    matching the legacy-asset contract. Delete once the importer deduplicates joints
    that have both an ``MjcActuator`` and an authored drive.
    """
    from pxr import Usd

    fix = "remove MjcActuator prims (double actuation)"
    stage = Usd.Stage.Open(mujoco_layer)
    actuators = [prim for prim in stage.TraverseAll() if prim.GetTypeName() == "MjcActuator"]
    if not actuators:
        _log_skipped(fix)
        return
    for prim in actuators:
        stage.RemovePrim(prim.GetPath())
    stage.GetRootLayer().Save()
    _log_applied(f"{fix} ({len(actuators)} actuator(s))")


def _author_static_friction(physics_layer: str) -> None:
    """Author ``physics:staticFriction`` on physics materials that lack it.

    UPSTREAM(asset): the converter authors only ``physics:dynamicFriction``; the static
    coefficient falls back to 0 and grasped objects slide out of a motionless hand.
    MuJoCo has a single sliding coefficient that plays both roles, so the faithful
    translation copies the dynamic value. Delete once the converter authors it.
    """
    from pxr import Sdf, Usd, UsdPhysics

    fix = "author static friction"
    stage = Usd.Stage.Open(physics_layer)
    materials = [prim for prim in stage.TraverseAll() if prim.HasAPI(UsdPhysics.MaterialAPI)]
    missing = [m for m in materials if not m.GetAttribute("physics:staticFriction").HasAuthoredValue()]
    if not missing:
        _log_skipped(fix)
        return
    for material in missing:
        dynamic_attr = material.GetAttribute("physics:dynamicFriction")
        value = dynamic_attr.Get() if dynamic_attr and dynamic_attr.HasAuthoredValue() else 1.0
        material.CreateAttribute("physics:staticFriction", Sdf.ValueTypeNames.Float).Set(value)
    stage.GetRootLayer().Save()
    _log_applied(f"{fix} ({len(missing)} material(s))")


def _author_filtered_pairs(physics_layer: str, pairs: list[tuple[str, str]]) -> None:
    """Author collision filtered pairs between bodies named in ``pairs``.

    UPSTREAM(asset): the converter splits welded geoms (e.g. fingertips) out of their
    source body, manufacturing collision pairs that cannot exist in MuJoCo, where
    same-body geoms never collide. The durable fix is to not split (or to author these
    filters in the conversion); delete once that lands.
    """
    from pxr import Usd, UsdPhysics

    fix = "author weld-split filtered pairs"
    stage = Usd.Stage.Open(physics_layer)
    by_name: dict[str, Usd.Prim] = {}
    for prim in stage.TraverseAll():
        by_name.setdefault(prim.GetName(), prim)

    authored = 0
    for name_a, name_b in pairs:
        body_a, body_b = by_name.get(name_a), by_name.get(name_b)
        if body_a is None or body_b is None:
            continue
        api = UsdPhysics.FilteredPairsAPI.Apply(body_a)
        rel = api.GetFilteredPairsRel()
        if body_b.GetPath() in rel.GetTargets():
            continue
        rel.AddTarget(body_b.GetPath())
        authored += 1
    if authored == 0:
        _log_skipped(fix)
        return
    stage.GetRootLayer().Save()
    _log_applied(f"{fix} ({authored} pair(s))")


def _author_joint_velocity_limit(physx_layer: str, limit_deg_s: float) -> None:
    """Author ``physxJoint:maxJointVelocity`` [deg/s] on every revolute joint.

    UPSTREAM(asset): the MJCF has no joint velocity limits (MuJoCo bounds motion
    implicitly), so PhysX joint velocities spike and produce NaN states. This is an
    engine requirement rather than source content, so it is authored into the
    engine-specific ``physx`` layer only. Delete once the converter emits a limit.
    """
    from pxr import Sdf, Usd, UsdPhysics

    fix = "author joint velocity limit"
    stage = Usd.Stage.Open(physx_layer)
    joints = [
        prim
        for prim in stage.TraverseAll()
        if prim.GetTypeName() == "PhysicsRevoluteJoint" or prim.IsA(UsdPhysics.RevoluteJoint)
    ]
    missing = [j for j in joints if not j.GetAttribute("physxJoint:maxJointVelocity").HasAuthoredValue()]
    if not missing:
        _log_skipped(fix)
        return
    for joint in missing:
        over = stage.OverridePrim(joint.GetPath())
        over.CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float).Set(limit_deg_s)
    stage.GetRootLayer().Save()
    _log_applied(f"{fix} ({limit_deg_s} deg/s on {len(missing)} joints)")


def _author_joint_armature(mujoco_layer: str, physx_layer: str, armature: float) -> None:
    """Author the motor rotor inertia on every revolute joint, in both physics variants.

    UPSTREAM(model): the Menagerie MJCF authors no ``armature``, but the physical hand's
    motors have rotor inertia, and the stock links are light enough that without it the
    joint-space inertia is near zero. The model is authored for MuJoCo's default 2 ms
    timestep; at the coarser steps engines commonly run, integration of the unconditioned
    dynamics turns contact into noise (Allegro reorient success 0.16 vs 1.0 at 1500
    iterations). Authored as ``mjc:armature`` in the ``mujoco`` layer (consumed by
    Newton's importer) and ``physxJoint:armature`` in the ``physx`` layer. Delete once
    the source model (or the robot spec upstream) authors it.
    """
    from pxr import Sdf, Usd, UsdPhysics

    fix = f"author joint armature ({armature})"

    def joints_of(layer_path):
        stage = Usd.Stage.Open(layer_path)
        return stage, [
            prim
            for prim in stage.TraverseAll()
            if prim.GetTypeName() == "PhysicsRevoluteJoint" or prim.IsA(UsdPhysics.RevoluteJoint)
        ]

    authored = 0
    # ``newton:armature`` is what Isaac Lab's Newton import path reads (its schema
    # resolver list is [newton, physx]; the mjc resolver is not registered).
    # ``mjc:armature`` keeps the mujoco variant faithful for native MuJoCo-USD
    # consumers; ``physxJoint:armature`` covers the physx variant.
    for layer_path, attr_name, sdf_type in (
        (mujoco_layer, "newton:armature", Sdf.ValueTypeNames.Float),
        (mujoco_layer, "mjc:armature", Sdf.ValueTypeNames.Double),
        (physx_layer, "physxJoint:armature", Sdf.ValueTypeNames.Float),
    ):
        stage, joints = joints_of(layer_path)
        missing = [j for j in joints if not j.GetAttribute(attr_name).HasAuthoredValue()]
        for joint in missing:
            over = stage.OverridePrim(joint.GetPath())
            over.CreateAttribute(attr_name, sdf_type).Set(armature)
        if missing:
            stage.GetRootLayer().Save()
            authored += len(missing)
    if authored == 0:
        _log_skipped(fix)
        return
    _log_applied(f"{fix} ({authored} joint attribute(s) across variants)")


def _author_shadow_fixed_tendons(physx_layer: str) -> None:
    """Author the PhysX fixed tendons that the Menagerie ``physx`` layer omits.

    UPSTREAM(asset): the MuJoCo USD Converter does not translate MJCF tendon couplings
    into PhysX tendon schemas, so the four distal finger joints are left uncoupled and
    undriven. This replicates the legacy ``ShadowHand`` asset's distal-middle couplings:
    per finger, tendon length ``-0.00805 * theta_middle + 0.00705 * theta_distal``
    constrained to ``+/-0.001`` around rest length 0, so the distal joint tracks the
    middle joint. Authored into the ``physx`` layer only. Delete once the converter
    emits the tendons.
    """
    from pxr import Sdf, Usd

    fix = "author shadow fixed tendons"
    stage = Usd.Stage.Open(physx_layer)
    by_name: dict[str, Usd.Prim] = {}
    for prim in stage.TraverseAll():
        name = prim.GetName()
        if name.startswith("rh_"):
            by_name.setdefault(name, prim)

    probe = by_name.get("rh_FFJ2")
    if probe is not None and probe.GetAttribute("physxTendon:rh_T_FFJ1c:gearing").HasAuthoredValue():
        _log_skipped(fix)
        return

    for finger in ("FF", "MF", "RF", "LF"):
        tendon = f"rh_T_{finger}J1c"
        root_joint = stage.OverridePrim(by_name[f"rh_{finger}J2"].GetPath())
        axis_joint = stage.OverridePrim(by_name[f"rh_{finger}J1"].GetPath())
        root_joint.AddAppliedSchema(f"PhysxTendonAxisRootAPI:{tendon}")
        for attr, type_name, value in (
            # All gains zero: parity with the legacy asset's RUNTIME values (its configured
            # FixedTendonPropertiesCfg never lands due to the instanceable-prim issue, so the
            # baseline trains with zero-gain tendons). A stiff coupling (limitStiffness 30)
            # drags coupled joints to their limits against the weak position drives.
            ("limitStiffness", Sdf.ValueTypeNames.Float, 0.0),
            ("damping", Sdf.ValueTypeNames.Float, 0.0),
            ("stiffness", Sdf.ValueTypeNames.Float, 0.0),
            ("restLength", Sdf.ValueTypeNames.Float, 0.0),
            ("lowerLimit", Sdf.ValueTypeNames.Float, -0.001),
            ("upperLimit", Sdf.ValueTypeNames.Float, 0.001),
            ("gearing", Sdf.ValueTypeNames.FloatArray, [-0.00805]),
        ):
            root_joint.CreateAttribute(f"physxTendon:{tendon}:{attr}", type_name).Set(value)
        axis_joint.AddAppliedSchema(f"PhysxTendonAxisAPI:{tendon}")
        axis_joint.CreateAttribute(f"physxTendon:{tendon}:gearing", Sdf.ValueTypeNames.FloatArray).Set([0.00705])
    stage.GetRootLayer().Save()
    _log_applied(f"{fix} (4 fingers)")


def patch_menagerie_asset(
    asset_dir: str,
    entry_name: str | None = None,
    *,
    author_static_friction: bool = True,
    filtered_body_pairs: Iterable[tuple[str, str]] = (),
    joint_velocity_limit_deg_s: float | None = None,
    joint_armature: float | None = None,
    author_shadow_tendons: bool = False,
) -> str:
    """Apply the Menagerie conversion fixes in place to a copy of an asset directory.

    The function is idempotent: it stamps a ``isaaclabMenageriePatchVersion`` marker on
    the entry layer's ``customLayerData`` and short-circuits when the marker is current,
    and every individual fix is detection-first, so re-running never double-authors.

    Args:
        asset_dir: Directory holding the entry ``.usda`` and its ``payloads/`` tree. It is
            modified in place, so callers must pass a copy, not the source asset.
        entry_name: File name of the entry layer within :paramref:`asset_dir`. When
            ``None``, the single top-level ``.usda`` file is used.
        author_static_friction: Copy the material's dynamic friction to the unauthored
            static coefficient.
        filtered_body_pairs: Body-name pairs to exclude from collision (pairs manufactured
            by the conversion's weld splits).
        joint_velocity_limit_deg_s: Max joint velocity [deg/s] to author on every revolute
            joint in the ``physx`` layer. When ``None``, no limit is authored.
        joint_armature: Motor rotor inertia to author on every revolute joint, in both
            the ``mujoco`` and ``physx`` layers. When ``None``, none is authored.
        author_shadow_tendons: Author the four Shadow-hand distal-middle fixed tendons in
            the ``physx`` layer.

    Returns:
        The path to the patched entry layer.
    """
    if entry_name is None:
        entries = [f for f in os.listdir(asset_dir) if f.endswith((".usda", ".usd"))]
        if len(entries) != 1:
            raise ValueError(f"Expected exactly one top-level USD entry in '{asset_dir}', found {entries}.")
        entry_name = entries[0]
    entry_path = os.path.join(asset_dir, entry_name)

    if _read_patch_marker(entry_path) == _PATCH_VERSION:
        print(f"[menagerie-patch] up-to-date: {asset_dir} (patch v{_PATCH_VERSION})")
        return entry_path

    physics_dir = os.path.join(asset_dir, "payloads", "Physics")
    mujoco_layer = os.path.join(physics_dir, "mujoco.usda")
    physics_layer = os.path.join(physics_dir, "physics.usda")
    physx_layer = os.path.join(physics_dir, "physx.usda")

    _strip_drive_deletes(mujoco_layer)
    _remove_mjc_actuators(mujoco_layer)
    if author_static_friction:
        _author_static_friction(physics_layer)
    if filtered_body_pairs:
        _author_filtered_pairs(physics_layer, list(filtered_body_pairs))
    if joint_velocity_limit_deg_s is not None:
        _author_joint_velocity_limit(physx_layer, joint_velocity_limit_deg_s)
    if joint_armature is not None:
        _author_joint_armature(mujoco_layer, physx_layer, joint_armature)
    if author_shadow_tendons:
        _author_shadow_fixed_tendons(physx_layer)

    _write_patch_marker(entry_path)
    return entry_path


def _read_patch_marker(entry_path: str) -> int | None:
    """Return the patch-version marker stamped on the entry layer, or ``None``."""
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(entry_path)
    if layer is None:
        return None
    return layer.customLayerData.get(_PATCH_MARKER_KEY)


def _write_patch_marker(entry_path: str) -> None:
    """Stamp the current patch version on the entry layer's ``customLayerData``."""
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(entry_path)
    data = dict(layer.customLayerData)
    data[_PATCH_MARKER_KEY] = _PATCH_VERSION
    layer.customLayerData = data
    layer.Save()


"""
Source resolution and patched-asset cache.
"""


def _s3_list_keys(bucket_host: str, prefix: str) -> list[str]:
    """Enumerate object keys under ``prefix`` via an anonymous S3 ``ListObjectsV2`` call."""
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys: list[str] = []
    token: str | None = None
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token is not None:
            query["continuation-token"] = token
        url = f"{bucket_host}/?{urllib.parse.urlencode(query)}"
        with urllib.request.urlopen(url, timeout=60) as response:
            root = ET.fromstring(response.read())
        keys.extend(node.text for node in root.findall(".//s3:Contents/s3:Key", namespace) if node.text)
        if root.findtext(".//s3:IsTruncated", default="false", namespaces=namespace) != "true":
            break
        token = root.findtext(".//s3:NextContinuationToken", namespaces=namespace)
    return keys


def _download_s3_asset(root_url: str, asset_reldir: str, dest_dir: str) -> None:
    """Download every file of an S3-hosted asset folder into ``dest_dir``."""
    parsed = urllib.parse.urlparse(root_url)
    bucket_host = f"{parsed.scheme}://{parsed.netloc}"
    prefix = f"{parsed.path.strip('/')}/{asset_reldir}/"
    keys = _s3_list_keys(bucket_host, prefix)
    if not keys:
        raise FileNotFoundError(f"No objects found under S3 prefix '{prefix}'.")
    for key in keys:
        rel = key[len(prefix) :]
        if not rel:
            continue
        dest = os.path.join(dest_dir, *rel.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = f"{bucket_host}/{urllib.parse.quote(key)}"
        with urllib.request.urlopen(url, timeout=120) as response, open(dest, "wb") as handle:
            shutil.copyfileobj(response, handle)


def _materialize_source(root: str, asset_reldir: str, dest_dir: str) -> None:
    """Copy or download the source asset folder into a fresh ``dest_dir``."""
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    if root.startswith(("http://", "https://")):
        os.makedirs(dest_dir, exist_ok=True)
        _download_s3_asset(root, asset_reldir, dest_dir)
    else:
        shutil.copytree(os.path.join(root, asset_reldir), dest_dir)


def _ensure_patched_asset(cfg: "MenageriePatchedUsdFileCfg") -> str:
    """Materialize (once) and patch the asset referenced by ``cfg``; return its entry path.

    The cache short-circuits on the entry layer's patch marker, so a materialize/patch only
    runs the first time an asset is spawned. A file lock serializes the work across ranks to
    avoid concurrent download/copy races on the shared cache directory.
    """
    usd_path = cfg.usd_path
    if not usd_path.startswith(MENAGERIE_ASSET_ROOT):
        raise ValueError(f"Menagerie asset '{usd_path}' is not under MENAGERIE_ASSET_ROOT '{MENAGERIE_ASSET_ROOT}'.")
    relpath = usd_path[len(MENAGERIE_ASSET_ROOT) :].lstrip("/")
    asset_reldir, entry_name = os.path.split(relpath)
    cache_dir = os.path.join(_PATCH_CACHE_ROOT, *asset_reldir.split("/"))
    entry_path = os.path.join(cache_dir, entry_name)

    os.makedirs(_PATCH_CACHE_ROOT, exist_ok=True)
    with FileLock(cache_dir.rstrip("/") + ".lock"):
        if not (os.path.exists(entry_path) and _read_patch_marker(entry_path) == _PATCH_VERSION):
            _materialize_source(MENAGERIE_ASSET_ROOT, asset_reldir, cache_dir)
            patch_menagerie_asset(
                cache_dir,
                entry_name,
                author_static_friction=cfg.author_static_friction,
                filtered_body_pairs=cfg.filtered_body_pairs,
                joint_velocity_limit_deg_s=cfg.joint_velocity_limit_deg_s,
                joint_armature=cfg.joint_armature,
                author_shadow_tendons=cfg.author_shadow_tendons,
            )
    return entry_path


def spawn_menagerie_asset(
    prim_path: str,
    cfg: "MenageriePatchedUsdFileCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> "Usd.Prim":
    """Spawn a Menagerie asset from its patched copy on disk.

    Ensures the patched copy exists (materialize + :func:`patch_menagerie_asset`), then
    delegates to the stock USD-file spawn path against the patched entry layer. No stage
    authoring happens here.
    """
    # Deferred imports: this module is imported by the task-config registry before the
    # simulation app starts, when pxr and the spawner internals are not yet loadable.
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
    from isaaclab.sim.utils import clone

    patched_usd_path = _ensure_patched_asset(cfg)

    @clone
    def _spawn(prim_path, cfg, translation=None, orientation=None, **inner_kwargs):
        return _spawn_from_usd_file(prim_path, patched_usd_path, cfg, translation, orientation)

    return _spawn(prim_path, cfg, translation, orientation, **kwargs)


@configclass
class MenageriePatchedUsdFileCfg(sim_utils.UsdFileCfg):
    """Spawn configuration that patches a Menagerie conversion on disk before loading it.

    The patch parameters describe the full set of fixes for the asset directory (shared by
    all physics variants of the same asset), so the cached copy is complete regardless of
    which variant spawns first.
    """

    func: Callable = spawn_menagerie_asset

    author_static_friction: bool = True
    """Copy the material's dynamic friction to the unauthored static coefficient."""

    filtered_body_pairs: list[tuple[str, str]] = []
    """Body-name pairs to exclude from collision (pairs manufactured by the conversion)."""

    joint_velocity_limit_deg_s: float | None = None
    """Max joint velocity [deg/s] to author on every revolute joint in the ``physx`` layer."""

    joint_armature: float | None = None
    """Motor rotor inertia to author on every revolute joint, in both physics variants."""

    author_shadow_tendons: bool = False
    """Author the four Shadow-hand distal-middle fixed tendons in the ``physx`` layer."""
