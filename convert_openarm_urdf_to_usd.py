#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_openarm_urdf_to_usd.py
===============================================================================
把 openarm_ros_joint_control_gripper.usd 場景裡的機器人
(目前 reference 到 Isaac 官方雲端資產 openarm_bimanual.usd)
就地替換成 v1_camera.urdf 匯入產生的機器人。

保留：Table / workspace_pad / Cube_2 / front_cam / GroundPlane / light /
      physicsScene / /Graph/ROS_JointStates 整組 ROS2 OmniGraph。

流程：
  Phase 1 (需要 Isaac Sim)：URDF -> USD，輸出獨立的機器人資產 USD
  Phase 2 (純 pxr，可獨立跑)：打開場景 USD，砍掉舊 Robot prim，
          建新的 Robot Xform reference 到 Phase 1 的資產，
          重建 physics override / 初始關節角 / 相機 / 修正 OmniGraph 目標

執行方式（Windows / Isaac Sim 安裝目錄下）
-------------------------------------------------------------------------------
  python.bat convert_openarm_urdf_to_usd.py ^
      --urdf      "D:/IsaacLab_csl/openarm_description-main/assets/robot/openarm_v1.0/urdf/example/v1_camera.urdf" ^
      --scene     "D:/.../openarm_ros_joint_control_gripper.usd" ^
      --robot-usd "D:/IsaacLab_csl/usd/openarm_v1_camera.usd" ^
      --out       "D:/.../openarm_ros_joint_control_gripper_urdf.usd"

  # 只跑 Phase 2（已經有 robot USD 了，不需要 Isaac Sim）
  python convert_openarm_urdf_to_usd.py --skip-import ^
      --scene ... --robot-usd ... --out ...

重要前置條件：package:// 路徑解析
-------------------------------------------------------------------------------
URDF 裡的 mesh 是 package://openarm_description/assets/...
Isaac 的 URDF Importer 是從 .urdf 往上層目錄找「名稱等於 package 名」的資料夾。
但實際資料夾叫 openarm_description-main（GitHub zip 解壓出來的名字），名字對不上，
importer 會找不到 mesh，轉出來每個 link 都會是空的 Xform。

本腳本會自動處理：匯入前先把 package://openarm_description/ 改寫成絕對路徑，
產生一份 *.resolved.urdf 再餵給 importer（原始 URDF 不會被動到），
同時檢查 44 個 mesh 是否真的都在，缺哪個會直接列出來。
package root 預設自動偵測（往上找底下有 assets/ 的 openarm_description* 目錄），
偵測不到就用 --package-root 手動指定。
===============================================================================
"""

import argparse
import os
import sys

# =============================================================================
# 匯入設定 —— 這裡的每個值都對應原本 USD 場景裡的設定，改之前請先看註解
# =============================================================================

# False = 保留 fixed joint 對應的 link。
#   必須是 False！不然 openarm_left/right_camera_link、link0、hand_tcp
#   會被 merge 進父 link，相機就沒有 prim 可以掛。
MERGE_FIXED_JOINTS = False

# URDF 根 link 是 "world" + fixed joint，機器人要固定在地面上
FIX_BASE = True

# 對應原場景 physxArticulation:enabledSelfCollisions = 0
SELF_COLLISION = False

# URDF 有完整 <inertial>，直接用它而不要讓 importer 從幾何估算
IMPORT_INERTIA_TENSOR = True

# URDF 單位已經是公尺（metersPerUnit = 1）
DISTANCE_SCALE = 1.0

# 位置驅動（ROS2 subscriber 送 positionCommand -> ArticulationController）
DEFAULT_DRIVE_TYPE = 1          # 0=none, 1=position, 2=velocity
DEFAULT_DRIVE_STRENGTH = 1e7    # stiffness
DEFAULT_POSITION_DRIVE_DAMPING = 1e5

# openarm_*_finger_joint2 在 URDF 裡是 <mimic joint="finger_joint1"/>。
#   False = 兩根手指各自是獨立可驅動關節（跟原本 Isaac 資產的行為一致，
#           /openarm/vr_joint_command_processed 可以同時命令 joint1 與 joint2）
#   True  = joint2 變成 mimic joint，不再吃外部命令
# 想維持現有 ROS 介面就保持 False。
PARSE_MIMIC = False

# 原場景所有 link 都設了 disableGravity = 1（重力關閉）。
# 要讓機器人受重力就改成 False。
DISABLE_GRAVITY = True
MAX_DEPENETRATION_VELOCITY = 1.0

# 原場景 articulation solver 設定
SOLVER_POSITION_ITERATIONS = 8
SOLVER_VELOCITY_ITERATIONS = 0

# 場景中機器人的位置
ROBOT_PRIM_PATH = "/World/envs/env_0/Robot"

# 依 URDF 的 camera_link 建立 Camera。鏡頭參數沿用原場景 wrist_cam。
# 局部旋轉沿用原 wrist_cam 的光學朝向：視線 ≈ camera_link 的 +Z（工具方向），
# 往 -X 微傾 6 度，與原本 wrist_cam 看出去的方向一致。
CAMERA_MOUNTS = [
    # (camera_link prim 名稱, 要建立的 Camera prim 名稱)
    ("openarm_left_camera_link",  "wrist_cam"),
    ("openarm_right_camera_link", "right_wrist_cam"),
]
CAMERA_LOCAL_ORIENT = (0.03700987249612808, 0.7061375975608826,
                       0.7061375379562378, 0.03700987249612808)  # (w, x, y, z)
CAMERA_PARAMS = dict(
    clippingRange=(0.05, 2.0),
    focalLength=24.0,
    focusDistance=400.0,
    fStop=0.0,
    horizontalAperture=20.955,
    verticalAperture=15.71625,
    horizontalApertureOffset=0.0,
    verticalApertureOffset=0.0,
)

# URDF 沒有身體相機對應的 link。原場景的 body_cam 掛在 openarm_body_link 上，
# 這裡改掛到 URDF 的 openarm_body_link0，位姿與參數照抄。不要就設 False。
RECREATE_BODY_CAM = True
BODY_CAM_PARENT = "openarm_body_link0"
BODY_CAM_NAME = "body_cam"
BODY_CAM_TRANSLATE = (0.009999999776482582, 0.0, 0.6499999761581421)
BODY_CAM_ORIENT = (0.6841982007026672, 0.1785295307636261,
                   -0.1785295307636261, -0.6841982007026672)
BODY_CAM_PARAMS = dict(CAMERA_PARAMS, clippingRange=(0.1, 3.0))

# OmniGraph 節點路徑
GRAPH_PUBLISHER = "/Graph/ROS_JointStates/PublisherJointState"
GRAPH_ARTICULATION_CTRL = "/Graph/ROS_JointStates/articulation_controller"


# =============================================================================
# Phase 0 — 把 package:// 改寫成絕對路徑
# =============================================================================
PACKAGE_NAME = "openarm_description"


def find_package_root(urdf_path: str, package_name: str = PACKAGE_NAME):
    """從 URDF 所在目錄往上找 package root。

    先找名稱剛好等於 package_name 的目錄；找不到再放寬成 package_name*
    （對應 openarm_description-main 這種 GitHub zip 解壓的名字）。
    兩種都要求底下有 assets/ 才算數。
    """
    d = os.path.dirname(os.path.abspath(urdf_path))
    exact, prefixed = None, None
    while True:
        name = os.path.basename(d)
        if os.path.isdir(os.path.join(d, "assets")):
            if name == package_name and exact is None:
                exact = d
            elif name.startswith(package_name) and prefixed is None:
                prefixed = d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return exact or prefixed


def resolve_package_paths(urdf_path: str, package_root: str = None,
                          package_name: str = PACKAGE_NAME) -> str:
    """把 package://<name>/ 換成絕對路徑，輸出 *.resolved.urdf。

    回傳改寫後的 URDF 路徑；如果檔案裡根本沒有 package://，就原封不動回傳原路徑。
    """
    import xml.etree.ElementTree as ET

    with open(urdf_path, "r", encoding="utf-8") as f:
        text = f.read()

    prefix = f"package://{package_name}/"
    if prefix not in text:
        print(f"[phase0] URDF 沒有 {prefix}，不需改寫")
        return urdf_path

    if package_root is None:
        package_root = find_package_root(urdf_path, package_name)
    if package_root is None:
        raise RuntimeError(
            f"找不到 {package_name} 的 package root（往上找底下有 assets/ 的目錄）。\n"
            f"請用 --package-root 指定，例如：\n"
            f'  --package-root "D:/IsaacLab_csl/openarm_description-main"')

    package_root = os.path.abspath(package_root)
    if not os.path.isdir(os.path.join(package_root, "assets")):
        print(f"[phase0] ⚠ {package_root} 底下沒有 assets/，路徑可能不對")
    print(f"[phase0] package root = {package_root}")

    # USD/URDF 這條路徑統一用正斜線，Windows 一樣吃得下
    abs_prefix = package_root.replace(os.sep, "/").rstrip("/") + "/"
    new_text = text.replace(prefix, abs_prefix)

    out_path = os.path.splitext(os.path.abspath(urdf_path))[0] + ".resolved.urdf"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(new_text)

    # 檢查每個 mesh 是不是真的存在 —— 這是最常炸掉的地方，先擋下來
    root = ET.fromstring(new_text)
    refs = [m.get("filename") for m in root.iter("mesh") if m.get("filename")]
    total = len(set(refs))
    missing = sorted({fn for fn in refs if not os.path.isfile(fn)})
    if missing:
        print(f"[phase0] ⚠ {len(missing)}/{total} 個 mesh 找不到檔案：")
        for p in missing[:10]:
            print(f"          {p}")
        if len(missing) > 10:
            print(f"          ...還有 {len(missing) - 10} 個")
        print("[phase0] ⚠ 這些 link 匯入後會是空的 Xform，請先確認 package root 對不對")
    else:
        print(f"[phase0] {total} 個 mesh 全部存在 ✓")

    print(f"[phase0] 已產生 {out_path}")
    return out_path


# =============================================================================
# Phase 1 — URDF -> USD（需要 Isaac Sim）
# =============================================================================
def phase1_import_urdf(urdf_path: str, dest_usd: str) -> str:
    """用 Isaac Sim 的 URDF Importer 把 URDF 轉成獨立的機器人 USD 資產。"""
    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": True})

    import omni.kit.commands
    from omni.isaac.core.utils.extensions import enable_extension  # noqa
    try:
        from isaacsim.core.utils.extensions import enable_extension
    except ImportError:
        pass
    enable_extension("isaacsim.asset.importer.urdf")

    status, cfg = omni.kit.commands.execute("URDFCreateImportConfig")

    def _set(name, value):
        """新舊版 API 都吃：先試 set_xxx()，不行就直接設 attribute。"""
        setter = getattr(cfg, "set_" + name, None)
        if callable(setter):
            setter(value)
        elif hasattr(cfg, name):
            setattr(cfg, name, value)
        else:
            print(f"[warn] ImportConfig 沒有 '{name}'，略過")

    _set("merge_fixed_joints", MERGE_FIXED_JOINTS)
    _set("fix_base", FIX_BASE)
    _set("self_collision", SELF_COLLISION)
    _set("import_inertia_tensor", IMPORT_INERTIA_TENSOR)
    _set("distance_scale", DISTANCE_SCALE)
    _set("default_drive_type", DEFAULT_DRIVE_TYPE)
    _set("default_drive_strength", DEFAULT_DRIVE_STRENGTH)
    _set("default_position_drive_damping", DEFAULT_POSITION_DRIVE_DAMPING)
    _set("parse_mimic", PARSE_MIMIC)
    _set("convex_decomp", False)        # URDF 已提供 *_symp.stl 簡化碰撞體
    _set("collision_from_visuals", False)
    _set("create_physics_scene", False)  # 場景本身已有 /physicsScene
    _set("make_default_prim", True)
    _set("replace_cylinders_with_capsules", False)
    _set("override_joint_dynamics", False)  # 用 URDF 的 effort/velocity limit

    print(f"[phase1] 匯入 {urdf_path}")
    print(f"[phase1] 輸出 {dest_usd}")
    os.makedirs(os.path.dirname(os.path.abspath(dest_usd)), exist_ok=True)

    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=cfg,
        dest_path=dest_usd,
    )
    print(f"[phase1] 完成，stage 內 prim path = {prim_path}")

    sim_app.close()
    return dest_usd


# =============================================================================
# Phase 2 — 把新機器人接回場景（純 pxr，不需要 Isaac Sim）
# =============================================================================
def phase2_swap_into_scene(scene_usd: str, robot_usd: str, out_usd: str):
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
    except ImportError:
        raise SystemExit(
            "\n[錯誤] 這個 Python 沒有 pxr（USD 函式庫）。\n"
            "  你現在用的是 Anaconda 的 python，它不帶 USD。兩個解法：\n\n"
            "  A) 用 Isaac Sim 自帶的 python（推薦，Phase 1 本來就一定要用它）：\n"
            "       C:\\isaacsim\\python.bat convert_openarm_urdf_to_usd.py ...\n"
            "     不知道裝在哪就跑：where /r C:\\ python.bat\n\n"
            "  B) 只想跑 Phase 2 的話，在目前這個環境裝 usd-core：\n"
            "       pip install usd-core\n")

    # ---- 先把舊場景裡想保留的資訊撈出來 ------------------------------------
    # 直接讀 Sdf layer：舊的 joints 是 `over` spec，而且外部 reference 資產
    # （Isaac 雲端 openarm_bimanual.usd）在離線時載不到，走 composed stage
    # 會整批讀不到值。
    src_layer = Sdf.Layer.FindOrOpen(scene_usd)
    if src_layer is None:
        raise RuntimeError(f"打不開場景 USD：{scene_usd}")
    old_joint_state = {}   # joint 名 -> {attr 名: 值}
    joints_spec = src_layer.GetPrimAtPath(Sdf.Path(ROBOT_PRIM_PATH + "/joints"))
    if joints_spec:
        for jspec in joints_spec.nameChildren:
            for aname, aspec in jspec.attributes.items():
                if aname.endswith("physics:position") or aname.endswith("physics:velocity"):
                    if aspec.default is not None:
                        old_joint_state.setdefault(jspec.name, {})[aname] = aspec.default
    print(f"[phase2] 從舊場景取回 {len(old_joint_state)} 個關節的初始狀態")
    del src_layer, joints_spec

    # ---- 掃描新機器人資產，找 articulation root / rigid body / camera link --
    rstage = Usd.Stage.Open(robot_usd)
    if not rstage:
        raise RuntimeError(f"打不開機器人 USD：{robot_usd}")
    robot_default = rstage.GetDefaultPrim()
    root_offset = ""
    if robot_default:
        root_offset = ""  # reference 會直接把 defaultPrim 掛在目標 prim 底下

    art_root_rel = None
    rigid_bodies = []
    joint_prims = []
    link_by_name = {}
    base = robot_default.GetPath() if robot_default else Sdf.Path("/")
    for p in rstage.Traverse():
        rel = p.GetPath().MakeRelativePath(base)
        if str(rel) in (".", ""):
            rel = None
        if p.HasAPI(UsdPhysics.ArticulationRootAPI) and art_root_rel is None:
            art_root_rel = str(rel) if rel else ""
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(str(rel) if rel else "")
        if p.IsA(UsdPhysics.Joint):
            joint_prims.append((p.GetName(), str(rel) if rel else ""))
        link_by_name.setdefault(p.GetName(), str(rel) if rel else "")
    print(f"[phase2] 機器人資產：rigid body {len(rigid_bodies)} 個、"
          f"joint {len(joint_prims)} 個、articulation root = "
          f"{art_root_rel if art_root_rel is not None else '(找不到)'}")
    del rstage

    # ---- 打開場景，砍掉舊 Robot，建新的 --------------------------------------
    stage = Usd.Stage.Open(scene_usd)
    stage.RemovePrim(Sdf.Path(ROBOT_PRIM_PATH))

    robot = UsdGeom.Xform.Define(stage, ROBOT_PRIM_PATH)
    rp = robot.GetPrim()

    # reference 用相對路徑，場景搬家也不會斷
    ref_path = os.path.relpath(os.path.abspath(robot_usd),
                               os.path.dirname(os.path.abspath(out_usd)))
    if not ref_path.startswith("."):
        ref_path = "./" + ref_path
    rp.GetReferences().AddReference(ref_path.replace(os.sep, "/"))

    # ---- 小工具：apiSchemas 一定要用 prepend list-op，不能用 explicit ------
    # explicit 會把 reference 進來的 PhysicsRigidBodyAPI / MassAPI / CollisionAPI
    # 整組蓋掉，機器人就廢了。
    def _prepend_api_schemas(prim, names):
        cur = prim.GetMetadata("apiSchemas")
        existing = list(cur.prependedItems) if cur else []
        for n in names:
            if n not in existing:
                existing.append(n)
        op = Sdf.TokenListOp()
        op.prependedItems = existing
        prim.SetMetadata("apiSchemas", op)

    def _set_srt(prim, translate, orient, scale=(1, 1, 1)):
        """建立 translate / orient(quatd) / scale 三段 xformOp，順序與原場景一致。"""
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translate))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(orient[0], Gf.Vec3d(*orient[1:])))
        xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))

    # 沿用原本的 xformOp 組合與 semantic label
    _set_srt(rp, (0, 0, 0), (1, 0, 0, 0))
    rp.CreateAttribute("semantics:labels:class",
                       Sdf.ValueTypeNames.TokenArray).Set(["robot"])
    _prepend_api_schemas(rp, ["SemanticsLabelsAPI:class"])
    rp.CreateAttribute("isaac:namespace", Sdf.ValueTypeNames.String).Set("")

    def _over(rel_path: str):
        """在 Robot 底下針對 referenced prim 開一個 over。"""
        path = ROBOT_PRIM_PATH if not rel_path else f"{ROBOT_PRIM_PATH}/{rel_path}"
        return stage.OverridePrim(Sdf.Path(path))

    # ---- 重建每個 rigid body 的 PhysX override ------------------------------
    for rel in rigid_bodies:
        prim = _over(rel)
        _prepend_api_schemas(prim, ["PhysxRigidBodyAPI"])
        prim.CreateAttribute("physxRigidBody:disableGravity",
                             Sdf.ValueTypeNames.Bool).Set(DISABLE_GRAVITY)
        prim.CreateAttribute("physxRigidBody:maxDepenetrationVelocity",
                             Sdf.ValueTypeNames.Float
                             ).Set(MAX_DEPENETRATION_VELOCITY)
        prim.CreateAttribute("physics:velocity",
                             Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(0, 0, 0))
        prim.CreateAttribute("physics:angularVelocity",
                             Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(0, 0, 0))
    print(f"[phase2] 套用 {len(rigid_bodies)} 個 rigid body 的 PhysX 設定"
          f"（disableGravity={DISABLE_GRAVITY}）")

    # ---- articulation solver 設定 ------------------------------------------
    if art_root_rel is not None:
        ar = _over(art_root_rel)
        ar.CreateAttribute("physxArticulation:enabledSelfCollisions",
                           Sdf.ValueTypeNames.Bool).Set(SELF_COLLISION)
        ar.CreateAttribute("physxArticulation:solverPositionIterationCount",
                           Sdf.ValueTypeNames.Int).Set(SOLVER_POSITION_ITERATIONS)
        ar.CreateAttribute("physxArticulation:solverVelocityIterationCount",
                           Sdf.ValueTypeNames.Int).Set(SOLVER_VELOCITY_ITERATIONS)

    # ---- 把舊場景的初始關節角搬過來（名字對得上的才套） ----------------------
    applied = 0
    for jname, rel in joint_prims:
        if jname not in old_joint_state:
            continue
        jp = _over(rel)
        for attr_name, value in old_joint_state[jname].items():
            jp.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float).Set(float(value))
        applied += 1
    print(f"[phase2] 套用 {applied} 個關節的初始狀態 "
          f"（舊場景有 {len(old_joint_state)} 個）")
    missing = sorted(set(old_joint_state) - {j for j, _ in joint_prims})
    if missing:
        print(f"[phase2] 新 URDF 沒有這些舊關節，已略過：{missing}")

    # ---- 依 URDF 的 camera_link 建 Camera -----------------------------------
    def _make_camera(parent_rel, cam_name, translate, orient, params):
        cam_path = Sdf.Path(f"{ROBOT_PRIM_PATH}/{parent_rel}/{cam_name}")
        cam = UsdGeom.Camera.Define(stage, cam_path)
        cp = cam.GetPrim()
        cam.CreateClippingRangeAttr(Gf.Vec2f(*params["clippingRange"]))
        cam.CreateFocalLengthAttr(params["focalLength"])
        cam.CreateFocusDistanceAttr(params["focusDistance"])
        cam.CreateFStopAttr(params["fStop"])
        cam.CreateHorizontalApertureAttr(params["horizontalAperture"])
        cam.CreateVerticalApertureAttr(params["verticalAperture"])
        cam.CreateHorizontalApertureOffsetAttr(params["horizontalApertureOffset"])
        cam.CreateVerticalApertureOffsetAttr(params["verticalApertureOffset"])
        cp.CreateAttribute("cameraProjectionType", Sdf.ValueTypeNames.Token,
                           custom=True).Set("pinhole")
        cp.CreateAttribute("omni:kit:cameraLock", Sdf.ValueTypeNames.Bool).Set(True)
        _set_srt(cp, translate, orient)
        return cam_path

    for link_name, cam_name in CAMERA_MOUNTS:
        rel = link_by_name.get(link_name)
        if rel is None:
            print(f"[phase2] ⚠ 機器人資產裡找不到 {link_name}，"
                  f"相機 {cam_name} 沒有建立。"
                  f"（多半是 MERGE_FIXED_JOINTS 沒設成 False）")
            continue
        p = _make_camera(rel, cam_name, (0, 0, 0), CAMERA_LOCAL_ORIENT, CAMERA_PARAMS)
        print(f"[phase2] 建立相機 {p}")

    if RECREATE_BODY_CAM:
        rel = link_by_name.get(BODY_CAM_PARENT)
        if rel is None:
            print(f"[phase2] ⚠ 找不到 {BODY_CAM_PARENT}，body_cam 沒有建立")
        else:
            p = _make_camera(rel, BODY_CAM_NAME, BODY_CAM_TRANSLATE,
                             BODY_CAM_ORIENT, BODY_CAM_PARAMS)
            print(f"[phase2] 建立相機 {p}")

    # ---- 修正 OmniGraph 指向 -------------------------------------------------
    art_abs = ROBOT_PRIM_PATH if not art_root_rel else \
        f"{ROBOT_PRIM_PATH}/{art_root_rel}"

    pub = stage.GetPrimAtPath(GRAPH_PUBLISHER)
    if pub:
        rel_attr = pub.GetRelationship("inputs:targetPrim")
        if not rel_attr:
            rel_attr = pub.CreateRelationship("inputs:targetPrim", custom=True)
        rel_attr.SetTargets([Sdf.Path(art_abs)])
        print(f"[phase2] PublisherJointState.targetPrim -> {art_abs}")
    else:
        print(f"[phase2] ⚠ 找不到 {GRAPH_PUBLISHER}")

    ctrl = stage.GetPrimAtPath(GRAPH_ARTICULATION_CTRL)
    if ctrl:
        a = ctrl.GetAttribute("inputs:robotPath")
        if not a:
            a = ctrl.CreateAttribute("inputs:robotPath",
                                     Sdf.ValueTypeNames.String, custom=True)
        a.Set(art_abs)
        r = ctrl.GetRelationship("inputs:targetPrim")
        if not r:
            r = ctrl.CreateRelationship("inputs:targetPrim", custom=True)
        r.SetTargets([Sdf.Path(art_abs)])
        print(f"[phase2] articulation_controller.robotPath -> {art_abs}")
    else:
        print(f"[phase2] ⚠ 找不到 {GRAPH_ARTICULATION_CTRL}")

    # ---- 存檔 ---------------------------------------------------------------
    # 更新 omni_layer.authoring_layer，否則 Kit 會指回舊檔名
    root_layer = stage.GetRootLayer()
    cld = dict(root_layer.customLayerData)
    if "omni_layer" in cld:
        ol = dict(cld["omni_layer"])
        ol["authoring_layer"] = "./" + os.path.basename(out_usd)
        cld["omni_layer"] = ol
        root_layer.customLayerData = cld

    root_layer.Export(out_usd)
    print(f"[phase2] 已輸出 {out_usd}")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", help="v1_camera.urdf 的路徑")
    ap.add_argument("--scene", required=True, help="原始 openarm_ros_joint_control_gripper.usd")
    ap.add_argument("--robot-usd", required=True, help="URDF 轉出來的機器人 USD 路徑")
    ap.add_argument("--out", required=True, help="輸出的新場景 USD")
    ap.add_argument("--package-root",
                    help="openarm_description package 的根目錄"
                         "（底下要有 assets/）。不給就自動往上找")
    ap.add_argument("--no-resolve-package", action="store_true",
                    help="不要改寫 package:// 路徑，直接把原始 URDF 餵給 importer")
    ap.add_argument("--skip-import", action="store_true",
                    help="跳過 Phase 0/1，直接用現有的 --robot-usd")
    args = ap.parse_args()

    # ---- 參數健檢：這幾個錯很常見，先擋在前面 -------------------------------
    if not os.path.isfile(args.scene):
        ap.error(f"找不到場景 USD：{args.scene}")

    if os.path.isdir(args.out):
        ap.error(f"--out 必須是「檔案」路徑，不是資料夾：{args.out}\n"
                 f"       例如：--out \"{os.path.join(args.out, 'openarm_ros_joint_control_gripper_urdf.usd')}\"")
    if os.path.splitext(args.out)[1].lower() not in (".usd", ".usda", ".usdc"):
        ap.error(f"--out 要以 .usd / .usda / .usdc 結尾：{args.out}")

    if os.path.isdir(args.robot_usd):
        ap.error(f"--robot-usd 必須是「檔案」路徑，不是資料夾：{args.robot_usd}")

    if args.skip_import and not os.path.isfile(args.robot_usd):
        ap.error(
            f"--skip-import 是「用現成的機器人 USD」，但這個檔不存在：\n"
            f"       {args.robot_usd}\n"
            f"       第一次跑請「不要」加 --skip-import，讓 Phase 1 先把 URDF 轉出來。\n"
            f"       Phase 1 需要 Isaac Sim 的 python.bat，不能用 conda 的 python。")

    for p in (args.out, args.robot_usd):
        d = os.path.dirname(os.path.abspath(p))
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
            print(f"[info] 已建立資料夾 {d}")

    if not args.skip_import:
        if not args.urdf:
            ap.error("沒有 --skip-import 就必須給 --urdf")
        if not os.path.isfile(args.urdf):
            ap.error(f"找不到 URDF：{args.urdf}")
        urdf_for_import = args.urdf
        if not args.no_resolve_package:
            urdf_for_import = resolve_package_paths(args.urdf, args.package_root)
        phase1_import_urdf(urdf_for_import, args.robot_usd)
    else:
        print("[phase0/1] 已跳過")

    phase2_swap_into_scene(args.scene, args.robot_usd, args.out)
    print("\n完成。用 Isaac Sim 打開 --out 的檔案，按 Play 後：")
    print("  發布 /isaac/openarm/joint_states")
    print("  訂閱 /openarm/vr_joint_command_processed")
    print("  關節名稱與原本完全相同（openarm_{left,right}_joint1..7 + finger_joint1/2）")


if __name__ == "__main__":
    sys.exit(main())