# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A Franka cube-stacking config with runtime visual DR attached.

Lives here rather than beside the runtime because it is a task configuration:
it names this task's cameras, semantic classes and scene, and imports
``isaaclab_tasks``. ``isaaclab_contrib`` cannot depend on ``isaaclab_tasks`` --
the dependency runs the other way, through ``isaaclab_assets`` -- so the
runtime stays task-agnostic and the task-coupled parts live with the task.

The scene is already semantically tagged (``robot``, ``table``, ``ground``,
``cube_1..3``), so the preserved foreground is a list of class names rather than
an asset-authoring exercise. The prompts describe only what surrounds the table,
because that is all the composite is allowed to replace.
"""

from isaaclab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_contrib.visual_dr import CameraDRCfg, CosmosBackendCfg, PromptBankCfg, VisualDRCfg
from isaaclab_contrib.visual_dr.cosmos import CosmosBackend
from isaaclab_contrib.visual_dr.observations import image_runtime_dr

from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.config.franka.stack_ik_rel_visuomotor_cosmos_env_cfg import (
    FrankaCubeStackVisuomotorCosmosEnvCfg,
)
from isaaclab_tasks.utils.presets import set_isaac_rtx_global_settings

FOREGROUND_CLAUSE = (
    "In the foreground a plain matte grey metal workbench, bare except for a white "
    "robot arm and exactly three small matte plastic cubes -- one red, one green, one "
    "blue. The workbench stays flat grey metal and the cubes keep their red, green and "
    "blue colours. Nothing else rests on the workbench."
)
"""Repeated in every variant.

Only matters when ``composite_foreground`` is cleared: with the composite on, these
pixels are pasted back from the render and the prompt cannot affect them. Without
it the model has no other description of the foreground, so it invents one -- the
workbench drifts in colour and laboratory clutter appears on it. Naming what must
stay is weak compared with pasting, but it is the only lever in that mode."""

SCENE_PROMPTS = (
    "A photograph of a biological research laboratory: researchers in white coats "
    "walking between workstations behind the bench, fume hoods and glass-fronted "
    "reagent cabinets along the far wall, bright even ceiling lighting, natural "
    "photographic colour and realistic skin tones.",
    "A candid photograph inside a busy life-sciences lab: scientists in lab coats and "
    "safety glasses moving between the far benches, centrifuges and incubators stacked "
    "against the back wall, cool fluorescent lighting, pale epoxy floors, documentary "
    "photography, high dynamic range.",
    "A real photograph of a biotech cleanroom corridor: technicians in white coats and "
    "gloves walking past in the distance, brushed stainless equipment racks and sealed "
    "glass doors along the walls, diffuse overhead lighting, photographic film grain, "
    "professional architectural photography.",
    "A photograph of a university biology laboratory during the day: postgraduate "
    "researchers in white coats crossing the room behind the bench, microscopes and "
    "pipette racks on the far benches, daylight through tall windows, realistic "
    "shadows and natural colour grading, shot on a full-frame camera at f/4.",
)
"""Scene-coupled by design: these describe this room and would be wrong elsewhere.
Each is combined with :data:`FOREGROUND_CLAUSE` so the room can change while the
workbench and cubes are described as fixed."""

BACKGROUND_PROMPTS = tuple(f"{scene} {FOREGROUND_CLAUSE}" for scene in SCENE_PROMPTS)
"""What the bank actually contains: scene plus the fixed-foreground clause."""

FOCAL_LENGTH = 12.0
"""Roughly an 82 degree horizontal field of view at the task's 20.955 mm aperture,
up from the 47 degrees the tight task framing uses. Lower widens further, at the
cost of the cubes covering fewer pixels."""

CLIPPING_RANGE = (0.05, 50.0)
"""Metres. The task ships a 2 m far plane, which clips away the very region the
background prompt is supposed to fill."""

CAMERA_RESOLUTION = (640, 480)
"""Width and height. Matches the 480 bucket at 4:3 exactly, so the frame goes to
Cosmos and comes back with no resampling in either direction -- the task's 200x200
was upsampled to 640x640 to generate and squeezed back down again."""

TABLE_CAM_POSITION = (1.35, 0.0, 0.7)
"""Pulled back and raised from the task's (1.0, 0.0, 0.4). The table is preserved
foreground and fills the lower frame, so the room only becomes a meaningful part
of the image once the camera stands off from it."""

NEGATIVE_PROMPT = (
    # Style: the checkpoint has plenty of stylised imagery to fall back on, and a
    # positive request for realism competes with it rather than excluding it.
    "cartoon, anime, illustration, drawing, painting, sketch, comic, cel shading, "
    "flat shading, posterized, oversaturated colours, video game screenshot, 3D "
    "render, CGI, toy-like, unrealistic proportions, low detail, "
    # Content: without these the model furnishes the workbench with laboratory
    # equipment and repaints it, which destroys the only surface the task uses.
    "clutter on the workbench, objects on the table, bottles on the bench, tools on "
    "the bench, papers on the bench, equipment on the workbench, additional cubes, "
    "wooden table, patterned tabletop, coloured tabletop, reflective glass tabletop, "
    "recolored cubes, extra objects in the foreground"
)
"""Steers away from the illustrated look the prompt alone does not rule out. The
checkpoint has plenty of stylised imagery in its training distribution, and a
positive prompt asking for realism competes with it rather than excluding it."""

TIME_OF_DAY = (
    "Shot in the early morning: pale dawn light through the tall windows, long soft "
    "shadows across the floor, cool blue-grey tones, interior lights still off.",
    "Shot in mid-morning: bright clear daylight through the windows, crisp shadows, neutral white balance.",
    "Shot at midday: strong overhead sunlight through the windows, bright high-contrast light and short hard shadows.",
    "Shot in the afternoon: warm golden sunlight angling through the windows, "
    "long shadows stretching across the benches.",
    "Shot at sunset: low orange sunlight through the windows, warm amber highlights "
    "on the equipment, deep shadows elsewhere.",
    "Shot at dusk during blue hour: dim blue twilight outside the windows, the "
    "interior ceiling lights now on and dominant.",
    "Shot at night: the windows are dark and reflective, lit only by cool "
    "fluorescent ceiling lights, the room's reflection visible in the glass.",
    "Shot at midnight: pitch black outside the windows, only overhead interior "
    "lighting, strong mirror-like reflections on the dark glass.",
)
"""Walked once across the run so the lab's windows drift from dawn to midnight.
Appended to whichever scene variant the episode selected, so the room stays the
same room while its light changes."""

DR_CAMERA = "table_cam"
"""Only the table view is randomized. The wrist camera sits centimetres from the
table and barely sees the room, so restyling it costs a generation per frame and
changes almost nothing."""

PRESERVE_CLASSES = ("robot", "table", "cube_1", "cube_2", "cube_3")
"""Everything the policy needs to act on. ``ground`` and anything untagged is
background, and untagged geometry is preserved anyway under the default
``unknown_policy``."""


@configclass
class StackRewardsCfg:
    """A staged reward for cube stacking, absent from the imitation-learning task.

    Weights climb by roughly an order of magnitude per stage so that reaching
    never outweighs grasping and grasping never outweighs a completed stack.
    They are a starting point, not a tuned result.
    """

    reach = RewardTermCfg(
        func=mdp.ee_object_distance,
        weight=0.1,
        params={"object_cfg": SceneEntityCfg("cube_2"), "std": 0.1},
    )
    grasp = RewardTermCfg(func=mdp.object_is_grasped, weight=1.0)
    stack = RewardTermCfg(func=mdp.object_is_stacked, weight=10.0)
    success = RewardTermCfg(func=mdp.stacking_success, weight=100.0)


@configclass
class FrankaStackRuntimeDRCfg(FrankaCubeStackVisuomotorCosmosEnvCfg):
    """Cube stacking with both cameras randomized.

    Cameras emit uncolored semantic IDs so the mask can be built from class
    names, and the auxiliary image observations are dropped because the policy
    consumes RGB only. Attach the runtime after construction -- observation
    dimension probes must not load a model.
    """

    visual_dr: VisualDRCfg = VisualDRCfg()
    rewards: StackRewardsCfg = StackRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # The Cosmos parent config renders with antialiasing off, which on a
        # data-center GPU means no denoising at all: DLSS Ray Reconstruction is
        # unavailable on H100, so RTX Real-Time returns raw ~1-spp output and the
        # images arrive covered in speckle. DLAA denoises without depending on ray
        # reconstruction, and is what the non-Cosmos visuomotor task already uses.
        # Re-rendering on reset matches that task too, so the first frame of an
        # episode is settled rather than half-converged.
        self.num_rerenders_on_reset = 3
        # Denoising is a render setting rather than a DR one, so both cameras get
        # it; only the table view is randomized.
        for name in ("table_cam", "wrist_cam"):
            set_isaac_rtx_global_settings(getattr(self.scene, name).renderer_cfg, antialiasing_mode="DLAA")

        camera = getattr(self.scene, DR_CAMERA)
        # The task frames this camera tightly on the table behind a 2 m far plane,
        # which leaves almost no background for DR to act on. Widen the lens, push
        # the far plane out and stand the camera off so the replaceable region is a
        # real part of the image rather than a strip above the table.
        camera.spawn.focal_length = FOCAL_LENGTH
        camera.spawn.clipping_range = CLIPPING_RANGE
        camera.offset.pos = TABLE_CAM_POSITION
        camera.width, camera.height = CAMERA_RESOLUTION
        camera.data_types = ["rgb", "distance_to_image_plane", "semantic_segmentation"]
        # The matching fields on CameraCfg are deprecated and do not reach the RTX
        # renderer, which still reports colorized ``idToLabels`` keys.
        camera.renderer_cfg.colorize_semantic_segmentation = False
        camera.renderer_cfg.semantic_segmentation_mapping = {}
        setattr(
            self.observations.policy,
            DR_CAMERA,
            ObservationTermCfg(func=image_runtime_dr, params={"camera": DR_CAMERA}),
        )

        for name in ("table_cam_segmentation", "table_cam_normals", "table_cam_depth"):
            setattr(self.observations.policy, name, None)

        self.visual_dr = VisualDRCfg(
            enabled=True,
            probability=1.0,
            cameras={
                # A one-pixel ring hides the seam the composite leaves, since the
                # background was generated without knowing what covers it.
                DR_CAMERA: CameraDRCfg(preserve_classes=PRESERVE_CLASSES, boundary_px=1)
            },
            backend=CosmosBackendCfg(
                class_type=CosmosBackend,
                prompts=PromptBankCfg(
                    variants=BACKGROUND_PROMPTS,
                    negative_prompt=NEGATIVE_PROMPT,
                    progression=TIME_OF_DAY,
                    # Matches the demo rollout length, so one run spans dawn to
                    # midnight; a longer run simply holds at midnight.
                    progression_steps=230,
                ),
                # Segmentation rather than depth: it names regions instead of
                # pinning exact geometry, so the model keeps the scene's layout and
                # perspective while staying free to fill the room with people.
                # Depth insists the space is an empty flat floor and no amount of
                # prompting puts researchers in it.
                control_kind="seg",
                control_guidance=2.0,  # Cosmos' tuned value for the seg hint.
                # 640x480 is the 480 bucket at 4:3, matching the camera exactly.
                aspect_ratio="4,3",
            ),
        )
