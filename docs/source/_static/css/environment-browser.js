/*
 * Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

(() => {
    "use strict";

    const initializeEnvironmentBrowser = () => {
        // Generated from the core and contributed Gym registry entries.
        // START-AUTO-GENERATED: environment-browser-task-rows
        const taskRows = [
            ["Isaac-Ant-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/classic/ant.jpg", true],
            ["Isaac-Ant", "rl_games,rsl_rl,skrl,sb3", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/classic/ant.jpg", true],
            ["Isaac-Cartpole-Direct", "rl_games,rsl_rl,skrl,sb3", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/classic/cartpole.jpg", true],
            ["Isaac-Cartpole", "rl_games,rsl_rl,skrl,sb3", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/classic/cartpole.jpg", true],
            ["Isaac-Cartpole-Camera-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo,depth,rgb,semantic_segmentation,simple_shading_constant_diffuse,simple_shading_diffuse_mdl,simple_shading_full_mdl", "tasks/classic/cartpole.jpg", false, {"*": ["rgb"], "rl_games": ["depth"]}],
            ["Isaac-Cartpole-Camera", "rl_games,rsl_rl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo,depth,resnet18,rgb,semantic_segmentation,simple_shading_constant_diffuse,simple_shading_diffuse_mdl,simple_shading_full_mdl,theia_tiny", "tasks/classic/cartpole.jpg", false, {"*": ["rgb"], "rsl_rl": ["resnet18", "theia_tiny"]}],
            ["Isaac-Fourbar-Pole-Swingup", "rsl_rl", "newton_kamino", "", "", "tasks/classic/fourbar_pole.jpg"],
            ["Isaac-Humanoid-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "tasks/classic/humanoid.jpg", true],
            ["Isaac-Humanoid", "rl_games,rsl_rl,skrl,sb3", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "tasks/classic/humanoid.jpg", true],
            ["Isaac-Lift-Cable-Franka", "rsl_rl", "newton_mjwarp_vbd_proxy", "", "ik,joint", "tasks/manipulation/franka_lift_cable.jpg", false, {"*": ["joint"]}],
            ["Isaac-Lift-Cable-Franka-Camera", "rsl_rl", "newton_mjwarp_vbd_proxy", "isaacsim_rtx,newton_renderer,ovrtx", "ik,joint", "tasks/manipulation/franka_lift_cable.jpg", false, {"*": ["joint"]}],
            ["Isaac-Lift-Cloth-Franka", "rsl_rl", "isaacsim_physx,newton_mjwarp_vbd_proxy", "", "ik,joint", "tasks/manipulation/franka_lift_cloth.jpg", false, {"*": ["joint"]}],
            ["Isaac-Lift-Cloth-Franka-Camera", "rsl_rl", "isaacsim_physx,newton_mjwarp_vbd_proxy", "isaacsim_rtx,newton_renderer,ovrtx", "ik,joint", "tasks/manipulation/franka_lift_cloth.jpg", false, {"*": ["joint"]}],
            ["Isaac-Lift-Franka", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "cube,shapes", "", false, {"*": ["shapes"]}],
            ["Isaac-Lift-KukaAllegro", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "cube,shapes", "tasks/manipulation/kuka_allegro_lift.jpg", false, {"*": ["shapes"]}],
            ["Isaac-Lift-KukaAllegro-Camera", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo128,albedo256,albedo64,cube,depth128,depth256,depth64,duo_camera,raycaster_depth128,raycaster_depth256,raycaster_depth64,rgb128,rgb256,rgb64,semantic_segmentation128,semantic_segmentation256,semantic_segmentation64,shapes,simple_shading_constant_diffuse128,simple_shading_constant_diffuse256,simple_shading_constant_diffuse64,simple_shading_diffuse_mdl128,simple_shading_diffuse_mdl256,simple_shading_diffuse_mdl64,simple_shading_full_mdl128,simple_shading_full_mdl256,simple_shading_full_mdl64,single_camera", "tasks/manipulation/kuka_allegro_lift.jpg", false, {"*": ["rgb64", "shapes", "single_camera"]}],
            ["Isaac-Lift-Soft-Franka", "rsl_rl", "isaacsim_physx,newton_mjwarp_vbd_proxy", "", "ik,joint", "newton/franka-mjwarp-vbd-coupling.png", false, {"*": ["joint"]}],
            ["Isaac-Lift-Soft-Franka-Camera", "rsl_rl", "isaacsim_physx,newton_mjwarp_vbd_proxy", "isaacsim_rtx,newton_renderer,ovrtx", "ik,joint", "newton/franka-mjwarp-vbd-coupling.png", false, {"*": ["joint"]}],
            ["Isaac-Open-Drawer-Franka-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "tasks/manipulation/franka_open_drawer.jpg"],
            ["Isaac-Open-Drawer-Franka", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "tasks/manipulation/franka_open_drawer.jpg"],
            ["Isaac-Pendulum-MARL-Direct", "rl_games,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/classic/cart_double_pendulum.jpg", false, {}, {"skrl": "MAPPO"}],
            ["Isaac-Reach-Franka", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "diffik,diffik_abs,joint_pos,newton_ik", "tasks/manipulation/franka_reach.jpg", true, {"*": ["joint_pos"]}],
            ["Isaac-Reach-Franka-OSC", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "diffik,diffik_abs,newton_ik", "tasks/manipulation/franka_reach.jpg"],
            ["Isaac-Reach-UR10", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "tasks/manipulation/ur10_reach.jpg", true],
            ["Isaac-Reorient-Cube-Allegro-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "tasks/manipulation/allegro_cube.jpg", true],
            ["Isaac-Reorient-Cube-Allegro", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "randomized,reset_only", "tasks/manipulation/allegro_cube.jpg", false, {"*": ["reset_only"]}],
            ["Isaac-Reorient-Cube-Shadow-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "tasks/manipulation/shadow_cube.jpg"],
            ["Isaac-Reorient-Cube-Shadow", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "asymmetric,randomized"],
            ["Isaac-Reorient-Cube-Shadow-Camera-Direct", "rl_games,rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo,depth,full,rgb,semantic_segmentation,simple_shading_constant_diffuse,simple_shading_diffuse_mdl,simple_shading_full_mdl", "tasks/manipulation/shadow_cube.jpg", false, {"*": ["full"]}],
            ["Isaac-Reorient-Cube-Shadow-Camera", "rl_games,rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo,depth,full,randomized,rgb,semantic_segmentation,simple_shading_constant_diffuse,simple_shading_diffuse_mdl,simple_shading_full_mdl", "", false, {"*": ["full"]}],
            ["Isaac-Reorient-Franka", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "cube,shapes", "", false, {"*": ["shapes"]}],
            ["Isaac-Reorient-KukaAllegro", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "cube,shapes", "tasks/manipulation/kuka_allegro_reorient.jpg", false, {"*": ["shapes"]}],
            ["Isaac-Reorient-KukaAllegro-Camera", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo128,albedo256,albedo64,cube,depth128,depth256,depth64,duo_camera,raycaster_depth128,raycaster_depth256,raycaster_depth64,rgb128,rgb256,rgb64,semantic_segmentation128,semantic_segmentation256,semantic_segmentation64,shapes,simple_shading_constant_diffuse128,simple_shading_constant_diffuse256,simple_shading_constant_diffuse64,simple_shading_diffuse_mdl128,simple_shading_diffuse_mdl256,simple_shading_diffuse_mdl64,simple_shading_full_mdl128,simple_shading_full_mdl256,simple_shading_full_mdl64,single_camera", "tasks/manipulation/kuka_allegro_reorient.jpg", false, {"*": ["rgb64", "shapes", "single_camera"]}],
            ["Isaac-Shadow-Handover-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "tasks/manipulation/shadow_hand_over.jpg", false, {}, {"skrl": "MAPPO"}],
            ["Isaac-Shadow-Handover", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "randomized"],
            ["Isaac-Velocity-Flat-AnymalD", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/anymal_d_flat.jpg", true],
            ["Isaac-Velocity-Flat-Cassie", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "", true],
            ["Isaac-Velocity-Flat-G1", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/g1_flat.jpg", true],
            ["Isaac-Velocity-Flat-H1", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/h1_flat.jpg", true],
            ["Isaac-Velocity-Flat-UnitreeGo2", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/go2_flat.jpg", true],
            ["Isaac-Velocity-Rough-AnymalD", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/anymal_d_rough.jpg"],
            ["Isaac-Velocity-Rough-Cassie", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/anymal_d_rough.jpg"],
            ["Isaac-Velocity-Rough-G1", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/g1_rough.jpg"],
            ["Isaac-Velocity-Rough-H1", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/h1_rough.jpg"],
            ["Isaac-Velocity-Rough-UnitreeGo2", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/go2_rough.jpg"],
            ["IsaacContrib-Assemble-Trocar-G129-Dex3", "rlinf", "", "", "", "tasks/manipulation/g1_assemble_trocar.jpg"],
            ["IsaacContrib-AutoMate-Assembly-Direct", "rl_games", "", "", "", "tasks/automate/00004.jpg"],
            ["IsaacContrib-AutoMate-Disassembly-Direct", "rl_games", "", "", "", "tasks/automate/01053_disassembly.jpg"],
            ["IsaacContrib-Cartpole-Camera-Showcase-Direct", "skrl", "", "", "box_box,box_discrete,box_multidiscrete,dict_box,dict_discrete,dict_multidiscrete,tuple_box,tuple_discrete,tuple_multidiscrete"],
            ["IsaacContrib-Cartpole-Showcase-Direct", "skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "box_box,box_discrete,box_multidiscrete,dict_box,dict_discrete,dict_multidiscrete,discrete_box,discrete_discrete,discrete_multidiscrete,multidiscrete_box,multidiscrete_discrete,multidiscrete_multidiscrete,tuple_box,tuple_discrete,tuple_multidiscrete"],
            ["IsaacContrib-Deploy-GearAssembly-Rizon4s-Grav", "rsl_rl", "", "", ""],
            ["IsaacContrib-Deploy-GearAssembly-Rizon4s-Grav-ROS-Inference", "rsl_rl", "", "", ""],
            ["IsaacContrib-Deploy-GearAssembly-UR10e-2F140", "rsl_rl", "", "", ""],
            ["IsaacContrib-Deploy-GearAssembly-UR10e-2F140-ROS-Inference", "rsl_rl", "", "", ""],
            ["IsaacContrib-Deploy-GearAssembly-UR10e-2F85", "rsl_rl", "", "", ""],
            ["IsaacContrib-Deploy-GearAssembly-UR10e-2F85-ROS-Inference", "rsl_rl", "", "", ""],
            ["IsaacContrib-Deploy-Reach-Rizon4s", "rsl_rl", "", "", ""],
            ["IsaacContrib-Deploy-Reach-Rizon4s-ROS-Inference", "rsl_rl", "", "", ""],
            ["IsaacContrib-Deploy-Reach-UR10e", "rsl_rl", "", "", "", "tasks/manipulation/ur10e_reach.jpg"],
            ["IsaacContrib-Deploy-Reach-UR10e-ROS-Inference", "rsl_rl", "", "", "", "tasks/manipulation/ur10e_reach.jpg"],
            ["IsaacContrib-DrLegs-HoldPose", "rsl_rl", "isaacsim_physx,newton_kamino", "", "", "tasks/locomotion/dr_legs.jpg"],
            ["IsaacContrib-DrLegs-Walk", "rsl_rl", "isaacsim_physx,newton_kamino", "", "", "tasks/locomotion/dr_legs.jpg"],
            ["IsaacContrib-ExhaustPipe-GR1T2-Pink-IK-Abs", "", "", "", ""],
            ["IsaacContrib-Factory-Franka", "rsl_rl", "isaacsim_physx", "", "accumulator,choice,gear_mesh_large,gear_mesh_medium,gear_mesh_small,nut_thread_m16,peg_insert_12mm,peg_insert_16mm,peg_insert_4mm,peg_insert_8mm,rod_insert_12mm,rod_insert_16mm,rod_insert_4mm,rod_insert_8mm"],
            ["IsaacContrib-Factory-GearMesh-Direct", "rl_games", "", "", "", "tasks/factory/gear_mesh.jpg"],
            ["IsaacContrib-Factory-NutThread-Direct", "rl_games", "", "", "", "tasks/factory/nut_thread.jpg"],
            ["IsaacContrib-Factory-PegInsert-Direct", "rl_games", "", "", "", "tasks/factory/peg_insert.jpg"],
            ["IsaacContrib-Forge-GearMesh-Direct", "rl_games", "", "", "", "tasks/factory/gear_mesh.jpg"],
            ["IsaacContrib-Forge-NutThread-Direct", "rl_games", "", "", "", "tasks/factory/nut_thread.jpg"],
            ["IsaacContrib-Forge-PegInsert-Direct", "rl_games", "", "", "", "tasks/factory/peg_insert.jpg"],
            ["IsaacContrib-Franka-Pour", "rsl_rl", "", "", "", "tasks/manipulation/franka_pour.jpg"],
            ["IsaacContrib-Humanoid-AMP-Dance-Direct", "skrl", "", "", "", "tasks/others/humanoid_amp.jpg"],
            ["IsaacContrib-Humanoid-AMP-Run-Direct", "skrl", "", "", "", "tasks/others/humanoid_amp.jpg"],
            ["IsaacContrib-Humanoid-AMP-Walk-Direct", "skrl", "", "", "", "tasks/others/humanoid_amp.jpg"],
            ["IsaacContrib-Keyboard-SO101", "rsl_rl", "isaacsim_physx,newton_mjwarp", "", ""],
            ["IsaacContrib-Lift-Cube-Franka", "rl_games,rsl_rl,skrl,sb3", "", "", "", "tasks/manipulation/franka_lift.jpg"],
            ["IsaacContrib-Lift-Cube-Franka-IK-Abs", "", "", "", "", "tasks/manipulation/franka_lift.jpg"],
            ["IsaacContrib-Lift-Cube-Franka-IK-Rel", "", "", "", "", "tasks/manipulation/franka_lift.jpg"],
            ["IsaacContrib-Lift-Cube-OpenArm", "rl_games,rsl_rl", "", "", "", "tasks/manipulation/openarm_uni_lift.jpg"],
            ["IsaacContrib-Multitask-Manipulation", "rsl_rl", "", "", "", "tasks/manipulation/multitask_manipulation.jpg"],
            ["IsaacContrib-Navigation-3DObstacles-ARL-Robot-1", "rl_games,rsl_rl,skrl", "", "", "", "tasks/drone_arl/arl_robot_1_navigation.jpg"],
            ["IsaacContrib-Navigation-Flat-AnymalC", "rsl_rl,skrl", "", "", "", "tasks/navigation/anymal_c_nav.jpg"],
            ["IsaacContrib-NutPour-GR1T2-Pink-IK-Abs", "", "", "", ""],
            ["IsaacContrib-Open-Drawer-Franka-IK-Abs", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", ""],
            ["IsaacContrib-Open-Drawer-Franka-IK-Rel", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", ""],
            ["IsaacContrib-Open-Drawer-OpenArm", "rl_games,rsl_rl", "", "", "", "tasks/manipulation/openarm_uni_open_drawer.jpg"],
            ["IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs", "", "", "", "", "tasks/manipulation/g1_pick_place_fixed_base.jpg"],
            ["IsaacContrib-PickPlace-G1-InspireFTP-Abs", "", "", "", "", "tasks/manipulation/g1_pick_place.jpg"],
            ["IsaacContrib-PickPlace-GR1T2-Abs", "", "", "isaacsim_rtx,newton_renderer,ovrtx", "", "tasks/manipulation/gr-1_pick_place.jpg"],
            ["IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs", "", "", "", "", "tasks/manipulation/gr-1_pick_place_waist.jpg"],
            ["IsaacContrib-PickPlace-Locomanipulation-G1-Abs", "", "", "isaacsim_rtx,newton_renderer,ovrtx", "", "tasks/manipulation/g1_pick_place_locomanipulation.jpg"],
            ["IsaacContrib-Place-Mug-Agibot-Left-Arm-RmpFlow", "", "isaacsim_physx", "", "", "tasks/manipulation/agibot_place_mug.jpg"],
            ["IsaacContrib-Place-Toy2Box-Agibot-Right-Arm-RmpFlow", "", "isaacsim_physx", "", "", "tasks/manipulation/agibot_place_toy.jpg"],
            ["IsaacContrib-Reach-OpenArm", "rl_games,rsl_rl,skrl", "", "", "", "tasks/manipulation/openarm_uni_reach.jpg"],
            ["IsaacContrib-Reach-OpenArmBi", "rl_games,rsl_rl", "", "", "", "tasks/manipulation/openarm_bi_reach.jpg"],
            ["IsaacContrib-Reorient-Cube-Shadow-OpenAI-FF-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "tasks/manipulation/shadow_cube.jpg"],
            ["IsaacContrib-Reorient-Cube-Shadow-OpenAI-LSTM-Direct", "rl_games,rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "tasks/manipulation/shadow_cube.jpg"],
            ["IsaacContrib-Stack-Cube-Bin-Franka-IK-Rel-Mimic", "", "isaacsim_physx", "", ""],
            ["IsaacContrib-Stack-Cube-BlueGreen-Franka-IK-Rel", "", "isaacsim_physx", "", ""],
            ["IsaacContrib-Stack-Cube-BlueGreenRed-Franka-IK-Rel", "", "isaacsim_physx", "", ""],
            ["IsaacContrib-Stack-Cube-Franka", "", "isaacsim_physx", "", "", "tasks/manipulation/franka_stack.jpg"],
            ["IsaacContrib-Stack-Cube-Franka-IK-Abs", "", "isaacsim_physx", "", "", "tasks/manipulation/franka_stack.jpg"],
            ["IsaacContrib-Stack-Cube-Franka-IK-Rel", "", "isaacsim_physx", "", "", "tasks/manipulation/franka_stack.jpg"],
            ["IsaacContrib-Stack-Cube-Franka-IK-Rel-Blueprint", "", "isaacsim_physx,newton_mjwarp", "", "", "tasks/manipulation/franka_stack.jpg"],
            ["IsaacContrib-Stack-Cube-Franka-IK-Rel-Skillgen", "", "isaacsim_physx", "", "", "tasks/manipulation/franka_stack.jpg"],
            ["IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor", "", "isaacsim_physx,newton_mjwarp", "", "", "tasks/manipulation/franka_stack.jpg"],
            ["IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos", "", "isaacsim_physx,newton_mjwarp", "", "", "tasks/manipulation/franka_stack.jpg"],
            ["IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow", "", "isaacsim_physx", "", "", "tasks/manipulation/galbot_stack_cube.jpg"],
            ["IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor", "", "isaacsim_physx", "isaacsim_rtx,newton_renderer,ovrtx", ""],
            ["IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-Joint-Position", "", "isaacsim_physx", "isaacsim_rtx,newton_renderer,ovrtx", ""],
            ["IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-RmpFlow", "", "isaacsim_physx", "isaacsim_rtx,newton_renderer,ovrtx", ""],
            ["IsaacContrib-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow", "", "isaacsim_physx", "", ""],
            ["IsaacContrib-Stack-Cube-Instance-Randomize-Franka", "", "", "", ""],
            ["IsaacContrib-Stack-Cube-Instance-Randomize-Franka-IK-Rel", "", "", "", ""],
            ["IsaacContrib-Stack-Cube-RedGreen-Franka-IK-Rel", "", "isaacsim_physx", "", ""],
            ["IsaacContrib-Stack-Cube-RedGreenBlue-Franka-IK-Rel", "", "isaacsim_physx", "", ""],
            ["IsaacContrib-Stack-Cube-SO101-IK-Abs-v0", "", "isaacsim_physx", "", ""],
            ["IsaacContrib-Stack-Cube-SO101-Joint-Teleop-v0", "", "isaacsim_physx", "", ""],
            ["IsaacContrib-Stack-Cube-SO101-v0", "", "isaacsim_physx", "", ""],
            ["IsaacContrib-Stack-Cube-UR10-Long-Suction-IK-Rel", "", "isaacsim_physx", "", "", "tasks/manipulation/ur10_stack_surface_gripper.jpg"],
            ["IsaacContrib-Stack-Cube-UR10-Short-Suction-IK-Rel", "", "isaacsim_physx", "", "", "tasks/manipulation/ur10_stack_surface_gripper.jpg"],
            ["IsaacContrib-TrackPositionNoObstacles-ARL-Robot-1", "rl_games,rsl_rl,skrl", "", "", "", "tasks/drone_arl/arl_robot_1_track_position_state_based.jpg"],
            ["IsaacContrib-Tracking-LocoManip-Digit", "rsl_rl", "isaacsim_physx", "", "", "tasks/locomotion/agility_digit_loco_manip.jpg"],
            ["IsaacContrib-UR10-Particle-Push", "rsl_rl", "", "", "", "tasks/manipulation/ur10_particle_push.jpg"],
            ["IsaacContrib-Velocity-Flat-AnymalB", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/anymal_b_flat.jpg", true],
            ["IsaacContrib-Velocity-Flat-AnymalC-Direct", "rl_games,rsl_rl,skrl", "", "", "", "tasks/locomotion/anymal_c_flat.jpg"],
            ["IsaacContrib-Velocity-Flat-AnymalC", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/anymal_c_flat.jpg", true],
            ["IsaacContrib-Velocity-Flat-Digit", "rsl_rl", "isaacsim_physx", "", "", "tasks/locomotion/agility_digit_flat.jpg"],
            ["IsaacContrib-Velocity-Flat-Spot", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp", "", "", "tasks/locomotion/spot_flat.jpg"],
            ["IsaacContrib-Velocity-Flat-UnitreeA1", "rsl_rl,skrl,sb3", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/a1_flat.jpg", true],
            ["IsaacContrib-Velocity-Flat-UnitreeGo1", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/go1_flat.jpg", true],
            ["IsaacContrib-Velocity-Rough-AnymalB", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/anymal_b_rough.jpg"],
            ["IsaacContrib-Velocity-Rough-AnymalC-Direct", "rl_games,rsl_rl,skrl", "", "", "", "tasks/locomotion/anymal_c_rough.jpg"],
            ["IsaacContrib-Velocity-Rough-AnymalC", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/anymal_c_rough.jpg"],
            ["IsaacContrib-Velocity-Rough-Digit", "rsl_rl", "isaacsim_physx", "", "", "tasks/locomotion/agility_digit_rough.jpg"],
            ["IsaacContrib-Velocity-Rough-UnitreeA1", "rsl_rl,skrl,sb3", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/a1_rough.jpg"],
            ["IsaacContrib-Velocity-Rough-UnitreeGo1", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/go1_rough.jpg"],
        ];
        // END-AUTO-GENERATED: environment-browser-task-rows

    const splitValues = (value) => value ? value.split(",") : [];
    const tasks = taskRows.map(([
        task, rl, physics, renderer, presets, previewImage = "", supportsWarpFrontend = false,
        pretrainedCheckpointPresetCompatibility = {},
        defaultAlgorithms = {},
    ]) => ({
        task,
        scope: task.startsWith("IsaacContrib-") ? "contrib" : "core",
        rl: splitValues(rl),
        physics: splitValues(physics),
        renderer: splitValues(renderer),
        presets: splitValues(presets),
        previewImage,
        supportsWarpFrontend,
        pretrainedCheckpointPresetCompatibility,
        defaultAlgorithms,
    }));

    const builder = document.querySelector("[data-environment-browser]");
    const preview = document.querySelector("[data-environment-preview]");
    const taskBrowser = document.querySelector("[data-environment-tasks]");
    const benchmarks = document.querySelector("[data-environment-benchmarks]");
    if (!builder || !preview || !taskBrowser) {
        return;
    }

    const fields = Object.fromEntries(
        [...builder.querySelectorAll("[data-environment-field]")].map((field) => [field.dataset.environmentField, field])
    );
    const commandOutput = builder.querySelector("[data-command-output]");
    const nonRlNote = builder.querySelector("[data-non-rl-note]");
    const copyButton = builder.querySelector("[data-copy-command]");
    const copyStatus = builder.querySelector("[data-copy-status]");
    const modeButtons = [...builder.querySelectorAll("[data-command-mode]")];
    const scopeButtons = [...taskBrowser.querySelectorAll("[data-task-scope]")];
    const taskList = taskBrowser.querySelector("[data-task-list]");
    const taskSearch = taskBrowser.querySelector("[data-task-search]");
    const taskCategory = taskBrowser.querySelector("[data-task-category]");
    const taskCount = taskBrowser.querySelector("[data-task-count]");
    const taskEmpty = taskBrowser.querySelector("[data-task-empty]");
    const taskCardRefreshers = new WeakMap();
    const state = {
        mode: "train",
        scope: "core",
        task: "Isaac-Cartpole",
        benchmarkWorkload: "collection",
        benchmarkChannel: "release",
    };
    const rlLibraryExtras = {rl_games: "rl-games", sb3: "sb3", skrl: "skrl", rlinf: "rlinf"};
    let benchmarkRows = [];
    const benchmarkErrors = new Set();

    const categoryFor = (task) => {
        if (/Velocity|Navigation|TrackPosition|Locomanip|Humanoid/.test(task)) {
            return "locomotion";
        }
        if (/Lift|Reach|Reorient|Drawer|Handover|Keyboard|Assemble|Assembly|Factory|Forge|Stack|PickPlace|Place|Particle|Cabinet|Gear|Trocar/.test(task)) {
            return "manipulation";
        }
        return "classic";
    };

    // Direct/camera variants of the same task are folded into a single card; this only strips
    // exact trailing suffixes, so unrelated tasks that merely contain "Camera"/"Direct" elsewhere
    // in their name are never merged.
    const variantOrder = ["manager", "direct", "camera", "direct-camera"];
    const variantLabels = {manager: "Manager", direct: "Direct", camera: "Camera", "direct-camera": "Direct-Camera"};
    const variantOf = (task) => {
        if (task.endsWith("-Camera-Direct")) {
            return "direct-camera";
        }
        if (task.endsWith("-Direct")) {
            return "direct";
        }
        if (task.endsWith("-Camera")) {
            return "camera";
        }
        return "manager";
    };
    const baseTaskName = (task) => task
        .replace(/-Camera-Direct$/, "")
        .replace(/-Direct$/, "")
        .replace(/-Camera$/, "");

    const groupTasks = (taskList) => {
        const groups = new Map();
        for (const task of taskList) {
            const base = baseTaskName(task.task);
            if (!groups.has(base)) {
                groups.set(base, []);
            }
            groups.get(base).push(task);
        }
        for (const variants of groups.values()) {
            variants.sort((left, right) => (
                variantOrder.indexOf(variantOf(left.task)) - variantOrder.indexOf(variantOf(right.task))
            ));
        }
        return groups;
    };

    // Keep capability labels compact on the cards while retaining the full names in tooltips and
    // accessible labels.
    const capabilitySymbolSets = [
        ["physics", [
            ["isaacsim_physx", "physx", "Isaac Sim PhysX"],
            ["newton_kamino", "kamino", "Newton Kamino"],
            ["newton_mjwarp", "mjwarp", "Newton MJWarp"],
            ["newton_mjwarp_vbd_proxy", "mjwarp vbd", "Newton MJWarp VBD proxy"],
            ["ovphysx", "ovphysx", "OV PhysX"],
        ]],
        ["renderer", [
            ["isaacsim_rtx", "rtx", "Isaac Sim RTX"],
            ["newton_renderer", "renderer", "Newton renderer"],
            ["ovrtx", "ovrtx", "OV RTX"],
        ]],
        ["rl", [
            ["rl_games", "rl_games", "RL Games"],
            ["rsl_rl", "rsl_rl", "RSL-RL"],
            ["skrl", "skrl", "skrl"],
            ["sb3", "sb3", "Stable-Baselines3"],
            ["rlinf", "rlinf", "RLinf"],
        ]],
    ];

    const buildCapabilitySymbols = (task) => {
        const values = {physics: task.physics, renderer: task.renderer, rl: task.rl};
        const icons = {physics: "fa-gears", renderer: "fa-eye", rl: "fa-chart-line"};
        return capabilitySymbolSets.flatMap(([type, symbols]) => symbols
            .filter(([value]) => values[type].includes(value))
            .map(([, shortLabel, fullLabel]) => {
                const symbol = document.createElement("span");
                symbol.className = `environment-task-symbol environment-task-symbol-${type}`;
                symbol.title = fullLabel;
                symbol.setAttribute("aria-label", fullLabel);
                const icon = document.createElement("i");
                icon.className = `fa-solid ${icons[type]}`;
                icon.setAttribute("aria-hidden", "true");
                const label = document.createElement("span");
                label.textContent = shortLabel;
                symbol.replaceChildren(icon, label);
                return symbol;
            }));
    };

    const preferredValue = (values, preferred) => preferred.find((value) => values.includes(value)) || values[0] || "";

    const populateSelect = (select, values, preferred) => {
        const choices = values.length ? values : [""];
        select.replaceChildren(...choices.map((value) => new Option(value || "default", value)));
        select.value = preferredValue(values, preferred);
        select.disabled = values.length === 0;
    };

    const selectedTask = () => tasks.find((task) => task.task === state.task) || tasks[0];

    const tasksForScope = (scope = state.scope) => tasks.filter((task) => (
        scope === "warp" ? task.supportsWarpFrontend : task.scope === scope
    ));

    const previewImageFor = (task) => {
        if (task.previewImage) {
            return task.previewImage;
        }
        const taskName = task.task;
        const imageRules = [
            [/Fourbar/, "tasks/classic/fourbar_pole.jpg"],
            [/Cartpole/, "tasks/classic/cartpole.jpg"],
            [/Pendulum/, "tasks/classic/cart_double_pendulum.jpg"],
            [/^Isaac-Ant/, "tasks/classic/ant.jpg"],
            [/^Isaac-Humanoid/, "tasks/classic/humanoid.jpg"],
            [/Lift-Cable-Franka/, "tasks/manipulation/franka_lift_cable.jpg"],
            [/Lift-Cloth-Franka/, "tasks/manipulation/franka_lift_cloth.jpg"],
            [/Lift-Soft-Franka/, "newton/franka-mjwarp-vbd-coupling.png"],
            [/Lift-(Cube-)?Franka/, "tasks/manipulation/franka_lift.jpg"],
            [/Lift-KukaAllegro/, "tasks/manipulation/kuka_allegro_lift.jpg"],
            [/Open-Drawer-Franka/, "tasks/manipulation/franka_open_drawer.jpg"],
            [/Reach-Franka/, "tasks/manipulation/franka_reach.jpg"],
            [/Reach-UR10/, "tasks/manipulation/ur10_reach.jpg"],
            [/Reorient-Cube-Allegro/, "tasks/manipulation/allegro_cube.jpg"],
            [/Reorient-Cube-Shadow/, "tasks/manipulation/shadow_cube.jpg"],
            [/Reorient-Franka/, "tasks/manipulation/franka_lift.jpg"],
            [/Reorient-KukaAllegro/, "tasks/manipulation/kuka_allegro_reorient.jpg"],
            [/Shadow-Handover/, "tasks/manipulation/shadow_hand_over.jpg"],
            [/AnymalB/, "tasks/locomotion/anymal_b_flat.jpg"],
            [/AnymalC/, "tasks/locomotion/anymal_c_flat.jpg"],
            [/AnymalD/, "tasks/locomotion/anymal_d_flat.jpg"],
            [/Cassie/, "tasks/locomotion/agility_digit_flat.jpg"],
            [/Digit/, "tasks/locomotion/agility_digit_flat.jpg"],
            [/Velocity-Flat-G1/, "tasks/locomotion/g1_flat.jpg"],
            [/Velocity-Rough-G1/, "tasks/locomotion/g1_rough.jpg"],
            [/Velocity-Flat-H1/, "tasks/locomotion/h1_flat.jpg"],
            [/Velocity-Rough-H1/, "tasks/locomotion/h1_rough.jpg"],
            [/Spot/, "tasks/locomotion/spot_flat.jpg"],
            [/UnitreeA1/, "tasks/locomotion/a1_flat.jpg"],
            [/UnitreeGo1/, "tasks/locomotion/go1_flat.jpg"],
            [/UnitreeGo2/, "tasks/locomotion/go2_flat.jpg"],
        ];
        return imageRules.find(([pattern]) => pattern.test(taskName))?.[1] || "tasks/classic/cartpole.jpg";
    };

    const previewVideos = {
        "Isaac-Cartpole": "cartpole-newton-mjwarp-rsl-rl.mp4",
        "Isaac-Ant": "ant-newton-mjwarp-rsl-rl.mp4",
        "Isaac-Velocity-Rough-G1": "velocity-rough-g1-newton-mjwarp-rsl-rl.mp4",
        "Isaac-Lift-KukaAllegro": "lift-kuka-allegro-newton-mjwarp-rsl-rl.mp4",
    };

    const updateTaskControls = () => {
        const task = selectedTask();
        populateSelect(fields.rl, task.rl, [fields.rl.value, "rsl_rl", "rl_games", "skrl", "sb3"]);
        const physics = state.scope === "warp" ? ["newton_mjwarp"] : task.physics;
        populateSelect(fields.physics, physics, [fields.physics.value, "newton_mjwarp", "isaacsim_physx", "ovphysx", "newton_kamino"]);
        const preferredRenderer = fields.physics.value.startsWith("newton") ? "newton_renderer" : "isaacsim_rtx";
        populateSelect(fields.renderer, task.renderer, [fields.renderer.value, preferredRenderer, "ovrtx"]);
        populateSelect(fields.presets, task.presets, [fields.presets.value, "joint", "ik", "rgb", "cube", "single_camera"]);
    };

    const updateModeControls = () => {
        const supportsRl = selectedTask().rl.length > 0;
        for (const modeButton of modeButtons) {
            modeButton.disabled = !supportsRl;
            const isActive = supportsRl && modeButton.dataset.commandMode === state.mode;
            modeButton.classList.toggle("is-active", isActive);
            modeButton.setAttribute("aria-pressed", String(isActive));
        }
        nonRlNote.hidden = supportsRl;
        const task = selectedTask();
        const selectedPreset = fields.presets.value;
        const compatiblePresets = [
            ...(task.pretrainedCheckpointPresetCompatibility["*"] || []),
            ...(task.pretrainedCheckpointPresetCompatibility[fields.rl.value] || []),
        ];
        const supportsPretrainedCheckpoint = supportsRl && state.scope === "core"
            && (!selectedPreset || compatiblePresets.includes(selectedPreset));
        fields.checkpoint.disabled = !supportsPretrainedCheckpoint;
        if (!supportsPretrainedCheckpoint) {
            fields.checkpoint.checked = false;
        }
    };

    const currentCommand = () => {
        const extras = [];
        if (fields.physics.value === "ovphysx" && fields.renderer.value === "ovrtx") {
            extras.push("ov");
        } else {
            if (fields.physics.value === "ovphysx") {
                extras.push("ovphysx");
            }
            if (fields.renderer.value === "ovrtx") {
                extras.push("ovrtx");
            }
        }
        if (fields.physics.value === "isaacsim_physx" || fields.renderer.value === "isaacsim_rtx") {
            extras.push("isaacsim");
        }
        if (rlLibraryExtras[fields.rl.value]) {
            extras.push(rlLibraryExtras[fields.rl.value]);
        }

        const parts = ["uv", "run"];
        if (extras.length) {
            parts.push("--extra", extras.join(","));
        }
        const task = selectedTask();
        const supportsRl = task.rl.length > 0;
        parts.push("isaaclab", supportsRl ? state.mode : "zero_agent");
        if (supportsRl && fields.rl.value) {
            parts.push("--rl_library", fields.rl.value);
        }
        parts.push("--task", state.task);
        const selectedAlgorithm = task.defaultAlgorithms[fields.rl.value];
        if (selectedAlgorithm) {
            parts.push("--algorithm", selectedAlgorithm);
        }
        if (state.scope === "warp") {
            parts.push("--frontend", "warp");
        }
        for (const selector of ["physics", "renderer", "presets"]) {
            if (fields[selector].value) {
                parts.push(`${selector}=${fields[selector].value}`);
            }
        }
        if (supportsRl && fields.checkpoint.checked) {
            parts.push("--checkpoint", "pretrained");
        }
        return parts.join(" ");
    };

    const updatePreview = () => {
        const previewImage = preview.querySelector("[data-preview-image]");
        const previewVideo = preview.querySelector("[data-preview-video]");
        const videoName = fields.rl.value === "rsl_rl" && fields.physics.value === "newton_mjwarp"
            ? previewVideos[state.task]
            : undefined;
        const previewImageName = previewImageFor(selectedTask()).split("/").pop();
        previewImage.src = new URL(`../../_images/${previewImageName}`, window.location.href).href;
        previewImage.alt = `${state.task} preview`;
        if (videoName) {
            const videoUrl = new URL(`../../_static/${videoName}`, window.location.href).href;
            if (previewVideo.src !== videoUrl) {
                previewVideo.src = videoUrl;
            }
            previewVideo.setAttribute("aria-label", `${state.task} preview`);
            previewVideo.hidden = false;
            previewImage.hidden = true;
            previewVideo.play().catch(() => {});
        } else {
            previewVideo.pause();
            previewVideo.hidden = true;
            previewImage.hidden = false;
        }
        preview.querySelector("[data-preview-task]").textContent = state.task;
        const supportsRl = selectedTask().rl.length > 0;
        preview.querySelector("[data-preview-mode]").textContent = supportsRl
            ? (state.mode === "train" ? "Train" : "Play")
            : "Zero agent";
        preview.querySelector("[data-preview-rl]").textContent = supportsRl ? fields.rl.value : "Not supported";
        preview.querySelector("[data-preview-physics]").textContent = fields.physics.value || "Default";
        preview.querySelector("[data-preview-renderer]").textContent = fields.renderer.value || "Default";
        preview.querySelector("[data-preview-presets]").textContent = fields.presets.value || "Default";
        const latestVramRows = benchmarkRows
            .filter((row) => row.channel === state.benchmarkChannel)
            .filter((row) => row.task === state.task && row.physics_backend === fields.physics.value)
            .filter((row) => !selectedTask().renderer.length || row.rendering_backend === fields.renderer.value)
            .filter((row) => row.rl_library === fields.rl.value)
            .sort((left, right) => right.recorded_at_utc.localeCompare(left.recorded_at_utc));
        const latestTraining = latestVramRows.find((row) => row.workload === "training");
        const vram = preview.querySelector("[data-preview-vram]");
        vram.textContent = latestTraining
            ? `${Number(latestTraining.vram_mean_gb).toFixed(2)} GB`
            : "Not available";
        vram.title = latestTraining
            ? `Peak VRAM: ${Number(latestTraining.vram_peak_gb).toFixed(2)} GB`
            : "";
        updateBenchmark();
    };

    const updateSelection = () => {
        fields.task.value = state.task;
        updateTaskControls();
        updateModeControls();
        commandOutput.textContent = currentCommand();
        updatePreview();
        for (const card of taskList.querySelectorAll(".environment-task-card")) {
            taskCardRefreshers.get(card)?.();
        }
        for (const row of taskList.querySelectorAll("[data-task-name]")) {
            const isSelected = row.dataset.taskName === state.task;
            row.classList.toggle("is-selected", isSelected);
            row.setAttribute("aria-pressed", String(isSelected));
        }
    };

    const createTaskCard = (variants) => {
        const card = document.createElement("div");
        card.className = "environment-task-card";

        const selectButton = document.createElement("button");
        selectButton.type = "button";
        selectButton.className = "environment-task-card-select";

        const image = document.createElement("img");
        image.alt = "";
        image.loading = "lazy";

        const content = document.createElement("span");
        content.className = "environment-task-card-content";
        const nameEl = document.createElement("span");
        nameEl.className = "environment-task-name";
        const symbolsEl = document.createElement("span");
        symbolsEl.className = "environment-task-symbols";
        content.append(nameEl);
        selectButton.append(image, content);

        const variantsRow = document.createElement("div");
        variantsRow.className = "environment-task-variants";
        variantsRow.setAttribute("role", "group");
        variantsRow.setAttribute("aria-label", "Task variant");
        for (const variantTask of variants) {
            const variantButton = document.createElement("button");
            variantButton.type = "button";
            variantButton.textContent = variantLabels[variantOf(variantTask.task)];
            variantButton.dataset.taskName = variantTask.task;
            variantButton.addEventListener("click", (event) => {
                event.stopPropagation();
                state.task = variantTask.task;
                refreshCard();
                updateSelection();
            });
            variantsRow.append(variantButton);
        }

        const refreshCard = () => {
            const activeTask = variants.find((task) => task.task === state.task) || variants[0];
            const isSelected = variants.includes(activeTask) && activeTask.task === state.task;
            image.src = new URL(`../../_static/${previewImageFor(activeTask)}`, window.location.href).href;
            image.alt = "";
            selectButton.dataset.taskName = activeTask.task;
            selectButton.setAttribute("aria-pressed", String(isSelected));
            nameEl.textContent = activeTask.task;
            symbolsEl.replaceChildren(...buildCapabilitySymbols(activeTask));
            card.classList.toggle("is-selected", isSelected);
            for (const button of variantsRow.querySelectorAll("[data-task-name]")) {
                const isActiveVariant = button.dataset.taskName === state.task;
                button.classList.toggle("is-selected", isActiveVariant);
                button.setAttribute("aria-pressed", String(isActiveVariant));
            }
        };

        selectButton.addEventListener("click", () => {
            state.task = selectButton.dataset.taskName;
            updateSelection();
        });

        taskCardRefreshers.set(card, refreshCard);
        refreshCard();
        card.append(selectButton, variantsRow, symbolsEl);
        return card;
    };

    const renderTasks = () => {
        const query = taskSearch.value.trim().toLowerCase();
        const category = taskCategory.value;
        const matchesFilter = (task) => {
            const searchableValues = [
                task.task,
                ...(task.physics.length ? task.physics : ["Default"]),
                ...(task.renderer.length ? task.renderer : ["Default"]),
                ...(task.rl.length ? task.rl : ["Not supported"]),
            ];
            const matchesQuery = searchableValues.some((value) => value.toLowerCase().includes(query));
            const matchesCategory = category === "all"
                || categoryFor(task.task) === category;
            return matchesQuery && matchesCategory;
        };
        const groups = groupTasks(tasksForScope());
        const visibleGroups = [...groups.values()]
            .map((variants) => variants.filter(matchesFilter))
            .filter((variants) => variants.length > 0);

        taskList.replaceChildren(...visibleGroups.map(createTaskCard));
        const matchingTaskCount = visibleGroups.reduce((total, variants) => total + variants.length, 0);
        taskCount.textContent = `${matchingTaskCount} ${matchingTaskCount === 1 ? "task" : "tasks"}`;
        taskEmpty.hidden = visibleGroups.length !== 0;
        taskList.hidden = visibleGroups.length === 0;
    };

    const initializeTasks = () => {
        fields.task.replaceChildren(...tasksForScope().map((task) => new Option(task.task, task.task)));
        fields.task.value = state.task;
        renderTasks();
        updateSelection();
    };

    const parseCsv = (contents) => {
        const parseLine = (line) => {
            const values = [];
            let value = "";
            let quoted = false;
            for (let index = 0; index < line.length; index += 1) {
                const character = line[index];
                if (character === '"' && quoted && line[index + 1] === '"') {
                    value += '"';
                    index += 1;
                } else if (character === '"') {
                    quoted = !quoted;
                } else if (character === "," && !quoted) {
                    values.push(value);
                    value = "";
                } else {
                    value += character;
                }
            }
            values.push(value);
            return values;
        };
        const [header, ...lines] = contents.trim().split(/\r?\n/).map(parseLine);
        return lines.map((line) => Object.fromEntries(header.map((name, index) => [name, line[index]])));
    };

    const formatFps = (value) => {
        if (value >= 1_000_000) {
            return `${Number((value / 1_000_000).toFixed(2))}M`;
        }
        if (value >= 1_000) {
            return `${Math.round(value / 1_000)}k`;
        }
        return Math.round(value).toString();
    };
    const standardFpsScale = (value) => {
        if (!Number.isFinite(value) || value <= 0) {
            return 1;
        }
        const paddedValue = value * 1.05;
        const magnitude = 10 ** Math.floor(Math.log10(paddedValue));
        return [1, 2, 5, 10].map((factor) => factor * magnitude)
            .find((candidate) => candidate >= paddedValue);
    };
    const backendLabels = {
        isaacsim_physx: "physx",
        newton_mjwarp: "mjwarp",
        newton_mjwarp_vbd_proxy: "mjwarp + vbd",
        ovphysx: "ovphysx",
    };
    const backendOrder = Object.keys(backendLabels);
    const backendClass = (backend) => `environment-chart-backend-${backend.replaceAll("_", "-")}`;
    const rendererLabels = {
        isaacsim_rtx: "rtx",
        newton_renderer: "newton",
        ovrtx: "ovrtx",
    };
    // Keep different benchmark configurations separate, including camera renderers.
    const seriesKey = (row) => JSON.stringify([
        row.physics_backend, row.rendering_backend, row.task_presets || "", row.num_envs, row.rl_library,
        row.camera_resolution || "",
    ]);
    const seriesLabel = (row) => [
        backendLabels[row.physics_backend] || row.physics_backend,
        rendererLabels[row.rendering_backend]
            || (row.task.includes("Camera") ? "Unspecified renderer" : ""),
        row.task_presets ? row.task_presets.split(",").join(", ")
            : (row.task.includes("Camera") ? "Presets not recorded" : ""),
        row.camera_resolution ? `${row.camera_resolution} px` : "",
        `${Number(row.num_envs).toLocaleString()} envs`,
        row.rl_library,
    ].filter(Boolean).join(" · ");
    const benchmarkSeries = (rows) => [...new Map(rows.map((row) => [seriesKey(row), row])).values()]
        .sort((left, right) => backendOrder.indexOf(left.physics_backend) - backendOrder.indexOf(right.physics_backend)
            || seriesKey(left).localeCompare(seriesKey(right)));
    const seriesColor = (row, rows) => {
        const index = benchmarkSeries(rows).findIndex((candidate) => seriesKey(candidate) === seriesKey(row));
        return `hsl(${(index * 137.508 + 30) % 360} 65% 42%)`;
    };

    const benchmarkDate = (row) => (row.snapshot_date_utc || row.benchmark_date_utc).slice(0, 10);
    const benchmarkDates = () => benchmarks.getAttribute(`data-benchmark-${state.benchmarkChannel}-dates`).split(",");
    const benchmarkFps = (row) => Number(row[state.benchmarkWorkload === "collection"
        ? "collection_fps_mean" : "total_fps_mean"]);
    const benchmarkMetricLabel = () => state.benchmarkWorkload === "collection" ? "Collection FPS" : "Training FPS";

    const renderBenchmarkChart = (rows, maximum) => {
        const namespace = "http://www.w3.org/2000/svg";
        const createSvgElement = (name, attributes = {}) => {
            const element = document.createElementNS(namespace, name);
            for (const [key, value] of Object.entries(attributes)) {
                element.setAttribute(key, value);
            }
            return element;
        };
        const svg = createSvgElement("svg", {viewBox: "0 0 600 360", role: "img"});
        const workloadLabel = state.benchmarkWorkload === "collection" ? "collection" : "training";
        svg.setAttribute("aria-label", `${state.task} ${workloadLabel} throughput history in frames per second`);
        const width = 600;
        const height = 360;
        const margins = {top: 38, right: 50, bottom: 60, left: 65};
        const plotWidth = width - margins.left - margins.right;
        const plotHeight = height - margins.top - margins.bottom;
        const dateKeys = benchmarkDates();
        const latestBySeriesAndDate = new Map();
        for (const row of rows) {
            const key = `${seriesKey(row)}:${benchmarkDate(row)}`;
            if (!latestBySeriesAndDate.has(key) || latestBySeriesAndDate.get(key).recorded_at_utc < row.recorded_at_utc) {
                latestBySeriesAndDate.set(key, row);
            }
        }
        const plottedRows = [...latestBySeriesAndDate.values()];
        const xPosition = (date) => dateKeys.length === 1
            ? margins.left + plotWidth / 2
            : margins.left + dateKeys.indexOf(date) * plotWidth / (dateKeys.length - 1);
        const yPosition = (value) => margins.top + plotHeight * (1 - value / maximum);

        for (let tick = 0; tick <= 4; tick += 1) {
            const value = maximum * tick / 4;
            const y = yPosition(value);
            svg.appendChild(createSvgElement("line", {
                x1: margins.left, y1: y, x2: width - margins.right, y2: y, class: "environment-chart-grid",
            }));
            const label = createSvgElement("text", {
                x: margins.left - 10, y: y + 4, class: "environment-chart-axis-label", "text-anchor": "end",
            });
            label.textContent = formatFps(value);
            svg.appendChild(label);
        }
        const axisTitle = createSvgElement("text", {
            x: 15, y: margins.top + plotHeight / 2, class: "environment-chart-axis-title",
            transform: `rotate(-90 15 ${margins.top + plotHeight / 2})`, "text-anchor": "middle",
        });
        axisTitle.textContent = benchmarkMetricLabel();
        svg.appendChild(axisTitle);

        const visibleDateIndexes = dateKeys.length <= 4
            ? dateKeys.map((_date, index) => index)
            : [0, Math.round((dateKeys.length - 1) / 3), Math.round(2 * (dateKeys.length - 1) / 3), dateKeys.length - 1];
        for (const index of [...new Set(visibleDateIndexes)]) {
            const date = dateKeys[index];
            const label = createSvgElement("text", {
                x: xPosition(date), y: height - 15, class: "environment-chart-axis-label", "text-anchor": "middle",
            });
            label.textContent = state.benchmarkChannel === "release" ? "EA 3.0" : new Date(`${date}T00:00:00Z`).toLocaleDateString(undefined, {
                month: "short", day: "numeric", year: "numeric", timeZone: "UTC",
            });
            svg.appendChild(label);
        }

        const configurations = benchmarkSeries(plottedRows);
        for (const representative of configurations) {
            const series = plottedRows
                .filter((row) => seriesKey(row) === seriesKey(representative))
                .sort((left, right) => benchmarkDate(left).localeCompare(benchmarkDate(right)));
            const seriesClass = backendClass(representative.physics_backend);
            const style = `--environment-series-color: ${seriesColor(representative, rows)}`;
            // Do not draw a trend through a snapshot with no measurement.
            for (let index = 1; index < series.length; index += 1) {
                const previous = series[index - 1];
                const current = series[index];
                if (dateKeys.indexOf(benchmarkDate(current)) - dateKeys.indexOf(benchmarkDate(previous)) !== 1) {
                    continue;
                }
                svg.appendChild(createSvgElement("line", {
                    x1: xPosition(benchmarkDate(previous)), y1: yPosition(benchmarkFps(previous)),
                    x2: xPosition(benchmarkDate(current)), y2: yPosition(benchmarkFps(current)),
                    class: `environment-chart-line ${seriesClass}`, style,
                }));
            }
            for (const [index, row] of series.entries()) {
                const date = benchmarkDate(row);
                const value = benchmarkFps(row);
                const x = xPosition(date);
                const y = yPosition(value);
                const circle = createSvgElement("circle", {
                    cx: x, cy: y, r: 5, class: `environment-chart-point ${seriesClass}`, style,
                });
                const title = createSvgElement("title");
                const tooltipWorkload = state.benchmarkWorkload === "collection" ? "Collection" : "Training";
                title.textContent = `${seriesLabel(row)} · ${tooltipWorkload}: ${Math.round(value).toLocaleString()} FPS · measured ${(row.measurement_timestamp || row.recorded_at_utc).slice(0, 10)}`;
                circle.appendChild(title);
                svg.appendChild(circle);
                // Dense camera charts expose exact values in tooltips without overlapping labels.
                if (index === series.length - 1 && configurations.length <= 3) {
                    const valueLabel = createSvgElement("text", {
                        x, y: y - 11, class: `environment-chart-value ${seriesClass}`, "text-anchor": "middle", style,
                    });
                    valueLabel.textContent = formatFps(value);
                    svg.appendChild(valueLabel);
                }
            }
        }
        return svg;
    };

    const renderBenchmarkLegend = (rows) => {
        const legend = benchmarks.querySelector(".environment-benchmark-legend");
        const entries = benchmarkSeries(rows).map((row) => {
            const entry = document.createElement("span");
            const swatch = document.createElement("i");
            swatch.className = `environment-legend-swatch ${backendClass(row.physics_backend)}`;
            swatch.style.setProperty("--environment-series-color", seriesColor(row, rows));
            swatch.setAttribute("aria-hidden", "true");
            entry.title = seriesLabel(row);
            entry.append(swatch, [backendLabels[row.physics_backend], rendererLabels[row.rendering_backend],
                row.task_presets ? row.task_presets.split(",").join(", ") : ""].filter(Boolean).join(" · "));
            return entry;
        });
        legend.replaceChildren(...entries);
    };

    const renderBenchmarkTable = (rows) => {
        const container = benchmarks.querySelector("[data-benchmark-table]");
        container.hidden = rows.length === 0;
        const table = document.createElement("table");
        table.setAttribute("aria-label", `${state.task} · ${benchmarkMetricLabel()}`);
        const header = table.createTHead().insertRow();
        for (const label of ["Configuration", "Mean FPS"]) {
            const cell = document.createElement("th");
            cell.scope = "col";
            cell.textContent = label;
            header.appendChild(cell);
        }
        const body = table.createTBody();
        const configurations = benchmarkSeries(rows);
        for (const physics of selectedTask().physics.filter((value) => value !== "newton_kamino")) {
            for (const renderer of selectedTask().renderer.length ? selectedTask().renderer : ["none"]) {
                if ((physics === "isaacsim_physx" && renderer === "ovrtx")
                    || (physics === "ovphysx" && renderer === "isaacsim_rtx")) {
                    continue;
                }
                if (!rows.some((row) => row.physics_backend === physics
                    && (renderer === "none" || row.rendering_backend === renderer))) {
                    configurations.push({physics_backend: physics, rendering_backend: renderer, missing: true});
                }
            }
        }
        for (const representative of configurations) {
            const entry = body.insertRow();
            const configuration = document.createElement("th");
            configuration.scope = "row";
            const label = document.createElement("span");
            const swatch = document.createElement("i");
            swatch.className = `environment-legend-swatch ${backendClass(representative.physics_backend)}`;
            swatch.style.setProperty("--environment-series-color", representative.missing
                ? "var(--environment-muted)" : seriesColor(representative, rows));
            swatch.setAttribute("aria-hidden", "true");
            label.append(swatch, backendLabels[representative.physics_backend]);
            configuration.title = representative.missing
                ? `${rendererLabels[representative.rendering_backend] || ""} 8,192-environment run unavailable`.trim()
                : seriesLabel(representative);
            configuration.appendChild(label);
            entry.appendChild(configuration);
            const row = rows.filter((candidate) => seriesKey(candidate) === seriesKey(representative))
                .sort((left, right) => benchmarkDate(right).localeCompare(benchmarkDate(left))
                    || right.recorded_at_utc.localeCompare(left.recorded_at_utc))[0];
            const value = entry.insertCell();
            if (!row) {
                value.textContent = "Not available";
                value.className = "environment-benchmark-missing";
                continue;
            }
            value.textContent = Math.round(benchmarkFps(row)).toLocaleString();
            const measuredDate = (row.measurement_timestamp || row.recorded_at_utc).slice(0, 10);
            value.title = `Measured ${measuredDate} · source record ${row.source_record_id} · ${row.source_entry_key} · commit ${row.git_commit}`;
        }

        container.replaceChildren(table);
    };

    const updateBenchmark = () => {
        if (!benchmarks) {
            return;
        }
        const taskRows = benchmarkRows.filter((row) => row.task === state.task && row.channel === state.benchmarkChannel);
        const rows = taskRows;
        const chart = benchmarks.querySelector("[data-benchmark-chart]");
        const empty = benchmarks.querySelector("[data-benchmark-empty]");
        const failed = benchmarkErrors.has(state.benchmarkChannel);
        const maximum = standardFpsScale(Math.max(...taskRows.flatMap((row) => [Number(row.collection_fps_mean), Number(row.total_fps_mean)])));
        chart.hidden = rows.length === 0;
        empty.hidden = rows.length !== 0 || failed;
        benchmarks.querySelector("[data-benchmark-error]").hidden = !failed;
        renderBenchmarkLegend(rows);
        renderBenchmarkTable(rows);
        chart.replaceChildren(...(rows.length ? [renderBenchmarkChart(rows, maximum)] : []));
    };

    const renderBenchmarks = async () => {
        if (!benchmarks) {
            return;
        }
        benchmarkRows = (await Promise.all(["release", "develop"].map(async (channel) => {
            try {
                const source = new URL(benchmarks.getAttribute(`data-benchmark-${channel}-source`), window.location.href);
                const response = await fetch(source);
                if (!response.ok) {
                    throw new Error(`Benchmark request failed with ${response.status}`);
                }
                return parseCsv(await response.text())
                    .filter((row) => row.data_origin === "measured" && row.workload === "training")
                    .filter((row) => [row.collection_fps_mean, row.total_fps_mean]
                        .every((value) => Number.isFinite(Number(value)) && Number(value) > 0))
                    .filter((row) => row.task.startsWith("Isaac-") && row.physics_backend !== "newton_kamino")
                    .filter((row) => !(row.physics_backend === "isaacsim_physx" && row.rendering_backend === "ovrtx")
                        && !(row.physics_backend === "ovphysx" && row.rendering_backend === "isaacsim_rtx"))
                    .map((row) => ({...row, channel}));
            } catch (error) {
                benchmarkErrors.add(channel);
                console.error(error);
                return [];
            }
        }))).flat();
        updatePreview();
    };

    fields.task.addEventListener("change", () => {
        state.task = fields.task.value;
        updateSelection();
    });
    for (const field of [fields.rl, fields.physics, fields.renderer, fields.presets, fields.checkpoint]) {
        field.addEventListener("change", () => {
            updateModeControls();
            commandOutput.textContent = currentCommand();
            updatePreview();
        });
    }
    for (const button of modeButtons) {
        button.addEventListener("click", () => {
            state.mode = button.dataset.commandMode;
            if (state.mode === "train") {
                fields.checkpoint.checked = false;
            }
            for (const modeButton of modeButtons) {
                const isActive = modeButton === button;
                modeButton.classList.toggle("is-active", isActive);
                modeButton.setAttribute("aria-pressed", String(isActive));
            }
            updateModeControls();
            commandOutput.textContent = currentCommand();
            updatePreview();
        });
    }
    for (const button of scopeButtons) {
        button.disabled = tasksForScope(button.dataset.taskScope).length === 0;
        button.addEventListener("click", () => {
            const scope = button.dataset.taskScope;
            const scopedTasks = tasksForScope(scope);
            if (scopedTasks.length === 0) {
                return;
            }
            state.scope = scope;
            for (const scopeButton of scopeButtons) {
                const isActive = scopeButton === button;
                scopeButton.classList.toggle("is-active", isActive);
                scopeButton.setAttribute("aria-pressed", String(isActive));
            }
            if (!scopedTasks.some((task) => task.task === state.task)) {
                state.task = scopedTasks[0].task;
            }
            fields.task.replaceChildren(...scopedTasks.map((task) => new Option(task.task, task.task)));
            updateModeControls();
            renderTasks();
            updateSelection();
        });
    }
    fields.checkpoint.addEventListener("change", () => {
        if (!fields.checkpoint.checked || state.mode === "play") {
            return;
        }
        state.mode = "play";
        for (const modeButton of modeButtons) {
            const isActive = modeButton.dataset.commandMode === "play";
            modeButton.classList.toggle("is-active", isActive);
            modeButton.setAttribute("aria-pressed", String(isActive));
        }
        commandOutput.textContent = currentCommand();
        updatePreview();
    });
    for (const button of benchmarks?.querySelectorAll("[data-benchmark-channel]") || []) {
        button.addEventListener("click", () => {
            state.benchmarkChannel = button.dataset.benchmarkChannel;
            for (const channelButton of benchmarks.querySelectorAll("[data-benchmark-channel]")) {
                const isActive = channelButton === button;
                channelButton.classList.toggle("is-active", isActive);
                channelButton.setAttribute("aria-pressed", String(isActive));
            }
            updatePreview();
        });
    }
    for (const button of benchmarks?.querySelectorAll("[data-benchmark-workload]") || []) {
        button.addEventListener("click", () => {
            state.benchmarkWorkload = button.dataset.benchmarkWorkload;
            for (const workloadButton of benchmarks.querySelectorAll("[data-benchmark-workload]")) {
                const isActive = workloadButton === button;
                workloadButton.classList.toggle("is-active", isActive);
                workloadButton.setAttribute("aria-pressed", String(isActive));
            }
            updateBenchmark();
        });
    }
    copyButton.addEventListener("click", async () => {
        const command = currentCommand();
        try {
            await navigator.clipboard.writeText(command);
        } catch (_error) {
            const textArea = document.createElement("textarea");
            textArea.value = command;
            textArea.style.position = "fixed";
            textArea.style.opacity = "0";
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand("copy");
            textArea.remove();
        }
        copyStatus.textContent = "Copied";
        copyButton.innerHTML = '<i class="fa-solid fa-check" aria-hidden="true"></i>';
        window.setTimeout(() => {
            copyStatus.textContent = "";
            copyButton.innerHTML = '<i class="fa-regular fa-copy" aria-hidden="true"></i>';
        }, 1600);
    });
    taskSearch.addEventListener("input", renderTasks);
    taskCategory.addEventListener("change", renderTasks);

        updateModeControls();
        initializeTasks();
        renderBenchmarks();
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initializeEnvironmentBrowser, {once: true});
    } else {
        initializeEnvironmentBrowser();
    }
})();
