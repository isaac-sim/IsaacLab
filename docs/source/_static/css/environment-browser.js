/*
 * Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

(() => {
    "use strict";

    const initializeEnvironmentBrowser = () => {
        // Generated from the core and contributed rows in source/overview/environments.rst.
        // START-AUTO-GENERATED: environment-browser-task-rows
        const taskRows = [
            ["Isaac-Ant-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", {}, "tasks/classic/ant.jpg"],
            ["Isaac-Ant", "rl_games,rsl_rl,skrl,sb3", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", {}, "tasks/classic/ant.jpg"],
            ["Isaac-Cartpole-Direct", "rl_games,rsl_rl,skrl,sb3", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", {}, "tasks/classic/cartpole.jpg"],
            ["Isaac-Cartpole", "rl_games,rsl_rl,skrl,sb3", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", {}, "tasks/classic/cartpole.jpg"],
            ["Isaac-Cartpole-Camera-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo,depth,rgb,semantic_segmentation,simple_shading_constant_diffuse,simple_shading_diffuse_mdl,simple_shading_full_mdl", {}, "tasks/classic/cartpole.jpg"],
            ["Isaac-Cartpole-Camera", "rl_games,rsl_rl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo,depth,resnet18,rgb,semantic_segmentation,simple_shading_constant_diffuse,simple_shading_diffuse_mdl,simple_shading_full_mdl,theia_tiny", {"rl_games_cfg_entry_point": ["albedo", "depth", "rgb", "semantic_segmentation", "simple_shading_constant_diffuse", "simple_shading_diffuse_mdl", "simple_shading_full_mdl"], "rl_games_feature_cfg_entry_point": ["resnet18", "theia_tiny"], "rsl_rl_cfg_entry_point": ["albedo", "depth", "rgb", "semantic_segmentation", "simple_shading_constant_diffuse", "simple_shading_diffuse_mdl", "simple_shading_full_mdl"], "rsl_rl_feature_cfg_entry_point": ["resnet18", "theia_tiny"]}, "tasks/classic/cartpole.jpg"],
            ["Isaac-Fourbar-Pole-Swingup", "rsl_rl", "newton_kamino", "", "", {}, "tasks/classic/fourbar_pole.jpg"],
            ["Isaac-Humanoid-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", {}, "tasks/classic/humanoid.jpg"],
            ["Isaac-Humanoid", "rl_games,rsl_rl,skrl,sb3", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", {}, "tasks/classic/humanoid.jpg"],
            ["Isaac-Lift-Cable-Franka", "rsl_rl", "newton_mjwarp_vbd_proxy", "", "ik,joint", {}, "tasks/manipulation/franka_lift_cable.jpg"],
            ["Isaac-Lift-Cable-Franka-Camera", "rsl_rl", "newton_mjwarp_vbd_proxy", "isaacsim_rtx,newton_renderer,ovrtx", "ik,joint", {}, "tasks/manipulation/franka_lift_cable.jpg"],
            ["Isaac-Lift-Cloth-Franka", "rsl_rl", "isaacsim_physx,newton_mjwarp_vbd_proxy", "", "ik,joint", {}, "tasks/manipulation/franka_lift_cloth.jpg"],
            ["Isaac-Lift-Cloth-Franka-Camera", "rsl_rl", "isaacsim_physx,newton_mjwarp_vbd_proxy", "isaacsim_rtx,newton_renderer,ovrtx", "ik,joint", {}, "tasks/manipulation/franka_lift_cloth.jpg"],
            ["Isaac-Lift-Franka", "rsl_rl", "isaacsim_physx,newton_mjwarp", "", "cube,shapes"],
            ["Isaac-Lift-KukaAllegro", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "cube,shapes", {}, "tasks/manipulation/kuka_allegro_lift.jpg"],
            ["Isaac-Lift-KukaAllegro-Camera", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo128,albedo256,albedo64,cube,depth128,depth256,depth64,duo_camera,raycaster_depth128,raycaster_depth256,raycaster_depth64,rgb128,rgb256,rgb64,semantic_segmentation128,semantic_segmentation256,semantic_segmentation64,shapes,simple_shading_constant_diffuse128,simple_shading_constant_diffuse256,simple_shading_constant_diffuse64,simple_shading_diffuse_mdl128,simple_shading_diffuse_mdl256,simple_shading_diffuse_mdl64,simple_shading_full_mdl128,simple_shading_full_mdl256,simple_shading_full_mdl64,single_camera", {}, "tasks/manipulation/kuka_allegro_lift.jpg"],
            ["Isaac-Lift-Soft-Franka", "rsl_rl", "isaacsim_physx,newton_mjwarp_vbd_proxy", "", "ik,joint", {}, "newton/franka-mjwarp-vbd-coupling.png"],
            ["Isaac-Lift-Soft-Franka-Camera", "rsl_rl", "isaacsim_physx,newton_mjwarp_vbd_proxy", "isaacsim_rtx,newton_renderer,ovrtx", "ik,joint", {}, "newton/franka-mjwarp-vbd-coupling.png"],
            ["Isaac-Open-Drawer-Franka-Direct", "rl_games,rsl_rl,skrl", "", "", "", {}, "tasks/manipulation/franka_open_drawer.jpg"],
            ["Isaac-Open-Drawer-Franka", "rl_games,rsl_rl,skrl", "", "", "", {}, "tasks/manipulation/franka_open_drawer.jpg"],
            ["Isaac-Pendulum-MARL-Direct", "rl_games,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", ""],
            ["Isaac-Reach-Franka", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "diffik,diffik_abs,joint_pos,newton_ik", {}, "tasks/manipulation/franka_reach.jpg"],
            ["Isaac-Reach-Franka-OSC", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "diffik_abs", {}, "tasks/manipulation/franka_reach.jpg"],
            ["Isaac-Reach-UR10", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", {}, "tasks/manipulation/ur10_reach.jpg"],
            ["Isaac-Reorient-Cube-Allegro-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", {}, "tasks/manipulation/allegro_cube.jpg"],
            ["Isaac-Reorient-Cube-Allegro", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "randomized,reset_only", {}, "tasks/manipulation/allegro_cube.jpg"],
            ["Isaac-Reorient-Cube-Shadow-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", {}, "tasks/manipulation/shadow_cube.jpg"],
            ["Isaac-Reorient-Cube-Shadow", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "asymmetric,randomized"],
            ["Isaac-Reorient-Cube-Shadow-Camera-Direct", "rl_games,rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo,depth,full,rgb,semantic_segmentation,simple_shading_constant_diffuse,simple_shading_diffuse_mdl,simple_shading_full_mdl", {}, "tasks/manipulation/shadow_cube.jpg"],
            ["Isaac-Reorient-Cube-Shadow-Camera", "rl_games,rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo,depth,full,randomized,rgb,semantic_segmentation,simple_shading_constant_diffuse,simple_shading_diffuse_mdl,simple_shading_full_mdl"],
            ["Isaac-Reorient-Franka", "rsl_rl", "isaacsim_physx,newton_mjwarp", "", "cube,shapes"],
            ["Isaac-Reorient-KukaAllegro", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "cube,shapes", {}, "tasks/manipulation/kuka_allegro_reorient.jpg"],
            ["Isaac-Reorient-KukaAllegro-Camera", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo128,albedo256,albedo64,cube,depth128,depth256,depth64,duo_camera,raycaster_depth128,raycaster_depth256,raycaster_depth64,rgb128,rgb256,rgb64,semantic_segmentation128,semantic_segmentation256,semantic_segmentation64,shapes,simple_shading_constant_diffuse128,simple_shading_constant_diffuse256,simple_shading_constant_diffuse64,simple_shading_diffuse_mdl128,simple_shading_diffuse_mdl256,simple_shading_diffuse_mdl64,simple_shading_full_mdl128,simple_shading_full_mdl256,simple_shading_full_mdl64,single_camera", {}, "tasks/manipulation/kuka_allegro_reorient.jpg"],
            ["Isaac-Shadow-Handover-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", {}, "tasks/manipulation/shadow_hand_over.jpg"],
            ["Isaac-Shadow-Handover", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "randomized"],
            ["Isaac-Velocity-Flat-AnymalD", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", {}, "tasks/locomotion/anymal_d_flat.jpg"],
            ["Isaac-Velocity-Flat-Cassie", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", ""],
            ["Isaac-Velocity-Flat-G1", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", {}, "tasks/locomotion/g1_flat.jpg"],
            ["Isaac-Velocity-Flat-H1", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", {}, "tasks/locomotion/h1_flat.jpg"],
            ["Isaac-Velocity-Flat-UnitreeGo2", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", {}, "tasks/locomotion/go2_flat.jpg"],
            ["Isaac-Velocity-Rough-AnymalD", "rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", {}, "tasks/locomotion/anymal_d_rough.jpg"],
            ["Isaac-Velocity-Rough-Cassie", "rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", {}, "tasks/locomotion/anymal_d_rough.jpg"],
            ["Isaac-Velocity-Rough-G1", "rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", {}, "tasks/locomotion/g1_rough.jpg"],
            ["Isaac-Velocity-Rough-H1", "rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", {}, "tasks/locomotion/h1_rough.jpg"],
            ["Isaac-Velocity-Rough-UnitreeGo2", "rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", {}, "tasks/locomotion/go2_rough.jpg"],
            ["IsaacContrib-Assemble-Trocar-G129-Dex3", "rlinf", "", "", "", {}, "tasks/manipulation/g1_assemble_trocar.jpg"],
            ["IsaacContrib-AutoMate-Assembly-Direct", "rl_games", "", "", "", {}, "tasks/automate/00004.jpg"],
            ["IsaacContrib-AutoMate-Disassembly-Direct", "rl_games", "", "", "", {}, "tasks/automate/01053_disassembly.jpg"],
            ["IsaacContrib-Cartpole-Camera-Showcase-Direct", "skrl", "", "", "box_box,box_discrete,box_multidiscrete,dict_box,dict_discrete,dict_multidiscrete,tuple_box,tuple_discrete,tuple_multidiscrete", {"skrl_box_box_cfg_entry_point": ["box_box"], "skrl_box_discrete_cfg_entry_point": ["box_discrete"], "skrl_box_multidiscrete_cfg_entry_point": ["box_multidiscrete"], "skrl_cfg_entry_point": ["box_box"], "skrl_dict_box_cfg_entry_point": ["dict_box"], "skrl_dict_discrete_cfg_entry_point": ["dict_discrete"], "skrl_dict_multidiscrete_cfg_entry_point": ["dict_multidiscrete"], "skrl_tuple_box_cfg_entry_point": ["tuple_box"], "skrl_tuple_discrete_cfg_entry_point": ["tuple_discrete"], "skrl_tuple_multidiscrete_cfg_entry_point": ["tuple_multidiscrete"]}],
            ["IsaacContrib-Cartpole-Showcase-Direct", "skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "box_box,box_discrete,box_multidiscrete,dict_box,dict_discrete,dict_multidiscrete,discrete_box,discrete_discrete,discrete_multidiscrete,multidiscrete_box,multidiscrete_discrete,multidiscrete_multidiscrete,tuple_box,tuple_discrete,tuple_multidiscrete", {"skrl_box_box_cfg_entry_point": ["box_box"], "skrl_box_discrete_cfg_entry_point": ["box_discrete"], "skrl_box_multidiscrete_cfg_entry_point": ["box_multidiscrete"], "skrl_cfg_entry_point": ["box_box"], "skrl_dict_box_cfg_entry_point": ["dict_box"], "skrl_dict_discrete_cfg_entry_point": ["dict_discrete"], "skrl_dict_multidiscrete_cfg_entry_point": ["dict_multidiscrete"], "skrl_discrete_box_cfg_entry_point": ["discrete_box"], "skrl_discrete_discrete_cfg_entry_point": ["discrete_discrete"], "skrl_discrete_multidiscrete_cfg_entry_point": ["discrete_multidiscrete"], "skrl_multidiscrete_box_cfg_entry_point": ["multidiscrete_box"], "skrl_multidiscrete_discrete_cfg_entry_point": ["multidiscrete_discrete"], "skrl_multidiscrete_multidiscrete_cfg_entry_point": ["multidiscrete_multidiscrete"], "skrl_tuple_box_cfg_entry_point": ["tuple_box"], "skrl_tuple_discrete_cfg_entry_point": ["tuple_discrete"], "skrl_tuple_multidiscrete_cfg_entry_point": ["tuple_multidiscrete"]}],
            ["IsaacContrib-Deploy-GearAssembly-Rizon4s-Grav", "rsl_rl", "", "", ""],
            ["IsaacContrib-Deploy-GearAssembly-Rizon4s-Grav-ROS-Inference", "rsl_rl", "", "", ""],
            ["IsaacContrib-Deploy-GearAssembly-UR10e-2F140", "rsl_rl", "", "", ""],
            ["IsaacContrib-Deploy-GearAssembly-UR10e-2F140-ROS-Inference", "rsl_rl", "", "", ""],
            ["IsaacContrib-Deploy-GearAssembly-UR10e-2F85", "rsl_rl", "", "", ""],
            ["IsaacContrib-Deploy-GearAssembly-UR10e-2F85-ROS-Inference", "rsl_rl", "", "", ""],
            ["IsaacContrib-Deploy-Reach-Rizon4s", "rsl_rl", "", "", ""],
            ["IsaacContrib-Deploy-Reach-Rizon4s-ROS-Inference", "rsl_rl", "", "", ""],
            ["IsaacContrib-Deploy-Reach-UR10e", "rsl_rl", "", "", "", {}, "tasks/manipulation/ur10e_reach.jpg"],
            ["IsaacContrib-Deploy-Reach-UR10e-ROS-Inference", "rsl_rl", "", "", "", {}, "tasks/manipulation/ur10e_reach.jpg"],
            ["IsaacContrib-DrLegs-HoldPose", "rsl_rl", "", "", "", {}, "tasks/locomotion/dr_legs.jpg"],
            ["IsaacContrib-DrLegs-Walk", "rsl_rl", "", "", "", {}, "tasks/locomotion/dr_legs.jpg"],
            ["IsaacContrib-ExhaustPipe-GR1T2-Pink-IK-Abs", "", "", "", ""],
            ["IsaacContrib-Factory-Franka", "rsl_rl", "isaacsim_physx,newton_mjwarp", "", "accumulator,choice,gear_mesh_large,gear_mesh_medium,gear_mesh_small,nut_thread_m16,peg_insert_12mm,peg_insert_16mm,peg_insert_4mm,peg_insert_8mm,rod_insert_12mm,rod_insert_16mm,rod_insert_4mm,rod_insert_8mm"],
            ["IsaacContrib-Factory-GearMesh-Direct", "rl_games", "", "", "", {}, "tasks/factory/gear_mesh.jpg"],
            ["IsaacContrib-Factory-NutThread-Direct", "rl_games", "", "", "", {}, "tasks/factory/nut_thread.jpg"],
            ["IsaacContrib-Factory-PegInsert-Direct", "rl_games", "", "", "", {}, "tasks/factory/peg_insert.jpg"],
            ["IsaacContrib-Forge-GearMesh-Direct", "rl_games", "", "", "", {}, "tasks/factory/gear_mesh.jpg"],
            ["IsaacContrib-Forge-NutThread-Direct", "rl_games", "", "", "", {}, "tasks/factory/nut_thread.jpg"],
            ["IsaacContrib-Forge-PegInsert-Direct", "rl_games", "", "", "", {}, "tasks/factory/peg_insert.jpg"],
            ["IsaacContrib-Franka-Pour", "rsl_rl", "", "", "", {}, "tasks/manipulation/franka_pour.jpg"],
            ["IsaacContrib-Humanoid-AMP-Dance-Direct", "skrl", "", "", "", {}, "tasks/others/humanoid_amp.jpg"],
            ["IsaacContrib-Humanoid-AMP-Run-Direct", "skrl", "", "", "", {}, "tasks/others/humanoid_amp.jpg"],
            ["IsaacContrib-Humanoid-AMP-Walk-Direct", "skrl", "", "", "", {}, "tasks/others/humanoid_amp.jpg"],
            ["IsaacContrib-Keyboard-SO101", "rsl_rl", "isaacsim_physx,newton_mjwarp", "", ""],
            ["IsaacContrib-Lift-Cube-Franka", "rl_games,rsl_rl,skrl,sb3", "", "", "", {}, "tasks/manipulation/franka_lift.jpg"],
            ["IsaacContrib-Lift-Cube-Franka-IK-Abs", "", "", "", "", {}, "tasks/manipulation/franka_lift.jpg"],
            ["IsaacContrib-Lift-Cube-Franka-IK-Rel", "", "", "", "", {}, "tasks/manipulation/franka_lift.jpg"],
            ["IsaacContrib-Lift-Cube-OpenArm", "rl_games,rsl_rl", "", "", "", {}, "tasks/manipulation/openarm_uni_lift.jpg"],
            ["IsaacContrib-Navigation-3DObstacles-ARL-Robot-1", "rl_games,rsl_rl,skrl", "", "", "", {}, "tasks/drone_arl/arl_robot_1_navigation.jpg"],
            ["IsaacContrib-Navigation-Flat-AnymalC", "rsl_rl,skrl", "", "", "", {}, "tasks/navigation/anymal_c_nav.jpg"],
            ["IsaacContrib-NutPour-GR1T2-Pink-IK-Abs", "", "", "", ""],
            ["IsaacContrib-Open-Drawer-Franka-IK-Abs", "rsl_rl", "", "", ""],
            ["IsaacContrib-Open-Drawer-Franka-IK-Rel", "rsl_rl", "", "", ""],
            ["IsaacContrib-Open-Drawer-OpenArm", "rl_games,rsl_rl", "", "", "", {}, "tasks/manipulation/openarm_uni_open_drawer.jpg"],
            ["IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs", "", "", "", "", {}, "tasks/manipulation/g1_pick_place_fixed_base.jpg"],
            ["IsaacContrib-PickPlace-G1-InspireFTP-Abs", "", "", "", "", {}, "tasks/manipulation/g1_pick_place.jpg"],
            ["IsaacContrib-PickPlace-GR1T2-Abs", "", "", "isaacsim_rtx,newton_renderer,ovrtx", "", {}, "tasks/manipulation/gr-1_pick_place.jpg"],
            ["IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs", "", "", "", "", {}, "tasks/manipulation/gr-1_pick_place_waist.jpg"],
            ["IsaacContrib-PickPlace-Locomanipulation-G1-Abs", "", "", "isaacsim_rtx,newton_renderer,ovrtx", "", {}, "tasks/manipulation/g1_pick_place_locomanipulation.jpg"],
            ["IsaacContrib-Place-Mug-Agibot-Left-Arm-RmpFlow", "", "isaacsim_physx,newton_mjwarp", "", "", {}, "tasks/manipulation/agibot_place_mug.jpg"],
            ["IsaacContrib-Place-Toy2Box-Agibot-Right-Arm-RmpFlow", "", "isaacsim_physx,newton_mjwarp", "", "", {}, "tasks/manipulation/agibot_place_toy.jpg"],
            ["IsaacContrib-Reach-OpenArm", "rl_games,rsl_rl,skrl", "", "", "", {}, "tasks/manipulation/openarm_uni_reach.jpg"],
            ["IsaacContrib-Reach-OpenArmBi", "rl_games,rsl_rl", "", "", "", {}, "tasks/manipulation/openarm_bi_reach.jpg"],
            ["IsaacContrib-Reorient-Cube-Shadow-OpenAI-FF-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", {}, "tasks/manipulation/shadow_cube.jpg"],
            ["IsaacContrib-Reorient-Cube-Shadow-OpenAI-LSTM-Direct", "rl_games,rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", {}, "tasks/manipulation/shadow_cube.jpg"],
            ["IsaacContrib-Stack-Cube-Bin-Franka-IK-Rel-Mimic", "", "isaacsim_physx,newton_mjwarp", "", ""],
            ["IsaacContrib-Stack-Cube-BlueGreen-Franka-IK-Rel", "", "isaacsim_physx,newton_mjwarp", "", ""],
            ["IsaacContrib-Stack-Cube-BlueGreenRed-Franka-IK-Rel", "", "isaacsim_physx,newton_mjwarp", "", ""],
            ["IsaacContrib-Stack-Cube-Franka", "", "isaacsim_physx,newton_mjwarp", "", "", {}, "tasks/manipulation/franka_stack.jpg"],
            ["IsaacContrib-Stack-Cube-Franka-IK-Abs", "", "isaacsim_physx,newton_mjwarp", "", "", {}, "tasks/manipulation/franka_stack.jpg"],
            ["IsaacContrib-Stack-Cube-Franka-IK-Rel", "", "isaacsim_physx,newton_mjwarp", "", "", {}, "tasks/manipulation/franka_stack.jpg"],
            ["IsaacContrib-Stack-Cube-Franka-IK-Rel-Blueprint", "", "isaacsim_physx,newton_mjwarp", "", "", {}, "tasks/manipulation/franka_stack.jpg"],
            ["IsaacContrib-Stack-Cube-Franka-IK-Rel-Skillgen", "", "isaacsim_physx,newton_mjwarp", "", "", {}, "tasks/manipulation/franka_stack.jpg"],
            ["IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor", "", "isaacsim_physx,newton_mjwarp", "", "", {}, "tasks/manipulation/franka_stack.jpg"],
            ["IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos", "", "isaacsim_physx,newton_mjwarp", "", "", {}, "tasks/manipulation/franka_stack.jpg"],
            ["IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow", "", "isaacsim_physx,newton_mjwarp", "", "", {}, "tasks/manipulation/galbot_stack_cube.jpg"],
            ["IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor", "", "isaacsim_physx,newton_mjwarp", "isaacsim_rtx,newton_renderer,ovrtx", ""],
            ["IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-Joint-Position", "", "isaacsim_physx,newton_mjwarp", "isaacsim_rtx,newton_renderer,ovrtx", ""],
            ["IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-RmpFlow", "", "isaacsim_physx,newton_mjwarp", "isaacsim_rtx,newton_renderer,ovrtx", ""],
            ["IsaacContrib-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow", "", "isaacsim_physx,newton_mjwarp", "", ""],
            ["IsaacContrib-Stack-Cube-Instance-Randomize-Franka", "", "", "", ""],
            ["IsaacContrib-Stack-Cube-Instance-Randomize-Franka-IK-Rel", "", "", "", ""],
            ["IsaacContrib-Stack-Cube-RedGreen-Franka-IK-Rel", "", "isaacsim_physx,newton_mjwarp", "", ""],
            ["IsaacContrib-Stack-Cube-RedGreenBlue-Franka-IK-Rel", "", "isaacsim_physx,newton_mjwarp", "", ""],
            ["IsaacContrib-Stack-Cube-SO101-IK-Abs-v0", "", "isaacsim_physx,newton_mjwarp", "", ""],
            ["IsaacContrib-Stack-Cube-SO101-Joint-Teleop-v0", "", "isaacsim_physx,newton_mjwarp", "", ""],
            ["IsaacContrib-Stack-Cube-SO101-v0", "", "isaacsim_physx,newton_mjwarp", "", ""],
            ["IsaacContrib-Stack-Cube-UR10-Long-Suction-IK-Rel", "", "isaacsim_physx,newton_mjwarp", "", "", {}, "tasks/manipulation/ur10_stack_surface_gripper.jpg"],
            ["IsaacContrib-Stack-Cube-UR10-Short-Suction-IK-Rel", "", "isaacsim_physx,newton_mjwarp", "", "", {}, "tasks/manipulation/ur10_stack_surface_gripper.jpg"],
            ["IsaacContrib-TrackPositionNoObstacles-ARL-Robot-1", "rl_games,rsl_rl,skrl", "", "", "", {}, "tasks/drone_arl/arl_robot_1_track_position_state_based.jpg"],
            ["IsaacContrib-Tracking-LocoManip-Digit", "rsl_rl", "isaacsim_physx", "", "", {}, "tasks/locomotion/agility_digit_loco_manip.jpg"],
            ["IsaacContrib-UR10-Particle-Push", "rsl_rl", "", "", "", {}, "tasks/manipulation/ur10_particle_push.jpg"],
            ["IsaacContrib-Velocity-Flat-AnymalB", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", {}, "tasks/locomotion/anymal_b_flat.jpg"],
            ["IsaacContrib-Velocity-Flat-AnymalC-Direct", "rl_games,rsl_rl,skrl", "", "", "", {}, "tasks/locomotion/anymal_c_flat.jpg"],
            ["IsaacContrib-Velocity-Flat-AnymalC", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", {}, "tasks/locomotion/anymal_c_flat.jpg"],
            ["IsaacContrib-Velocity-Flat-Digit", "rsl_rl", "isaacsim_physx", "", "", {}, "tasks/locomotion/agility_digit_flat.jpg"],
            ["IsaacContrib-Velocity-Flat-Spot", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp", "", "", {}, "tasks/locomotion/spot_flat.jpg"],
            ["IsaacContrib-Velocity-Flat-UnitreeA1", "rsl_rl,skrl,sb3", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", {}, "tasks/locomotion/a1_flat.jpg"],
            ["IsaacContrib-Velocity-Flat-UnitreeGo1", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", {}, "tasks/locomotion/go1_flat.jpg"],
            ["IsaacContrib-Velocity-Rough-AnymalB", "rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", {}, "tasks/locomotion/anymal_b_rough.jpg"],
            ["IsaacContrib-Velocity-Rough-AnymalC-Direct", "rl_games,rsl_rl,skrl", "", "", "", {}, "tasks/locomotion/anymal_c_rough.jpg"],
            ["IsaacContrib-Velocity-Rough-AnymalC", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", {}, "tasks/locomotion/anymal_c_rough.jpg"],
            ["IsaacContrib-Velocity-Rough-Digit", "rsl_rl", "isaacsim_physx", "", "", {}, "tasks/locomotion/agility_digit_rough.jpg"],
            ["IsaacContrib-Velocity-Rough-UnitreeA1", "rsl_rl,skrl,sb3", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", {}, "tasks/locomotion/a1_rough.jpg"],
            ["IsaacContrib-Velocity-Rough-UnitreeGo1", "rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", {}, "tasks/locomotion/go1_rough.jpg"],
        ];
        // END-AUTO-GENERATED: environment-browser-task-rows

    const splitValues = (value) => value ? value.split(",") : [];
    const tasks = taskRows.map(([task, rl, physics, renderer, presets, agentPresetCompatibility = {}, previewImage = ""]) => ({
        task,
        scope: task.startsWith("IsaacContrib-") ? "contrib" : "core",
        rl: splitValues(rl),
        physics: splitValues(physics),
        renderer: splitValues(renderer),
        presets: splitValues(presets),
        agentPresetCompatibility,
        previewImage,
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
    const copyButton = builder.querySelector("[data-copy-command]");
    const copyStatus = builder.querySelector("[data-copy-status]");
    const modeButtons = [...builder.querySelectorAll("[data-command-mode]")];
    const taskList = taskBrowser.querySelector("[data-task-list]");
    const taskSearch = taskBrowser.querySelector("[data-task-search]");
    const taskCategory = taskBrowser.querySelector("[data-task-category]");
    const taskCount = taskBrowser.querySelector("[data-task-count]");
    const taskEmpty = taskBrowser.querySelector("[data-task-empty]");
    const state = {
        mode: "train",
        task: "Isaac-Cartpole",
        benchmarkWorkload: "runtime",
    };
    let benchmarkRows = [];

    const categoryFor = (task) => {
        if (/Velocity|Navigation|TrackPosition|Locomanip|Humanoid/.test(task)) {
            return "locomotion";
        }
        if (/Lift|Reach|Reorient|Drawer|Handover|Keyboard|Assemble|Assembly|Factory|Forge|Stack|PickPlace|Place|Particle|Cabinet|Gear|Trocar/.test(task)) {
            return "manipulation";
        }
        return "classic";
    };

    const preferredValue = (values, preferred) => preferred.find((value) => values.includes(value)) || values[0] || "";

    const populateSelect = (select, values, preferred) => {
        const choices = values.length ? values : [""];
        select.replaceChildren(...choices.map((value) => new Option(value || "default", value)));
        select.value = preferredValue(values, preferred);
        select.disabled = values.length === 0;
    };

    const selectedTask = () => tasks.find((task) => task.task === state.task) || tasks[0];

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
            [/Keyboard-SO101/, "tasks/manipulation/so101_keyboard.jpg"],
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
        populateSelect(fields.physics, task.physics, [fields.physics.value, "newton_mjwarp", "isaacsim_physx", "ovphysx", "newton_kamino"]);
        const preferredRenderer = fields.physics.value.startsWith("newton") ? "newton_renderer" : "isaacsim_rtx";
        populateSelect(fields.renderer, task.renderer, [fields.renderer.value, preferredRenderer, "ovrtx"]);
        populateSelect(fields.presets, task.presets, [fields.presets.value, "joint", "ik", "rgb", "cube", "single_camera"]);
    };

    const updateModeControls = () => {
        const supportsPretrainedCheckpoint = state.mode === "play";
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

        const parts = ["uv", "run"];
        for (const extra of extras) {
            parts.push("--extra", extra);
        }
        parts.push("isaaclab", state.mode);
        if (fields.rl.value) {
            parts.push("--rl_library", fields.rl.value);
        }
        parts.push("--task", state.task);
        const task = selectedTask();
        const selectedAgent = Object.entries(task.agentPresetCompatibility).find(([agent, presets]) => (
            agent.startsWith(`${fields.rl.value}_`) && presets.includes(fields.presets.value)
        ))?.[0];
        if (selectedAgent && selectedAgent !== `${fields.rl.value}_cfg_entry_point`) {
            parts.push("--agent", selectedAgent);
        }
        if (state.task.includes("-Warp")) {
            parts.push("--frontend", "warp");
        }
        for (const selector of ["physics", "renderer", "presets"]) {
            if (fields[selector].value) {
                parts.push(`${selector}=${fields[selector].value}`);
            }
        }
        if (fields.checkpoint.checked) {
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
        preview.querySelector("[data-preview-mode]").textContent = state.mode === "train" ? "Train" : "Play";
        preview.querySelector("[data-preview-rl]").textContent = fields.rl.value || "Default";
        preview.querySelector("[data-preview-physics]").textContent = fields.physics.value || "Default";
        preview.querySelector("[data-preview-renderer]").textContent = fields.renderer.value || "Default";
        preview.querySelector("[data-preview-presets]").textContent = fields.presets.value || "Default";
        const latestVramRows = benchmarkRows
            .filter((row) => row.task === state.task && row.physics_backend === fields.physics.value)
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
        commandOutput.textContent = currentCommand();
        updatePreview();
        for (const row of taskList.querySelectorAll("[data-task-name]")) {
            const isSelected = row.dataset.taskName === state.task;
            row.classList.toggle("is-selected", isSelected);
            row.setAttribute("aria-pressed", String(isSelected));
        }
    };

    const renderTasks = () => {
        const query = taskSearch.value.trim().toLowerCase();
        const category = taskCategory.value;
        const visibleTasks = tasks.filter((task) => {
            const matchesQuery = task.task.toLowerCase().includes(query);
            const matchesCategory = category === "all"
                || (category === "contrib" ? task.scope === "contrib" : task.scope === "core" && categoryFor(task.task) === category);
            return matchesQuery && matchesCategory;
        });
        taskList.replaceChildren(...visibleTasks.map((task) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "environment-task-row";
            button.dataset.taskName = task.task;
            const isSelected = task.task === state.task;
            button.classList.toggle("is-selected", isSelected);
            button.setAttribute("aria-pressed", String(isSelected));
            button.innerHTML = `<span class="environment-task-name"></span><span class="environment-task-meta"></span>`;
            button.querySelector(".environment-task-name").textContent = task.task;
            const meta = button.querySelector(".environment-task-meta");
            const workflow = task.task.includes("Direct") ? "Direct" : "Manager based";
            meta.replaceChildren(...[workflow, `${task.rl.length} RL ${task.rl.length === 1 ? "library" : "libraries"}`].map((label) => {
                const badge = document.createElement("span");
                badge.textContent = label;
                return badge;
            }));
            button.addEventListener("click", () => {
                state.task = task.task;
                updateSelection();
            });
            return button;
        }));
        taskCount.textContent = `${visibleTasks.length} ${visibleTasks.length === 1 ? "task" : "tasks"}`;
        taskEmpty.hidden = visibleTasks.length !== 0;
        taskList.hidden = visibleTasks.length === 0;
    };

    const initializeTasks = () => {
        fields.task.replaceChildren(...tasks.map((task) => new Option(task.task, task.task)));
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
        isaacsim_physx: "Isaac Sim PhysX",
        newton_kamino: "Newton Kamino",
        newton_mjwarp: "Newton MJWarp",
        ovphysx: "OV PhysX",
    };
    const backendOrder = Object.keys(backendLabels);
    const backendClass = (backend) => `environment-chart-backend-${backend.replaceAll("_", "-")}`;

    const renderBenchmarkChart = (rows, maximum) => {
        const namespace = "http://www.w3.org/2000/svg";
        const createSvgElement = (name, attributes = {}) => {
            const element = document.createElementNS(namespace, name);
            for (const [key, value] of Object.entries(attributes)) {
                element.setAttribute(key, value);
            }
            return element;
        };
        const svg = createSvgElement("svg", {viewBox: "0 0 900 400", role: "img"});
        const workloadLabel = state.benchmarkWorkload === "runtime" ? "collection" : "training";
        svg.setAttribute("aria-label", `${state.task} ${workloadLabel} throughput history in frames per second`);
        const width = 900;
        const height = 400;
        const margins = {top: 38, right: 38, bottom: 60, left: 82};
        const plotWidth = width - margins.left - margins.right;
        const plotHeight = height - margins.top - margins.bottom;
        const benchmarkDate = (row) => (row.benchmark_date_utc || row.recorded_at_utc).slice(0, 10);
        const dateKeys = [...new Set(rows.map(benchmarkDate))].sort();
        const latestBySeriesAndDate = new Map();
        for (const row of rows) {
            const key = `${row.physics_backend}:${benchmarkDate(row)}`;
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
        axisTitle.textContent = "Total FPS";
        svg.appendChild(axisTitle);

        const visibleDateIndexes = dateKeys.length <= 4
            ? dateKeys.map((_date, index) => index)
            : [0, Math.round((dateKeys.length - 1) / 3), Math.round(2 * (dateKeys.length - 1) / 3), dateKeys.length - 1];
        for (const index of [...new Set(visibleDateIndexes)]) {
            const date = dateKeys[index];
            const label = createSvgElement("text", {
                x: xPosition(date), y: height - 15, class: "environment-chart-axis-label", "text-anchor": "middle",
            });
            label.textContent = new Date(`${date}T00:00:00Z`).toLocaleDateString(undefined, {
                month: "short", day: "numeric", year: "numeric", timeZone: "UTC",
            });
            svg.appendChild(label);
        }

        const backends = [...new Set(plottedRows.map((row) => row.physics_backend))]
            .sort((left, right) => backendOrder.indexOf(left) - backendOrder.indexOf(right));
        for (const backend of backends) {
            const series = plottedRows
                .filter((row) => row.physics_backend === backend)
                .sort((left, right) => benchmarkDate(left).localeCompare(benchmarkDate(right)));
            const seriesClass = backendClass(backend);
            if (series.length > 1) {
                const points = series.map((row) => {
                    const date = benchmarkDate(row);
                    return `${xPosition(date)},${yPosition(Number(row.total_fps_mean))}`;
                }).join(" ");
                svg.appendChild(createSvgElement("polyline", {
                    points, class: `environment-chart-line ${seriesClass}`,
                }));
            }
            for (const [index, row] of series.entries()) {
                const date = benchmarkDate(row);
                const value = Number(row.total_fps_mean);
                const x = xPosition(date);
                const y = yPosition(value);
                const circle = createSvgElement("circle", {
                    cx: x, cy: y, r: 5, class: `environment-chart-point ${seriesClass}`,
                });
                const title = createSvgElement("title");
                const tooltipWorkload = state.benchmarkWorkload === "runtime" ? "Collection" : "Training";
                title.textContent = `${backendLabels[backend] || backend} · ${tooltipWorkload}: ${Math.round(value).toLocaleString()} FPS on ${date}`;
                circle.appendChild(title);
                svg.appendChild(circle);
                if (index === series.length - 1) {
                    const valueLabel = createSvgElement("text", {
                        x, y: y - 11, class: `environment-chart-value ${seriesClass}`, "text-anchor": "middle",
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
        const backends = [...new Set(rows.map((row) => row.physics_backend))]
            .sort((left, right) => backendOrder.indexOf(left) - backendOrder.indexOf(right));
        const entries = backends.map((backend) => {
            const entry = document.createElement("span");
            entry.innerHTML = `<i class="environment-legend-swatch ${backendClass(backend)}"></i>`;
            entry.append(backendLabels[backend] || backend);
            return entry;
        });
        legend.replaceChildren(...entries);
    };

    const updateBenchmark = () => {
        if (!benchmarks) {
            return;
        }
        const taskRows = benchmarkRows.filter((row) => row.task === state.task);
        const rows = taskRows.filter((row) => row.workload === state.benchmarkWorkload);
        const chart = benchmarks.querySelector("[data-benchmark-chart]");
        const empty = benchmarks.querySelector("[data-benchmark-empty]");
        const toolbar = benchmarks.querySelector(".environment-benchmark-toolbar");
        const maximum = standardFpsScale(Math.max(...taskRows.map((row) => Number(row.total_fps_mean))));
        chart.hidden = rows.length === 0;
        empty.hidden = rows.length !== 0;
        toolbar.hidden = taskRows.length === 0;
        renderBenchmarkLegend(rows);
        chart.replaceChildren(...(rows.length ? [renderBenchmarkChart(rows, maximum)] : []));
    };

    const renderBenchmarks = async () => {
        if (!benchmarks) {
            return;
        }
        try {
            const source = new URL(benchmarks.dataset.benchmarkSource, window.location.href);
            const response = await fetch(source);
            if (!response.ok) {
                throw new Error(`Benchmark request failed with ${response.status}`);
            }
            benchmarkRows = parseCsv(await response.text()).filter((row) => row.data_origin === "measured");
            updatePreview();
        } catch (error) {
            benchmarks.querySelector("[data-benchmark-chart]").hidden = true;
            benchmarks.querySelector("[data-benchmark-empty]").hidden = true;
            benchmarks.querySelector(".environment-benchmark-toolbar").hidden = true;
            benchmarks.querySelector("[data-benchmark-error]").hidden = false;
            console.error(error);
        }
    };

    fields.task.addEventListener("change", () => {
        state.task = fields.task.value;
        updateSelection();
    });
    for (const field of [fields.rl, fields.physics, fields.renderer, fields.presets, fields.checkpoint]) {
        field.addEventListener("change", () => {
            commandOutput.textContent = currentCommand();
            updatePreview();
        });
    }
    for (const button of modeButtons) {
        button.addEventListener("click", () => {
            state.mode = button.dataset.commandMode;
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
