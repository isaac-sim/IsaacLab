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
            ["Isaac-Ant-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "environment-previews/ant.jpg", true, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Ant", "rl_games,rsl_rl,skrl,sb3", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "environment-previews/ant.jpg", true, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Cartpole-Direct", "rl_games,rsl_rl,skrl,sb3,torchrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "environment-previews/cartpole.jpg", true, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Cartpole", "rl_games,rsl_rl,skrl,sb3,torchrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "environment-previews/cartpole.jpg", true, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Cartpole-Camera-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo,depth,rgb,semantic_segmentation,simple_shading_constant_diffuse,simple_shading_diffuse_mdl,simple_shading_full_mdl", "tasks/classic/cartpole.jpg", false, {"*": ["rgb"], "rl_games": ["depth"]}, {}, "", null, ["||", "||albedo", "||depth", "||rgb", "||semantic_segmentation", "|isaacsim_rtx|", "|isaacsim_rtx|albedo", "|isaacsim_rtx|depth", "|isaacsim_rtx|rgb", "|isaacsim_rtx|semantic_segmentation", "|isaacsim_rtx|simple_shading_constant_diffuse", "|isaacsim_rtx|simple_shading_diffuse_mdl", "|isaacsim_rtx|simple_shading_full_mdl", "|newton_renderer|", "|newton_renderer|albedo", "|newton_renderer|depth", "|newton_renderer|rgb", "|newton_renderer|semantic_segmentation", "|ovrtx|", "|ovrtx|albedo", "|ovrtx|depth", "|ovrtx|rgb", "|ovrtx|semantic_segmentation", "|ovrtx|simple_shading_constant_diffuse", "|ovrtx|simple_shading_diffuse_mdl", "|ovrtx|simple_shading_full_mdl", "isaacsim_physx||", "isaacsim_physx||albedo", "isaacsim_physx||depth", "isaacsim_physx||rgb", "isaacsim_physx||semantic_segmentation", "isaacsim_physx|isaacsim_rtx|", "isaacsim_physx|isaacsim_rtx|albedo", "isaacsim_physx|isaacsim_rtx|depth", "isaacsim_physx|isaacsim_rtx|rgb", "isaacsim_physx|isaacsim_rtx|semantic_segmentation", "isaacsim_physx|isaacsim_rtx|simple_shading_constant_diffuse", "isaacsim_physx|isaacsim_rtx|simple_shading_diffuse_mdl", "isaacsim_physx|isaacsim_rtx|simple_shading_full_mdl", "isaacsim_physx|newton_renderer|", "isaacsim_physx|newton_renderer|albedo", "isaacsim_physx|newton_renderer|depth", "isaacsim_physx|newton_renderer|rgb", "isaacsim_physx|newton_renderer|semantic_segmentation", "newton_kamino||", "newton_kamino||albedo", "newton_kamino||depth", "newton_kamino||rgb", "newton_kamino||semantic_segmentation", "newton_kamino|isaacsim_rtx|", "newton_kamino|isaacsim_rtx|albedo", "newton_kamino|isaacsim_rtx|depth", "newton_kamino|isaacsim_rtx|rgb", "newton_kamino|isaacsim_rtx|semantic_segmentation", "newton_kamino|isaacsim_rtx|simple_shading_constant_diffuse", "newton_kamino|isaacsim_rtx|simple_shading_diffuse_mdl", "newton_kamino|isaacsim_rtx|simple_shading_full_mdl", "newton_kamino|newton_renderer|", "newton_kamino|newton_renderer|albedo", "newton_kamino|newton_renderer|depth", "newton_kamino|newton_renderer|rgb", "newton_kamino|newton_renderer|semantic_segmentation", "newton_kamino|ovrtx|", "newton_kamino|ovrtx|albedo", "newton_kamino|ovrtx|depth", "newton_kamino|ovrtx|rgb", "newton_kamino|ovrtx|semantic_segmentation", "newton_kamino|ovrtx|simple_shading_constant_diffuse", "newton_kamino|ovrtx|simple_shading_diffuse_mdl", "newton_kamino|ovrtx|simple_shading_full_mdl", "newton_mjwarp||", "newton_mjwarp||albedo", "newton_mjwarp||depth", "newton_mjwarp||rgb", "newton_mjwarp||semantic_segmentation", "newton_mjwarp|isaacsim_rtx|", "newton_mjwarp|isaacsim_rtx|albedo", "newton_mjwarp|isaacsim_rtx|depth", "newton_mjwarp|isaacsim_rtx|rgb", "newton_mjwarp|isaacsim_rtx|semantic_segmentation", "newton_mjwarp|isaacsim_rtx|simple_shading_constant_diffuse", "newton_mjwarp|isaacsim_rtx|simple_shading_diffuse_mdl", "newton_mjwarp|isaacsim_rtx|simple_shading_full_mdl", "newton_mjwarp|newton_renderer|", "newton_mjwarp|newton_renderer|albedo", "newton_mjwarp|newton_renderer|depth", "newton_mjwarp|newton_renderer|rgb", "newton_mjwarp|newton_renderer|semantic_segmentation", "newton_mjwarp|ovrtx|", "newton_mjwarp|ovrtx|albedo", "newton_mjwarp|ovrtx|depth", "newton_mjwarp|ovrtx|rgb", "newton_mjwarp|ovrtx|semantic_segmentation", "newton_mjwarp|ovrtx|simple_shading_constant_diffuse", "newton_mjwarp|ovrtx|simple_shading_diffuse_mdl", "newton_mjwarp|ovrtx|simple_shading_full_mdl", "ovphysx||", "ovphysx||albedo", "ovphysx||depth", "ovphysx||rgb", "ovphysx||semantic_segmentation", "ovphysx|newton_renderer|", "ovphysx|newton_renderer|albedo", "ovphysx|newton_renderer|depth", "ovphysx|newton_renderer|rgb", "ovphysx|newton_renderer|semantic_segmentation", "ovphysx|ovrtx|", "ovphysx|ovrtx|albedo", "ovphysx|ovrtx|depth", "ovphysx|ovrtx|rgb", "ovphysx|ovrtx|semantic_segmentation", "ovphysx|ovrtx|simple_shading_constant_diffuse", "ovphysx|ovrtx|simple_shading_diffuse_mdl", "ovphysx|ovrtx|simple_shading_full_mdl"]],
            ["Isaac-Cartpole-Camera", "rl_games,rsl_rl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo,depth,resnet18,rgb,semantic_segmentation,simple_shading_constant_diffuse,simple_shading_diffuse_mdl,simple_shading_full_mdl,theia_tiny", "tasks/classic/cartpole.jpg", false, {"*": ["rgb"], "rsl_rl": ["resnet18", "theia_tiny"]}, {}, "", null, ["||", "||albedo", "||depth", "||resnet18", "||rgb", "||semantic_segmentation", "||theia_tiny", "|isaacsim_rtx|", "|isaacsim_rtx|albedo", "|isaacsim_rtx|depth", "|isaacsim_rtx|resnet18", "|isaacsim_rtx|rgb", "|isaacsim_rtx|semantic_segmentation", "|isaacsim_rtx|simple_shading_constant_diffuse", "|isaacsim_rtx|simple_shading_diffuse_mdl", "|isaacsim_rtx|simple_shading_full_mdl", "|isaacsim_rtx|theia_tiny", "|newton_renderer|", "|newton_renderer|albedo", "|newton_renderer|depth", "|newton_renderer|resnet18", "|newton_renderer|rgb", "|newton_renderer|semantic_segmentation", "|newton_renderer|theia_tiny", "|ovrtx|", "|ovrtx|albedo", "|ovrtx|depth", "|ovrtx|resnet18", "|ovrtx|rgb", "|ovrtx|semantic_segmentation", "|ovrtx|simple_shading_constant_diffuse", "|ovrtx|simple_shading_diffuse_mdl", "|ovrtx|simple_shading_full_mdl", "|ovrtx|theia_tiny", "isaacsim_physx||", "isaacsim_physx||albedo", "isaacsim_physx||depth", "isaacsim_physx||resnet18", "isaacsim_physx||rgb", "isaacsim_physx||semantic_segmentation", "isaacsim_physx||theia_tiny", "isaacsim_physx|isaacsim_rtx|", "isaacsim_physx|isaacsim_rtx|albedo", "isaacsim_physx|isaacsim_rtx|depth", "isaacsim_physx|isaacsim_rtx|resnet18", "isaacsim_physx|isaacsim_rtx|rgb", "isaacsim_physx|isaacsim_rtx|semantic_segmentation", "isaacsim_physx|isaacsim_rtx|simple_shading_constant_diffuse", "isaacsim_physx|isaacsim_rtx|simple_shading_diffuse_mdl", "isaacsim_physx|isaacsim_rtx|simple_shading_full_mdl", "isaacsim_physx|isaacsim_rtx|theia_tiny", "isaacsim_physx|newton_renderer|", "isaacsim_physx|newton_renderer|albedo", "isaacsim_physx|newton_renderer|depth", "isaacsim_physx|newton_renderer|resnet18", "isaacsim_physx|newton_renderer|rgb", "isaacsim_physx|newton_renderer|semantic_segmentation", "isaacsim_physx|newton_renderer|theia_tiny", "newton_kamino||", "newton_kamino||albedo", "newton_kamino||depth", "newton_kamino||resnet18", "newton_kamino||rgb", "newton_kamino||semantic_segmentation", "newton_kamino||theia_tiny", "newton_kamino|isaacsim_rtx|", "newton_kamino|isaacsim_rtx|albedo", "newton_kamino|isaacsim_rtx|depth", "newton_kamino|isaacsim_rtx|resnet18", "newton_kamino|isaacsim_rtx|rgb", "newton_kamino|isaacsim_rtx|semantic_segmentation", "newton_kamino|isaacsim_rtx|simple_shading_constant_diffuse", "newton_kamino|isaacsim_rtx|simple_shading_diffuse_mdl", "newton_kamino|isaacsim_rtx|simple_shading_full_mdl", "newton_kamino|isaacsim_rtx|theia_tiny", "newton_kamino|newton_renderer|", "newton_kamino|newton_renderer|albedo", "newton_kamino|newton_renderer|depth", "newton_kamino|newton_renderer|resnet18", "newton_kamino|newton_renderer|rgb", "newton_kamino|newton_renderer|semantic_segmentation", "newton_kamino|newton_renderer|theia_tiny", "newton_kamino|ovrtx|", "newton_kamino|ovrtx|albedo", "newton_kamino|ovrtx|depth", "newton_kamino|ovrtx|resnet18", "newton_kamino|ovrtx|rgb", "newton_kamino|ovrtx|semantic_segmentation", "newton_kamino|ovrtx|simple_shading_constant_diffuse", "newton_kamino|ovrtx|simple_shading_diffuse_mdl", "newton_kamino|ovrtx|simple_shading_full_mdl", "newton_kamino|ovrtx|theia_tiny", "newton_mjwarp||", "newton_mjwarp||albedo", "newton_mjwarp||depth", "newton_mjwarp||resnet18", "newton_mjwarp||rgb", "newton_mjwarp||semantic_segmentation", "newton_mjwarp||theia_tiny", "newton_mjwarp|isaacsim_rtx|", "newton_mjwarp|isaacsim_rtx|albedo", "newton_mjwarp|isaacsim_rtx|depth", "newton_mjwarp|isaacsim_rtx|resnet18", "newton_mjwarp|isaacsim_rtx|rgb", "newton_mjwarp|isaacsim_rtx|semantic_segmentation", "newton_mjwarp|isaacsim_rtx|simple_shading_constant_diffuse", "newton_mjwarp|isaacsim_rtx|simple_shading_diffuse_mdl", "newton_mjwarp|isaacsim_rtx|simple_shading_full_mdl", "newton_mjwarp|isaacsim_rtx|theia_tiny", "newton_mjwarp|newton_renderer|", "newton_mjwarp|newton_renderer|albedo", "newton_mjwarp|newton_renderer|depth", "newton_mjwarp|newton_renderer|resnet18", "newton_mjwarp|newton_renderer|rgb", "newton_mjwarp|newton_renderer|semantic_segmentation", "newton_mjwarp|newton_renderer|theia_tiny", "newton_mjwarp|ovrtx|", "newton_mjwarp|ovrtx|albedo", "newton_mjwarp|ovrtx|depth", "newton_mjwarp|ovrtx|resnet18", "newton_mjwarp|ovrtx|rgb", "newton_mjwarp|ovrtx|semantic_segmentation", "newton_mjwarp|ovrtx|simple_shading_constant_diffuse", "newton_mjwarp|ovrtx|simple_shading_diffuse_mdl", "newton_mjwarp|ovrtx|simple_shading_full_mdl", "newton_mjwarp|ovrtx|theia_tiny", "ovphysx||", "ovphysx||albedo", "ovphysx||depth", "ovphysx||resnet18", "ovphysx||rgb", "ovphysx||semantic_segmentation", "ovphysx||theia_tiny", "ovphysx|newton_renderer|", "ovphysx|newton_renderer|albedo", "ovphysx|newton_renderer|depth", "ovphysx|newton_renderer|resnet18", "ovphysx|newton_renderer|rgb", "ovphysx|newton_renderer|semantic_segmentation", "ovphysx|newton_renderer|theia_tiny", "ovphysx|ovrtx|", "ovphysx|ovrtx|albedo", "ovphysx|ovrtx|depth", "ovphysx|ovrtx|resnet18", "ovphysx|ovrtx|rgb", "ovphysx|ovrtx|semantic_segmentation", "ovphysx|ovrtx|simple_shading_constant_diffuse", "ovphysx|ovrtx|simple_shading_diffuse_mdl", "ovphysx|ovrtx|simple_shading_full_mdl", "ovphysx|ovrtx|theia_tiny"]],
            ["Isaac-Fourbar-Pole-Swingup", "rsl_rl", "newton_kamino", "", "", "environment-previews/fourbar_pole.jpg", false, {}, {}, "", null, ["||", "newton_kamino||"]],
            ["Isaac-Humanoid-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "environment-previews/humanoid.jpg", true, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Humanoid", "rl_games,rsl_rl,skrl,sb3", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "environment-previews/humanoid.jpg", true, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Lift-Cable-Franka", "rsl_rl", "newton_mjwarp_vbd_proxy", "", "ik,joint", "environment-previews/franka_lift_cable.jpg", false, {"*": ["joint"]}, {}, "", null, ["||", "||ik", "||joint", "newton_mjwarp_vbd_proxy||", "newton_mjwarp_vbd_proxy||ik", "newton_mjwarp_vbd_proxy||joint"]],
            ["Isaac-Lift-Cable-Franka-Camera", "rsl_rl", "newton_mjwarp_vbd_proxy", "isaacsim_rtx,newton_renderer,ovrtx", "ik,joint", "environment-previews/franka_lift_cable.jpg", false, {"*": ["joint"]}, {}, "", null, ["||", "||ik", "||joint", "|isaacsim_rtx|", "|isaacsim_rtx|ik", "|isaacsim_rtx|joint", "|newton_renderer|", "|newton_renderer|ik", "|newton_renderer|joint", "|ovrtx|", "|ovrtx|ik", "|ovrtx|joint", "newton_mjwarp_vbd_proxy||", "newton_mjwarp_vbd_proxy||ik", "newton_mjwarp_vbd_proxy||joint", "newton_mjwarp_vbd_proxy|isaacsim_rtx|", "newton_mjwarp_vbd_proxy|isaacsim_rtx|ik", "newton_mjwarp_vbd_proxy|isaacsim_rtx|joint", "newton_mjwarp_vbd_proxy|newton_renderer|", "newton_mjwarp_vbd_proxy|newton_renderer|ik", "newton_mjwarp_vbd_proxy|newton_renderer|joint", "newton_mjwarp_vbd_proxy|ovrtx|", "newton_mjwarp_vbd_proxy|ovrtx|ik", "newton_mjwarp_vbd_proxy|ovrtx|joint"]],
            ["Isaac-Lift-Cloth-Franka", "rsl_rl", "isaacsim_physx,newton_mjwarp_vbd_proxy", "", "ik,joint", "environment-previews/franka_lift_cloth.jpg", false, {"*": ["joint"]}, {}, "", null, ["||", "||ik", "||joint", "isaacsim_physx||", "isaacsim_physx||ik", "isaacsim_physx||joint", "newton_mjwarp_vbd_proxy||", "newton_mjwarp_vbd_proxy||ik", "newton_mjwarp_vbd_proxy||joint"]],
            ["Isaac-Lift-Cloth-Franka-Camera", "rsl_rl", "isaacsim_physx,newton_mjwarp_vbd_proxy", "isaacsim_rtx,newton_renderer,ovrtx", "ik,joint", "environment-previews/franka_lift_cloth.jpg", false, {"*": ["joint"]}, {}, "", null, ["||", "||ik", "||joint", "|isaacsim_rtx|", "|isaacsim_rtx|ik", "|isaacsim_rtx|joint", "|newton_renderer|", "|newton_renderer|ik", "|newton_renderer|joint", "|ovrtx|", "|ovrtx|ik", "|ovrtx|joint", "isaacsim_physx||", "isaacsim_physx||ik", "isaacsim_physx||joint", "isaacsim_physx|isaacsim_rtx|", "isaacsim_physx|isaacsim_rtx|ik", "isaacsim_physx|isaacsim_rtx|joint", "isaacsim_physx|newton_renderer|", "isaacsim_physx|newton_renderer|ik", "isaacsim_physx|newton_renderer|joint", "newton_mjwarp_vbd_proxy||", "newton_mjwarp_vbd_proxy||ik", "newton_mjwarp_vbd_proxy||joint", "newton_mjwarp_vbd_proxy|isaacsim_rtx|", "newton_mjwarp_vbd_proxy|isaacsim_rtx|ik", "newton_mjwarp_vbd_proxy|isaacsim_rtx|joint", "newton_mjwarp_vbd_proxy|newton_renderer|", "newton_mjwarp_vbd_proxy|newton_renderer|ik", "newton_mjwarp_vbd_proxy|newton_renderer|joint", "newton_mjwarp_vbd_proxy|ovrtx|", "newton_mjwarp_vbd_proxy|ovrtx|ik", "newton_mjwarp_vbd_proxy|ovrtx|joint"]],
            ["Isaac-Lift-Franka", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "cube,minimal,shapes", "environment-previews/franka_lift.jpg", false, {"*": ["shapes"]}, {}, "", null, ["||", "||cube", "||minimal", "||shapes", "isaacsim_physx||", "isaacsim_physx||cube", "isaacsim_physx||minimal", "isaacsim_physx||shapes", "newton_mjwarp||", "newton_mjwarp||cube", "newton_mjwarp||minimal", "newton_mjwarp||shapes", "ovphysx||", "ovphysx||cube", "ovphysx||minimal", "ovphysx||shapes"]],
            ["Isaac-Lift-KukaAllegro", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "cube,shapes", "environment-previews/kuka_allegro_lift.jpg", false, {"*": ["shapes"]}, {}, "", null, ["||", "||cube", "||shapes", "isaacsim_physx||", "isaacsim_physx||cube", "isaacsim_physx||shapes", "newton_mjwarp||", "newton_mjwarp||cube", "newton_mjwarp||shapes", "ovphysx||", "ovphysx||cube", "ovphysx||shapes"]],
            ["Isaac-Lift-KukaAllegro-Camera", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo128,albedo256,albedo64,cube,depth128,depth256,depth64,duo_camera,raycaster_depth128,raycaster_depth256,raycaster_depth64,rgb128,rgb256,rgb64,semantic_segmentation128,semantic_segmentation256,semantic_segmentation64,shapes,simple_shading_constant_diffuse128,simple_shading_constant_diffuse256,simple_shading_constant_diffuse64,simple_shading_diffuse_mdl128,simple_shading_diffuse_mdl256,simple_shading_diffuse_mdl64,simple_shading_full_mdl128,simple_shading_full_mdl256,simple_shading_full_mdl64,single_camera", "environment-previews/kuka_allegro_lift.jpg", false, {"*": ["rgb64", "shapes", "single_camera"]}, {}, "", null, ["||", "||albedo128", "||albedo256", "||albedo64", "||cube", "||depth128", "||depth256", "||depth64", "||duo_camera", "||raycaster_depth128", "||raycaster_depth256", "||raycaster_depth64", "||rgb128", "||rgb256", "||rgb64", "||semantic_segmentation128", "||semantic_segmentation256", "||semantic_segmentation64", "||shapes", "||single_camera", "|isaacsim_rtx|", "|isaacsim_rtx|albedo128", "|isaacsim_rtx|albedo256", "|isaacsim_rtx|albedo64", "|isaacsim_rtx|cube", "|isaacsim_rtx|depth128", "|isaacsim_rtx|depth256", "|isaacsim_rtx|depth64", "|isaacsim_rtx|duo_camera", "|isaacsim_rtx|rgb128", "|isaacsim_rtx|rgb256", "|isaacsim_rtx|rgb64", "|isaacsim_rtx|semantic_segmentation128", "|isaacsim_rtx|semantic_segmentation256", "|isaacsim_rtx|semantic_segmentation64", "|isaacsim_rtx|shapes", "|isaacsim_rtx|simple_shading_constant_diffuse128", "|isaacsim_rtx|simple_shading_constant_diffuse256", "|isaacsim_rtx|simple_shading_constant_diffuse64", "|isaacsim_rtx|simple_shading_diffuse_mdl128", "|isaacsim_rtx|simple_shading_diffuse_mdl256", "|isaacsim_rtx|simple_shading_diffuse_mdl64", "|isaacsim_rtx|simple_shading_full_mdl128", "|isaacsim_rtx|simple_shading_full_mdl256", "|isaacsim_rtx|simple_shading_full_mdl64", "|isaacsim_rtx|single_camera", "|newton_renderer|", "|newton_renderer|albedo128", "|newton_renderer|albedo256", "|newton_renderer|albedo64", "|newton_renderer|cube", "|newton_renderer|depth128", "|newton_renderer|depth256", "|newton_renderer|depth64", "|newton_renderer|duo_camera", "|newton_renderer|rgb128", "|newton_renderer|rgb256", "|newton_renderer|rgb64", "|newton_renderer|semantic_segmentation128", "|newton_renderer|semantic_segmentation256", "|newton_renderer|semantic_segmentation64", "|newton_renderer|shapes", "|newton_renderer|single_camera", "|ovrtx|", "|ovrtx|albedo128", "|ovrtx|albedo256", "|ovrtx|albedo64", "|ovrtx|cube", "|ovrtx|depth128", "|ovrtx|depth256", "|ovrtx|depth64", "|ovrtx|duo_camera", "|ovrtx|rgb128", "|ovrtx|rgb256", "|ovrtx|rgb64", "|ovrtx|semantic_segmentation128", "|ovrtx|semantic_segmentation256", "|ovrtx|semantic_segmentation64", "|ovrtx|shapes", "|ovrtx|simple_shading_constant_diffuse128", "|ovrtx|simple_shading_constant_diffuse256", "|ovrtx|simple_shading_constant_diffuse64", "|ovrtx|simple_shading_diffuse_mdl128", "|ovrtx|simple_shading_diffuse_mdl256", "|ovrtx|simple_shading_diffuse_mdl64", "|ovrtx|simple_shading_full_mdl128", "|ovrtx|simple_shading_full_mdl256", "|ovrtx|simple_shading_full_mdl64", "|ovrtx|single_camera", "isaacsim_physx||", "isaacsim_physx||albedo128", "isaacsim_physx||albedo256", "isaacsim_physx||albedo64", "isaacsim_physx||cube", "isaacsim_physx||depth128", "isaacsim_physx||depth256", "isaacsim_physx||depth64", "isaacsim_physx||duo_camera", "isaacsim_physx||raycaster_depth128", "isaacsim_physx||raycaster_depth256", "isaacsim_physx||raycaster_depth64", "isaacsim_physx||rgb128", "isaacsim_physx||rgb256", "isaacsim_physx||rgb64", "isaacsim_physx||semantic_segmentation128", "isaacsim_physx||semantic_segmentation256", "isaacsim_physx||semantic_segmentation64", "isaacsim_physx||shapes", "isaacsim_physx||single_camera", "isaacsim_physx|isaacsim_rtx|", "isaacsim_physx|isaacsim_rtx|albedo128", "isaacsim_physx|isaacsim_rtx|albedo256", "isaacsim_physx|isaacsim_rtx|albedo64", "isaacsim_physx|isaacsim_rtx|cube", "isaacsim_physx|isaacsim_rtx|depth128", "isaacsim_physx|isaacsim_rtx|depth256", "isaacsim_physx|isaacsim_rtx|depth64", "isaacsim_physx|isaacsim_rtx|duo_camera", "isaacsim_physx|isaacsim_rtx|rgb128", "isaacsim_physx|isaacsim_rtx|rgb256", "isaacsim_physx|isaacsim_rtx|rgb64", "isaacsim_physx|isaacsim_rtx|semantic_segmentation128", "isaacsim_physx|isaacsim_rtx|semantic_segmentation256", "isaacsim_physx|isaacsim_rtx|semantic_segmentation64", "isaacsim_physx|isaacsim_rtx|shapes", "isaacsim_physx|isaacsim_rtx|simple_shading_constant_diffuse128", "isaacsim_physx|isaacsim_rtx|simple_shading_constant_diffuse256", "isaacsim_physx|isaacsim_rtx|simple_shading_constant_diffuse64", "isaacsim_physx|isaacsim_rtx|simple_shading_diffuse_mdl128", "isaacsim_physx|isaacsim_rtx|simple_shading_diffuse_mdl256", "isaacsim_physx|isaacsim_rtx|simple_shading_diffuse_mdl64", "isaacsim_physx|isaacsim_rtx|simple_shading_full_mdl128", "isaacsim_physx|isaacsim_rtx|simple_shading_full_mdl256", "isaacsim_physx|isaacsim_rtx|simple_shading_full_mdl64", "isaacsim_physx|isaacsim_rtx|single_camera", "isaacsim_physx|newton_renderer|", "isaacsim_physx|newton_renderer|albedo128", "isaacsim_physx|newton_renderer|albedo256", "isaacsim_physx|newton_renderer|albedo64", "isaacsim_physx|newton_renderer|cube", "isaacsim_physx|newton_renderer|depth128", "isaacsim_physx|newton_renderer|depth256", "isaacsim_physx|newton_renderer|depth64", "isaacsim_physx|newton_renderer|duo_camera", "isaacsim_physx|newton_renderer|rgb128", "isaacsim_physx|newton_renderer|rgb256", "isaacsim_physx|newton_renderer|rgb64", "isaacsim_physx|newton_renderer|semantic_segmentation128", "isaacsim_physx|newton_renderer|semantic_segmentation256", "isaacsim_physx|newton_renderer|semantic_segmentation64", "isaacsim_physx|newton_renderer|shapes", "isaacsim_physx|newton_renderer|single_camera", "newton_mjwarp||", "newton_mjwarp||albedo128", "newton_mjwarp||albedo256", "newton_mjwarp||albedo64", "newton_mjwarp||cube", "newton_mjwarp||depth128", "newton_mjwarp||depth256", "newton_mjwarp||depth64", "newton_mjwarp||duo_camera", "newton_mjwarp||raycaster_depth128", "newton_mjwarp||raycaster_depth256", "newton_mjwarp||raycaster_depth64", "newton_mjwarp||rgb128", "newton_mjwarp||rgb256", "newton_mjwarp||rgb64", "newton_mjwarp||semantic_segmentation128", "newton_mjwarp||semantic_segmentation256", "newton_mjwarp||semantic_segmentation64", "newton_mjwarp||shapes", "newton_mjwarp||single_camera", "newton_mjwarp|isaacsim_rtx|", "newton_mjwarp|isaacsim_rtx|albedo128", "newton_mjwarp|isaacsim_rtx|albedo256", "newton_mjwarp|isaacsim_rtx|albedo64", "newton_mjwarp|isaacsim_rtx|cube", "newton_mjwarp|isaacsim_rtx|depth128", "newton_mjwarp|isaacsim_rtx|depth256", "newton_mjwarp|isaacsim_rtx|depth64", "newton_mjwarp|isaacsim_rtx|duo_camera", "newton_mjwarp|isaacsim_rtx|rgb128", "newton_mjwarp|isaacsim_rtx|rgb256", "newton_mjwarp|isaacsim_rtx|rgb64", "newton_mjwarp|isaacsim_rtx|semantic_segmentation128", "newton_mjwarp|isaacsim_rtx|semantic_segmentation256", "newton_mjwarp|isaacsim_rtx|semantic_segmentation64", "newton_mjwarp|isaacsim_rtx|shapes", "newton_mjwarp|isaacsim_rtx|simple_shading_constant_diffuse128", "newton_mjwarp|isaacsim_rtx|simple_shading_constant_diffuse256", "newton_mjwarp|isaacsim_rtx|simple_shading_constant_diffuse64", "newton_mjwarp|isaacsim_rtx|simple_shading_diffuse_mdl128", "newton_mjwarp|isaacsim_rtx|simple_shading_diffuse_mdl256", "newton_mjwarp|isaacsim_rtx|simple_shading_diffuse_mdl64", "newton_mjwarp|isaacsim_rtx|simple_shading_full_mdl128", "newton_mjwarp|isaacsim_rtx|simple_shading_full_mdl256", "newton_mjwarp|isaacsim_rtx|simple_shading_full_mdl64", "newton_mjwarp|isaacsim_rtx|single_camera", "newton_mjwarp|newton_renderer|", "newton_mjwarp|newton_renderer|albedo128", "newton_mjwarp|newton_renderer|albedo256", "newton_mjwarp|newton_renderer|albedo64", "newton_mjwarp|newton_renderer|cube", "newton_mjwarp|newton_renderer|depth128", "newton_mjwarp|newton_renderer|depth256", "newton_mjwarp|newton_renderer|depth64", "newton_mjwarp|newton_renderer|duo_camera", "newton_mjwarp|newton_renderer|rgb128", "newton_mjwarp|newton_renderer|rgb256", "newton_mjwarp|newton_renderer|rgb64", "newton_mjwarp|newton_renderer|semantic_segmentation128", "newton_mjwarp|newton_renderer|semantic_segmentation256", "newton_mjwarp|newton_renderer|semantic_segmentation64", "newton_mjwarp|newton_renderer|shapes", "newton_mjwarp|newton_renderer|single_camera", "newton_mjwarp|ovrtx|", "newton_mjwarp|ovrtx|albedo128", "newton_mjwarp|ovrtx|albedo256", "newton_mjwarp|ovrtx|albedo64", "newton_mjwarp|ovrtx|cube", "newton_mjwarp|ovrtx|depth128", "newton_mjwarp|ovrtx|depth256", "newton_mjwarp|ovrtx|depth64", "newton_mjwarp|ovrtx|duo_camera", "newton_mjwarp|ovrtx|rgb128", "newton_mjwarp|ovrtx|rgb256", "newton_mjwarp|ovrtx|rgb64", "newton_mjwarp|ovrtx|semantic_segmentation128", "newton_mjwarp|ovrtx|semantic_segmentation256", "newton_mjwarp|ovrtx|semantic_segmentation64", "newton_mjwarp|ovrtx|shapes", "newton_mjwarp|ovrtx|simple_shading_constant_diffuse128", "newton_mjwarp|ovrtx|simple_shading_constant_diffuse256", "newton_mjwarp|ovrtx|simple_shading_constant_diffuse64", "newton_mjwarp|ovrtx|simple_shading_diffuse_mdl128", "newton_mjwarp|ovrtx|simple_shading_diffuse_mdl256", "newton_mjwarp|ovrtx|simple_shading_diffuse_mdl64", "newton_mjwarp|ovrtx|simple_shading_full_mdl128", "newton_mjwarp|ovrtx|simple_shading_full_mdl256", "newton_mjwarp|ovrtx|simple_shading_full_mdl64", "newton_mjwarp|ovrtx|single_camera", "ovphysx||", "ovphysx||albedo128", "ovphysx||albedo256", "ovphysx||albedo64", "ovphysx||cube", "ovphysx||depth128", "ovphysx||depth256", "ovphysx||depth64", "ovphysx||duo_camera", "ovphysx||raycaster_depth128", "ovphysx||raycaster_depth256", "ovphysx||raycaster_depth64", "ovphysx||rgb128", "ovphysx||rgb256", "ovphysx||rgb64", "ovphysx||semantic_segmentation128", "ovphysx||semantic_segmentation256", "ovphysx||semantic_segmentation64", "ovphysx||shapes", "ovphysx||single_camera", "ovphysx|newton_renderer|", "ovphysx|newton_renderer|albedo128", "ovphysx|newton_renderer|albedo256", "ovphysx|newton_renderer|albedo64", "ovphysx|newton_renderer|cube", "ovphysx|newton_renderer|depth128", "ovphysx|newton_renderer|depth256", "ovphysx|newton_renderer|depth64", "ovphysx|newton_renderer|duo_camera", "ovphysx|newton_renderer|rgb128", "ovphysx|newton_renderer|rgb256", "ovphysx|newton_renderer|rgb64", "ovphysx|newton_renderer|semantic_segmentation128", "ovphysx|newton_renderer|semantic_segmentation256", "ovphysx|newton_renderer|semantic_segmentation64", "ovphysx|newton_renderer|shapes", "ovphysx|newton_renderer|single_camera", "ovphysx|ovrtx|", "ovphysx|ovrtx|albedo128", "ovphysx|ovrtx|albedo256", "ovphysx|ovrtx|albedo64", "ovphysx|ovrtx|cube", "ovphysx|ovrtx|depth128", "ovphysx|ovrtx|depth256", "ovphysx|ovrtx|depth64", "ovphysx|ovrtx|duo_camera", "ovphysx|ovrtx|rgb128", "ovphysx|ovrtx|rgb256", "ovphysx|ovrtx|rgb64", "ovphysx|ovrtx|semantic_segmentation128", "ovphysx|ovrtx|semantic_segmentation256", "ovphysx|ovrtx|semantic_segmentation64", "ovphysx|ovrtx|shapes", "ovphysx|ovrtx|simple_shading_constant_diffuse128", "ovphysx|ovrtx|simple_shading_constant_diffuse256", "ovphysx|ovrtx|simple_shading_constant_diffuse64", "ovphysx|ovrtx|simple_shading_diffuse_mdl128", "ovphysx|ovrtx|simple_shading_diffuse_mdl256", "ovphysx|ovrtx|simple_shading_diffuse_mdl64", "ovphysx|ovrtx|simple_shading_full_mdl128", "ovphysx|ovrtx|simple_shading_full_mdl256", "ovphysx|ovrtx|simple_shading_full_mdl64", "ovphysx|ovrtx|single_camera"]],
            ["Isaac-Lift-Soft-Franka", "rsl_rl", "isaacsim_physx,newton_mjwarp_vbd_proxy", "", "ik,joint", "environment-previews/franka_lift_soft.jpg", false, {"*": ["joint"]}, {}, "tetrahedralization", null, ["||", "||ik", "||joint", "isaacsim_physx||", "isaacsim_physx||ik", "isaacsim_physx||joint", "newton_mjwarp_vbd_proxy||", "newton_mjwarp_vbd_proxy||ik", "newton_mjwarp_vbd_proxy||joint"]],
            ["Isaac-Lift-Soft-Franka-Camera", "rsl_rl", "isaacsim_physx,newton_mjwarp_vbd_proxy", "isaacsim_rtx,newton_renderer,ovrtx", "ik,joint", "environment-previews/franka_lift_soft.jpg", false, {"*": ["joint"]}, {}, "tetrahedralization", null, ["||", "||ik", "||joint", "|isaacsim_rtx|", "|isaacsim_rtx|ik", "|isaacsim_rtx|joint", "|newton_renderer|", "|newton_renderer|ik", "|newton_renderer|joint", "|ovrtx|", "|ovrtx|ik", "|ovrtx|joint", "isaacsim_physx||", "isaacsim_physx||ik", "isaacsim_physx||joint", "isaacsim_physx|isaacsim_rtx|", "isaacsim_physx|isaacsim_rtx|ik", "isaacsim_physx|isaacsim_rtx|joint", "isaacsim_physx|newton_renderer|", "isaacsim_physx|newton_renderer|ik", "isaacsim_physx|newton_renderer|joint", "newton_mjwarp_vbd_proxy||", "newton_mjwarp_vbd_proxy||ik", "newton_mjwarp_vbd_proxy||joint", "newton_mjwarp_vbd_proxy|isaacsim_rtx|", "newton_mjwarp_vbd_proxy|isaacsim_rtx|ik", "newton_mjwarp_vbd_proxy|isaacsim_rtx|joint", "newton_mjwarp_vbd_proxy|newton_renderer|", "newton_mjwarp_vbd_proxy|newton_renderer|ik", "newton_mjwarp_vbd_proxy|newton_renderer|joint", "newton_mjwarp_vbd_proxy|ovrtx|", "newton_mjwarp_vbd_proxy|ovrtx|ik", "newton_mjwarp_vbd_proxy|ovrtx|joint"]],
            ["Isaac-Open-Drawer-Franka-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "environment-previews/franka_open_drawer.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Open-Drawer-Franka", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "environment-previews/franka_open_drawer.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Pendulum-MARL-Direct", "rl_games,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "environment-previews/cart_double_pendulum.jpg", false, {}, {"skrl": "MAPPO"}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Reach-Franka", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "diffik,diffik_abs,hold,joint_pos,minimal,newton_ik", "environment-previews/franka_reach.jpg", true, {"*": ["joint_pos"]}, {}, "", ["joint_pos", "minimal"], ["||", "||diffik", "||diffik_abs", "||hold", "||joint_pos", "||minimal", "||newton_ik", "isaacsim_physx||", "isaacsim_physx||diffik", "isaacsim_physx||diffik_abs", "isaacsim_physx||hold", "isaacsim_physx||joint_pos", "isaacsim_physx||minimal", "newton_mjwarp||", "newton_mjwarp||diffik", "newton_mjwarp||diffik_abs", "newton_mjwarp||hold", "newton_mjwarp||joint_pos", "newton_mjwarp||minimal", "newton_mjwarp||newton_ik", "ovphysx||", "ovphysx||diffik", "ovphysx||diffik_abs", "ovphysx||hold", "ovphysx||joint_pos", "ovphysx||minimal"]],
            ["Isaac-Reach-Franka-OSC", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "diffik_abs,minimal", "environment-previews/franka_reach.jpg", false, {"*": ["diffik_abs"]}, {}, "", null, ["||", "||diffik_abs", "||minimal", "isaacsim_physx||", "isaacsim_physx||diffik_abs", "isaacsim_physx||minimal", "newton_mjwarp||", "newton_mjwarp||diffik_abs", "newton_mjwarp||minimal", "ovphysx||", "ovphysx||diffik_abs", "ovphysx||minimal"]],
            ["Isaac-Reach-UR10", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "environment-previews/ur10_reach.jpg", true, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Reorient-Cube-Allegro-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "environment-previews/allegro_cube.jpg", true, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Reorient-Cube-Allegro", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "randomized,reset_only", "environment-previews/allegro_cube.jpg", false, {"*": ["reset_only"]}, {}, "", null, ["||", "||randomized", "||reset_only", "isaacsim_physx||", "isaacsim_physx||randomized", "isaacsim_physx||reset_only", "newton_mjwarp||", "newton_mjwarp||randomized", "newton_mjwarp||reset_only", "ovphysx||", "ovphysx||randomized", "ovphysx||reset_only"]],
            ["Isaac-Reorient-Cube-Shadow-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "environment-previews/shadow_cube.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Reorient-Cube-Shadow", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "asymmetric,randomized", "environment-previews/shadow_cube.jpg", false, {}, {}, "", null, ["||", "||asymmetric", "||randomized", "isaacsim_physx||", "isaacsim_physx||asymmetric", "isaacsim_physx||randomized", "newton_mjwarp||", "newton_mjwarp||asymmetric", "newton_mjwarp||randomized", "ovphysx||", "ovphysx||asymmetric", "ovphysx||randomized"]],
            ["Isaac-Reorient-Cube-Shadow-Camera-Direct", "rl_games,rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo,full,rgb,semantic_segmentation,simple_shading_constant_diffuse,simple_shading_diffuse_mdl,simple_shading_full_mdl", "tasks/manipulation/shadow_cube.jpg", false, {"*": ["full"]}, {}, "", null, ["||", "||albedo", "||full", "||rgb", "||semantic_segmentation", "|isaacsim_rtx|", "|isaacsim_rtx|albedo", "|isaacsim_rtx|full", "|isaacsim_rtx|rgb", "|isaacsim_rtx|semantic_segmentation", "|isaacsim_rtx|simple_shading_constant_diffuse", "|isaacsim_rtx|simple_shading_diffuse_mdl", "|isaacsim_rtx|simple_shading_full_mdl", "|newton_renderer|", "|newton_renderer|albedo", "|newton_renderer|full", "|newton_renderer|rgb", "|newton_renderer|semantic_segmentation", "|ovrtx|", "|ovrtx|albedo", "|ovrtx|full", "|ovrtx|rgb", "|ovrtx|semantic_segmentation", "|ovrtx|simple_shading_constant_diffuse", "|ovrtx|simple_shading_diffuse_mdl", "|ovrtx|simple_shading_full_mdl", "isaacsim_physx||", "isaacsim_physx||albedo", "isaacsim_physx||full", "isaacsim_physx||rgb", "isaacsim_physx||semantic_segmentation", "isaacsim_physx|isaacsim_rtx|", "isaacsim_physx|isaacsim_rtx|albedo", "isaacsim_physx|isaacsim_rtx|full", "isaacsim_physx|isaacsim_rtx|rgb", "isaacsim_physx|isaacsim_rtx|semantic_segmentation", "isaacsim_physx|isaacsim_rtx|simple_shading_constant_diffuse", "isaacsim_physx|isaacsim_rtx|simple_shading_diffuse_mdl", "isaacsim_physx|isaacsim_rtx|simple_shading_full_mdl", "isaacsim_physx|newton_renderer|", "isaacsim_physx|newton_renderer|albedo", "isaacsim_physx|newton_renderer|full", "isaacsim_physx|newton_renderer|rgb", "isaacsim_physx|newton_renderer|semantic_segmentation", "newton_mjwarp||", "newton_mjwarp||albedo", "newton_mjwarp||full", "newton_mjwarp||rgb", "newton_mjwarp||semantic_segmentation", "newton_mjwarp|isaacsim_rtx|", "newton_mjwarp|isaacsim_rtx|albedo", "newton_mjwarp|isaacsim_rtx|full", "newton_mjwarp|isaacsim_rtx|rgb", "newton_mjwarp|isaacsim_rtx|semantic_segmentation", "newton_mjwarp|isaacsim_rtx|simple_shading_constant_diffuse", "newton_mjwarp|isaacsim_rtx|simple_shading_diffuse_mdl", "newton_mjwarp|isaacsim_rtx|simple_shading_full_mdl", "newton_mjwarp|newton_renderer|", "newton_mjwarp|newton_renderer|albedo", "newton_mjwarp|newton_renderer|full", "newton_mjwarp|newton_renderer|rgb", "newton_mjwarp|newton_renderer|semantic_segmentation", "newton_mjwarp|ovrtx|", "newton_mjwarp|ovrtx|albedo", "newton_mjwarp|ovrtx|full", "newton_mjwarp|ovrtx|rgb", "newton_mjwarp|ovrtx|semantic_segmentation", "newton_mjwarp|ovrtx|simple_shading_constant_diffuse", "newton_mjwarp|ovrtx|simple_shading_diffuse_mdl", "newton_mjwarp|ovrtx|simple_shading_full_mdl", "ovphysx||", "ovphysx||albedo", "ovphysx||full", "ovphysx||rgb", "ovphysx||semantic_segmentation", "ovphysx|newton_renderer|", "ovphysx|newton_renderer|albedo", "ovphysx|newton_renderer|full", "ovphysx|newton_renderer|rgb", "ovphysx|newton_renderer|semantic_segmentation", "ovphysx|ovrtx|", "ovphysx|ovrtx|albedo", "ovphysx|ovrtx|full", "ovphysx|ovrtx|rgb", "ovphysx|ovrtx|semantic_segmentation", "ovphysx|ovrtx|simple_shading_constant_diffuse", "ovphysx|ovrtx|simple_shading_diffuse_mdl", "ovphysx|ovrtx|simple_shading_full_mdl"]],
            ["Isaac-Reorient-Cube-Shadow-Camera", "rl_games,rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo,full,randomized,rgb,semantic_segmentation,simple_shading_constant_diffuse,simple_shading_diffuse_mdl,simple_shading_full_mdl", "tasks/manipulation/shadow_cube.jpg", false, {"*": ["full"]}, {}, "", null, ["||", "||albedo", "||full", "||randomized", "||rgb", "||semantic_segmentation", "|isaacsim_rtx|", "|isaacsim_rtx|albedo", "|isaacsim_rtx|full", "|isaacsim_rtx|randomized", "|isaacsim_rtx|rgb", "|isaacsim_rtx|semantic_segmentation", "|isaacsim_rtx|simple_shading_constant_diffuse", "|isaacsim_rtx|simple_shading_diffuse_mdl", "|isaacsim_rtx|simple_shading_full_mdl", "|newton_renderer|", "|newton_renderer|albedo", "|newton_renderer|full", "|newton_renderer|randomized", "|newton_renderer|rgb", "|newton_renderer|semantic_segmentation", "|ovrtx|", "|ovrtx|albedo", "|ovrtx|full", "|ovrtx|randomized", "|ovrtx|rgb", "|ovrtx|semantic_segmentation", "|ovrtx|simple_shading_constant_diffuse", "|ovrtx|simple_shading_diffuse_mdl", "|ovrtx|simple_shading_full_mdl", "isaacsim_physx||", "isaacsim_physx||albedo", "isaacsim_physx||full", "isaacsim_physx||randomized", "isaacsim_physx||rgb", "isaacsim_physx||semantic_segmentation", "isaacsim_physx|isaacsim_rtx|", "isaacsim_physx|isaacsim_rtx|albedo", "isaacsim_physx|isaacsim_rtx|full", "isaacsim_physx|isaacsim_rtx|randomized", "isaacsim_physx|isaacsim_rtx|rgb", "isaacsim_physx|isaacsim_rtx|semantic_segmentation", "isaacsim_physx|isaacsim_rtx|simple_shading_constant_diffuse", "isaacsim_physx|isaacsim_rtx|simple_shading_diffuse_mdl", "isaacsim_physx|isaacsim_rtx|simple_shading_full_mdl", "isaacsim_physx|newton_renderer|", "isaacsim_physx|newton_renderer|albedo", "isaacsim_physx|newton_renderer|full", "isaacsim_physx|newton_renderer|randomized", "isaacsim_physx|newton_renderer|rgb", "isaacsim_physx|newton_renderer|semantic_segmentation", "newton_mjwarp||", "newton_mjwarp||albedo", "newton_mjwarp||full", "newton_mjwarp||randomized", "newton_mjwarp||rgb", "newton_mjwarp||semantic_segmentation", "newton_mjwarp|isaacsim_rtx|", "newton_mjwarp|isaacsim_rtx|albedo", "newton_mjwarp|isaacsim_rtx|full", "newton_mjwarp|isaacsim_rtx|randomized", "newton_mjwarp|isaacsim_rtx|rgb", "newton_mjwarp|isaacsim_rtx|semantic_segmentation", "newton_mjwarp|isaacsim_rtx|simple_shading_constant_diffuse", "newton_mjwarp|isaacsim_rtx|simple_shading_diffuse_mdl", "newton_mjwarp|isaacsim_rtx|simple_shading_full_mdl", "newton_mjwarp|newton_renderer|", "newton_mjwarp|newton_renderer|albedo", "newton_mjwarp|newton_renderer|full", "newton_mjwarp|newton_renderer|randomized", "newton_mjwarp|newton_renderer|rgb", "newton_mjwarp|newton_renderer|semantic_segmentation", "newton_mjwarp|ovrtx|", "newton_mjwarp|ovrtx|albedo", "newton_mjwarp|ovrtx|full", "newton_mjwarp|ovrtx|randomized", "newton_mjwarp|ovrtx|rgb", "newton_mjwarp|ovrtx|semantic_segmentation", "newton_mjwarp|ovrtx|simple_shading_constant_diffuse", "newton_mjwarp|ovrtx|simple_shading_diffuse_mdl", "newton_mjwarp|ovrtx|simple_shading_full_mdl", "ovphysx||", "ovphysx||albedo", "ovphysx||full", "ovphysx||randomized", "ovphysx||rgb", "ovphysx||semantic_segmentation", "ovphysx|newton_renderer|", "ovphysx|newton_renderer|albedo", "ovphysx|newton_renderer|full", "ovphysx|newton_renderer|randomized", "ovphysx|newton_renderer|rgb", "ovphysx|newton_renderer|semantic_segmentation", "ovphysx|ovrtx|", "ovphysx|ovrtx|albedo", "ovphysx|ovrtx|full", "ovphysx|ovrtx|randomized", "ovphysx|ovrtx|rgb", "ovphysx|ovrtx|semantic_segmentation", "ovphysx|ovrtx|simple_shading_constant_diffuse", "ovphysx|ovrtx|simple_shading_diffuse_mdl", "ovphysx|ovrtx|simple_shading_full_mdl"]],
            ["Isaac-Reorient-KukaAllegro", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "cube,shapes", "environment-previews/kuka_allegro_reorient.jpg", false, {"*": ["shapes"]}, {}, "", null, ["||", "||cube", "||shapes", "isaacsim_physx||", "isaacsim_physx||cube", "isaacsim_physx||shapes", "newton_mjwarp||", "newton_mjwarp||cube", "newton_mjwarp||shapes", "ovphysx||", "ovphysx||cube", "ovphysx||shapes"]],
            ["Isaac-Reorient-KukaAllegro-Camera", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "isaacsim_rtx,newton_renderer,ovrtx", "albedo128,albedo256,albedo64,cube,depth128,depth256,depth64,duo_camera,raycaster_depth128,raycaster_depth256,raycaster_depth64,rgb128,rgb256,rgb64,semantic_segmentation128,semantic_segmentation256,semantic_segmentation64,shapes,simple_shading_constant_diffuse128,simple_shading_constant_diffuse256,simple_shading_constant_diffuse64,simple_shading_diffuse_mdl128,simple_shading_diffuse_mdl256,simple_shading_diffuse_mdl64,simple_shading_full_mdl128,simple_shading_full_mdl256,simple_shading_full_mdl64,single_camera", "environment-previews/kuka_allegro_reorient.jpg", false, {"*": ["rgb64", "shapes", "single_camera"]}, {}, "", null, ["||", "||albedo128", "||albedo256", "||albedo64", "||cube", "||depth128", "||depth256", "||depth64", "||duo_camera", "||raycaster_depth128", "||raycaster_depth256", "||raycaster_depth64", "||rgb128", "||rgb256", "||rgb64", "||semantic_segmentation128", "||semantic_segmentation256", "||semantic_segmentation64", "||shapes", "||single_camera", "|isaacsim_rtx|", "|isaacsim_rtx|albedo128", "|isaacsim_rtx|albedo256", "|isaacsim_rtx|albedo64", "|isaacsim_rtx|cube", "|isaacsim_rtx|depth128", "|isaacsim_rtx|depth256", "|isaacsim_rtx|depth64", "|isaacsim_rtx|duo_camera", "|isaacsim_rtx|rgb128", "|isaacsim_rtx|rgb256", "|isaacsim_rtx|rgb64", "|isaacsim_rtx|semantic_segmentation128", "|isaacsim_rtx|semantic_segmentation256", "|isaacsim_rtx|semantic_segmentation64", "|isaacsim_rtx|shapes", "|isaacsim_rtx|simple_shading_constant_diffuse128", "|isaacsim_rtx|simple_shading_constant_diffuse256", "|isaacsim_rtx|simple_shading_constant_diffuse64", "|isaacsim_rtx|simple_shading_diffuse_mdl128", "|isaacsim_rtx|simple_shading_diffuse_mdl256", "|isaacsim_rtx|simple_shading_diffuse_mdl64", "|isaacsim_rtx|simple_shading_full_mdl128", "|isaacsim_rtx|simple_shading_full_mdl256", "|isaacsim_rtx|simple_shading_full_mdl64", "|isaacsim_rtx|single_camera", "|newton_renderer|", "|newton_renderer|albedo128", "|newton_renderer|albedo256", "|newton_renderer|albedo64", "|newton_renderer|cube", "|newton_renderer|depth128", "|newton_renderer|depth256", "|newton_renderer|depth64", "|newton_renderer|duo_camera", "|newton_renderer|rgb128", "|newton_renderer|rgb256", "|newton_renderer|rgb64", "|newton_renderer|semantic_segmentation128", "|newton_renderer|semantic_segmentation256", "|newton_renderer|semantic_segmentation64", "|newton_renderer|shapes", "|newton_renderer|single_camera", "|ovrtx|", "|ovrtx|albedo128", "|ovrtx|albedo256", "|ovrtx|albedo64", "|ovrtx|cube", "|ovrtx|depth128", "|ovrtx|depth256", "|ovrtx|depth64", "|ovrtx|duo_camera", "|ovrtx|rgb128", "|ovrtx|rgb256", "|ovrtx|rgb64", "|ovrtx|semantic_segmentation128", "|ovrtx|semantic_segmentation256", "|ovrtx|semantic_segmentation64", "|ovrtx|shapes", "|ovrtx|simple_shading_constant_diffuse128", "|ovrtx|simple_shading_constant_diffuse256", "|ovrtx|simple_shading_constant_diffuse64", "|ovrtx|simple_shading_diffuse_mdl128", "|ovrtx|simple_shading_diffuse_mdl256", "|ovrtx|simple_shading_diffuse_mdl64", "|ovrtx|simple_shading_full_mdl128", "|ovrtx|simple_shading_full_mdl256", "|ovrtx|simple_shading_full_mdl64", "|ovrtx|single_camera", "isaacsim_physx||", "isaacsim_physx||albedo128", "isaacsim_physx||albedo256", "isaacsim_physx||albedo64", "isaacsim_physx||cube", "isaacsim_physx||depth128", "isaacsim_physx||depth256", "isaacsim_physx||depth64", "isaacsim_physx||duo_camera", "isaacsim_physx||raycaster_depth128", "isaacsim_physx||raycaster_depth256", "isaacsim_physx||raycaster_depth64", "isaacsim_physx||rgb128", "isaacsim_physx||rgb256", "isaacsim_physx||rgb64", "isaacsim_physx||semantic_segmentation128", "isaacsim_physx||semantic_segmentation256", "isaacsim_physx||semantic_segmentation64", "isaacsim_physx||shapes", "isaacsim_physx||single_camera", "isaacsim_physx|isaacsim_rtx|", "isaacsim_physx|isaacsim_rtx|albedo128", "isaacsim_physx|isaacsim_rtx|albedo256", "isaacsim_physx|isaacsim_rtx|albedo64", "isaacsim_physx|isaacsim_rtx|cube", "isaacsim_physx|isaacsim_rtx|depth128", "isaacsim_physx|isaacsim_rtx|depth256", "isaacsim_physx|isaacsim_rtx|depth64", "isaacsim_physx|isaacsim_rtx|duo_camera", "isaacsim_physx|isaacsim_rtx|rgb128", "isaacsim_physx|isaacsim_rtx|rgb256", "isaacsim_physx|isaacsim_rtx|rgb64", "isaacsim_physx|isaacsim_rtx|semantic_segmentation128", "isaacsim_physx|isaacsim_rtx|semantic_segmentation256", "isaacsim_physx|isaacsim_rtx|semantic_segmentation64", "isaacsim_physx|isaacsim_rtx|shapes", "isaacsim_physx|isaacsim_rtx|simple_shading_constant_diffuse128", "isaacsim_physx|isaacsim_rtx|simple_shading_constant_diffuse256", "isaacsim_physx|isaacsim_rtx|simple_shading_constant_diffuse64", "isaacsim_physx|isaacsim_rtx|simple_shading_diffuse_mdl128", "isaacsim_physx|isaacsim_rtx|simple_shading_diffuse_mdl256", "isaacsim_physx|isaacsim_rtx|simple_shading_diffuse_mdl64", "isaacsim_physx|isaacsim_rtx|simple_shading_full_mdl128", "isaacsim_physx|isaacsim_rtx|simple_shading_full_mdl256", "isaacsim_physx|isaacsim_rtx|simple_shading_full_mdl64", "isaacsim_physx|isaacsim_rtx|single_camera", "isaacsim_physx|newton_renderer|", "isaacsim_physx|newton_renderer|albedo128", "isaacsim_physx|newton_renderer|albedo256", "isaacsim_physx|newton_renderer|albedo64", "isaacsim_physx|newton_renderer|cube", "isaacsim_physx|newton_renderer|depth128", "isaacsim_physx|newton_renderer|depth256", "isaacsim_physx|newton_renderer|depth64", "isaacsim_physx|newton_renderer|duo_camera", "isaacsim_physx|newton_renderer|rgb128", "isaacsim_physx|newton_renderer|rgb256", "isaacsim_physx|newton_renderer|rgb64", "isaacsim_physx|newton_renderer|semantic_segmentation128", "isaacsim_physx|newton_renderer|semantic_segmentation256", "isaacsim_physx|newton_renderer|semantic_segmentation64", "isaacsim_physx|newton_renderer|shapes", "isaacsim_physx|newton_renderer|single_camera", "newton_mjwarp||", "newton_mjwarp||albedo128", "newton_mjwarp||albedo256", "newton_mjwarp||albedo64", "newton_mjwarp||cube", "newton_mjwarp||depth128", "newton_mjwarp||depth256", "newton_mjwarp||depth64", "newton_mjwarp||duo_camera", "newton_mjwarp||raycaster_depth128", "newton_mjwarp||raycaster_depth256", "newton_mjwarp||raycaster_depth64", "newton_mjwarp||rgb128", "newton_mjwarp||rgb256", "newton_mjwarp||rgb64", "newton_mjwarp||semantic_segmentation128", "newton_mjwarp||semantic_segmentation256", "newton_mjwarp||semantic_segmentation64", "newton_mjwarp||shapes", "newton_mjwarp||single_camera", "newton_mjwarp|isaacsim_rtx|", "newton_mjwarp|isaacsim_rtx|albedo128", "newton_mjwarp|isaacsim_rtx|albedo256", "newton_mjwarp|isaacsim_rtx|albedo64", "newton_mjwarp|isaacsim_rtx|cube", "newton_mjwarp|isaacsim_rtx|depth128", "newton_mjwarp|isaacsim_rtx|depth256", "newton_mjwarp|isaacsim_rtx|depth64", "newton_mjwarp|isaacsim_rtx|duo_camera", "newton_mjwarp|isaacsim_rtx|rgb128", "newton_mjwarp|isaacsim_rtx|rgb256", "newton_mjwarp|isaacsim_rtx|rgb64", "newton_mjwarp|isaacsim_rtx|semantic_segmentation128", "newton_mjwarp|isaacsim_rtx|semantic_segmentation256", "newton_mjwarp|isaacsim_rtx|semantic_segmentation64", "newton_mjwarp|isaacsim_rtx|shapes", "newton_mjwarp|isaacsim_rtx|simple_shading_constant_diffuse128", "newton_mjwarp|isaacsim_rtx|simple_shading_constant_diffuse256", "newton_mjwarp|isaacsim_rtx|simple_shading_constant_diffuse64", "newton_mjwarp|isaacsim_rtx|simple_shading_diffuse_mdl128", "newton_mjwarp|isaacsim_rtx|simple_shading_diffuse_mdl256", "newton_mjwarp|isaacsim_rtx|simple_shading_diffuse_mdl64", "newton_mjwarp|isaacsim_rtx|simple_shading_full_mdl128", "newton_mjwarp|isaacsim_rtx|simple_shading_full_mdl256", "newton_mjwarp|isaacsim_rtx|simple_shading_full_mdl64", "newton_mjwarp|isaacsim_rtx|single_camera", "newton_mjwarp|newton_renderer|", "newton_mjwarp|newton_renderer|albedo128", "newton_mjwarp|newton_renderer|albedo256", "newton_mjwarp|newton_renderer|albedo64", "newton_mjwarp|newton_renderer|cube", "newton_mjwarp|newton_renderer|depth128", "newton_mjwarp|newton_renderer|depth256", "newton_mjwarp|newton_renderer|depth64", "newton_mjwarp|newton_renderer|duo_camera", "newton_mjwarp|newton_renderer|rgb128", "newton_mjwarp|newton_renderer|rgb256", "newton_mjwarp|newton_renderer|rgb64", "newton_mjwarp|newton_renderer|semantic_segmentation128", "newton_mjwarp|newton_renderer|semantic_segmentation256", "newton_mjwarp|newton_renderer|semantic_segmentation64", "newton_mjwarp|newton_renderer|shapes", "newton_mjwarp|newton_renderer|single_camera", "newton_mjwarp|ovrtx|", "newton_mjwarp|ovrtx|albedo128", "newton_mjwarp|ovrtx|albedo256", "newton_mjwarp|ovrtx|albedo64", "newton_mjwarp|ovrtx|cube", "newton_mjwarp|ovrtx|depth128", "newton_mjwarp|ovrtx|depth256", "newton_mjwarp|ovrtx|depth64", "newton_mjwarp|ovrtx|duo_camera", "newton_mjwarp|ovrtx|rgb128", "newton_mjwarp|ovrtx|rgb256", "newton_mjwarp|ovrtx|rgb64", "newton_mjwarp|ovrtx|semantic_segmentation128", "newton_mjwarp|ovrtx|semantic_segmentation256", "newton_mjwarp|ovrtx|semantic_segmentation64", "newton_mjwarp|ovrtx|shapes", "newton_mjwarp|ovrtx|simple_shading_constant_diffuse128", "newton_mjwarp|ovrtx|simple_shading_constant_diffuse256", "newton_mjwarp|ovrtx|simple_shading_constant_diffuse64", "newton_mjwarp|ovrtx|simple_shading_diffuse_mdl128", "newton_mjwarp|ovrtx|simple_shading_diffuse_mdl256", "newton_mjwarp|ovrtx|simple_shading_diffuse_mdl64", "newton_mjwarp|ovrtx|simple_shading_full_mdl128", "newton_mjwarp|ovrtx|simple_shading_full_mdl256", "newton_mjwarp|ovrtx|simple_shading_full_mdl64", "newton_mjwarp|ovrtx|single_camera", "ovphysx||", "ovphysx||albedo128", "ovphysx||albedo256", "ovphysx||albedo64", "ovphysx||cube", "ovphysx||depth128", "ovphysx||depth256", "ovphysx||depth64", "ovphysx||duo_camera", "ovphysx||raycaster_depth128", "ovphysx||raycaster_depth256", "ovphysx||raycaster_depth64", "ovphysx||rgb128", "ovphysx||rgb256", "ovphysx||rgb64", "ovphysx||semantic_segmentation128", "ovphysx||semantic_segmentation256", "ovphysx||semantic_segmentation64", "ovphysx||shapes", "ovphysx||single_camera", "ovphysx|newton_renderer|", "ovphysx|newton_renderer|albedo128", "ovphysx|newton_renderer|albedo256", "ovphysx|newton_renderer|albedo64", "ovphysx|newton_renderer|cube", "ovphysx|newton_renderer|depth128", "ovphysx|newton_renderer|depth256", "ovphysx|newton_renderer|depth64", "ovphysx|newton_renderer|duo_camera", "ovphysx|newton_renderer|rgb128", "ovphysx|newton_renderer|rgb256", "ovphysx|newton_renderer|rgb64", "ovphysx|newton_renderer|semantic_segmentation128", "ovphysx|newton_renderer|semantic_segmentation256", "ovphysx|newton_renderer|semantic_segmentation64", "ovphysx|newton_renderer|shapes", "ovphysx|newton_renderer|single_camera", "ovphysx|ovrtx|", "ovphysx|ovrtx|albedo128", "ovphysx|ovrtx|albedo256", "ovphysx|ovrtx|albedo64", "ovphysx|ovrtx|cube", "ovphysx|ovrtx|depth128", "ovphysx|ovrtx|depth256", "ovphysx|ovrtx|depth64", "ovphysx|ovrtx|duo_camera", "ovphysx|ovrtx|rgb128", "ovphysx|ovrtx|rgb256", "ovphysx|ovrtx|rgb64", "ovphysx|ovrtx|semantic_segmentation128", "ovphysx|ovrtx|semantic_segmentation256", "ovphysx|ovrtx|semantic_segmentation64", "ovphysx|ovrtx|shapes", "ovphysx|ovrtx|simple_shading_constant_diffuse128", "ovphysx|ovrtx|simple_shading_constant_diffuse256", "ovphysx|ovrtx|simple_shading_constant_diffuse64", "ovphysx|ovrtx|simple_shading_diffuse_mdl128", "ovphysx|ovrtx|simple_shading_diffuse_mdl256", "ovphysx|ovrtx|simple_shading_diffuse_mdl64", "ovphysx|ovrtx|simple_shading_full_mdl128", "ovphysx|ovrtx|simple_shading_full_mdl256", "ovphysx|ovrtx|simple_shading_full_mdl64", "ovphysx|ovrtx|single_camera"]],
            ["Isaac-Shadow-Handover-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "environment-previews/shadow_hand_over.jpg", false, {}, {"skrl": "MAPPO"}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Shadow-Handover", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "randomized", "environment-previews/shadow_hand_over.jpg", false, {}, {}, "", null, ["||", "||randomized", "isaacsim_physx||", "isaacsim_physx||randomized", "newton_mjwarp||", "newton_mjwarp||randomized", "ovphysx||", "ovphysx||randomized"]],
            ["Isaac-Velocity-Flat-AnymalD", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "environment-previews/anymal_d_flat.jpg", true, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Velocity-Flat-Cassie", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "environment-previews/cassie_flat.jpg", true, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Velocity-Flat-G1", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "environment-previews/g1_flat.jpg", true, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Velocity-Flat-H1", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "environment-previews/h1_flat.jpg", true, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Velocity-Flat-UnitreeGo2", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "environment-previews/go2_flat.jpg", true, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Velocity-Rough-AnymalD", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/anymal_d_rough.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Velocity-Rough-Cassie", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/cassie_rough.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Velocity-Rough-G1", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/g1_rough.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Velocity-Rough-H1", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/h1_rough.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["Isaac-Velocity-Rough-UnitreeGo2", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/go2_rough.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["IsaacContrib-Assemble-Trocar-G129-Dex3", "rlinf", "", "", "", "tasks/manipulation/g1_assemble_trocar.jpg"],
            ["IsaacContrib-AutoMate-Assembly-Direct", "rl_games", "", "", "", "environment-previews/automate_assembly.jpg"],
            ["IsaacContrib-AutoMate-Disassembly-Direct", "rl_games", "", "", "", "environment-previews/automate_disassembly.jpg"],
            ["IsaacContrib-Cartpole-Camera-Showcase-Direct", "skrl", "", "", "box_box,box_discrete,box_multidiscrete,dict_box,dict_discrete,dict_multidiscrete,tuple_box,tuple_discrete,tuple_multidiscrete", "", false, {}, {}, "", null, ["||", "||box_box", "||box_discrete", "||box_multidiscrete", "||dict_box", "||dict_discrete", "||dict_multidiscrete", "||tuple_box", "||tuple_discrete", "||tuple_multidiscrete"]],
            ["IsaacContrib-Cartpole-Showcase-Direct", "skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "box_box,box_discrete,box_multidiscrete,dict_box,dict_discrete,dict_multidiscrete,discrete_box,discrete_discrete,discrete_multidiscrete,multidiscrete_box,multidiscrete_discrete,multidiscrete_multidiscrete,tuple_box,tuple_discrete,tuple_multidiscrete", "environment-previews/cartpole.jpg", false, {}, {}, "", null, ["||", "||box_box", "||box_discrete", "||box_multidiscrete", "||dict_box", "||dict_discrete", "||dict_multidiscrete", "||discrete_box", "||discrete_discrete", "||discrete_multidiscrete", "||multidiscrete_box", "||multidiscrete_discrete", "||multidiscrete_multidiscrete", "||tuple_box", "||tuple_discrete", "||tuple_multidiscrete", "isaacsim_physx||", "isaacsim_physx||box_box", "isaacsim_physx||box_discrete", "isaacsim_physx||box_multidiscrete", "isaacsim_physx||dict_box", "isaacsim_physx||dict_discrete", "isaacsim_physx||dict_multidiscrete", "isaacsim_physx||discrete_box", "isaacsim_physx||discrete_discrete", "isaacsim_physx||discrete_multidiscrete", "isaacsim_physx||multidiscrete_box", "isaacsim_physx||multidiscrete_discrete", "isaacsim_physx||multidiscrete_multidiscrete", "isaacsim_physx||tuple_box", "isaacsim_physx||tuple_discrete", "isaacsim_physx||tuple_multidiscrete", "newton_kamino||", "newton_kamino||box_box", "newton_kamino||box_discrete", "newton_kamino||box_multidiscrete", "newton_kamino||dict_box", "newton_kamino||dict_discrete", "newton_kamino||dict_multidiscrete", "newton_kamino||discrete_box", "newton_kamino||discrete_discrete", "newton_kamino||discrete_multidiscrete", "newton_kamino||multidiscrete_box", "newton_kamino||multidiscrete_discrete", "newton_kamino||multidiscrete_multidiscrete", "newton_kamino||tuple_box", "newton_kamino||tuple_discrete", "newton_kamino||tuple_multidiscrete", "newton_mjwarp||", "newton_mjwarp||box_box", "newton_mjwarp||box_discrete", "newton_mjwarp||box_multidiscrete", "newton_mjwarp||dict_box", "newton_mjwarp||dict_discrete", "newton_mjwarp||dict_multidiscrete", "newton_mjwarp||discrete_box", "newton_mjwarp||discrete_discrete", "newton_mjwarp||discrete_multidiscrete", "newton_mjwarp||multidiscrete_box", "newton_mjwarp||multidiscrete_discrete", "newton_mjwarp||multidiscrete_multidiscrete", "newton_mjwarp||tuple_box", "newton_mjwarp||tuple_discrete", "newton_mjwarp||tuple_multidiscrete", "ovphysx||", "ovphysx||box_box", "ovphysx||box_discrete", "ovphysx||box_multidiscrete", "ovphysx||dict_box", "ovphysx||dict_discrete", "ovphysx||dict_multidiscrete", "ovphysx||discrete_box", "ovphysx||discrete_discrete", "ovphysx||discrete_multidiscrete", "ovphysx||multidiscrete_box", "ovphysx||multidiscrete_discrete", "ovphysx||multidiscrete_multidiscrete", "ovphysx||tuple_box", "ovphysx||tuple_discrete", "ovphysx||tuple_multidiscrete"]],
            ["IsaacContrib-Deploy-GearAssembly-Rizon4s-Grav", "rsl_rl", "", "", "", "environment-previews/rizon4s_gear_assembly.jpg"],
            ["IsaacContrib-Deploy-GearAssembly-Rizon4s-Grav-ROS-Inference", "rsl_rl", "", "", "", "environment-previews/rizon4s_gear_assembly.jpg"],
            ["IsaacContrib-Deploy-GearAssembly-UR10e-2F140", "rsl_rl", "", "", "", "environment-previews/ur10e_2f140_gear_assembly.jpg"],
            ["IsaacContrib-Deploy-GearAssembly-UR10e-2F140-ROS-Inference", "rsl_rl", "", "", "", "environment-previews/ur10e_2f140_gear_assembly.jpg"],
            ["IsaacContrib-Deploy-GearAssembly-UR10e-2F85", "rsl_rl", "", "", "", "environment-previews/ur10e_2f85_gear_assembly.jpg"],
            ["IsaacContrib-Deploy-GearAssembly-UR10e-2F85-ROS-Inference", "rsl_rl", "", "", "", "environment-previews/ur10e_2f85_gear_assembly.jpg"],
            ["IsaacContrib-Deploy-Reach-Rizon4s", "rsl_rl", "", "", "", "environment-previews/rizon4s_reach.jpg"],
            ["IsaacContrib-Deploy-Reach-Rizon4s-ROS-Inference", "rsl_rl", "", "", "", "environment-previews/rizon4s_reach.jpg"],
            ["IsaacContrib-Deploy-Reach-UR10e", "rsl_rl", "", "", "", "environment-previews/ur10e_reach.jpg"],
            ["IsaacContrib-Deploy-Reach-UR10e-ROS-Inference", "rsl_rl", "", "", "", "environment-previews/ur10e_reach.jpg"],
            ["IsaacContrib-DrLegs-HoldPose", "rsl_rl", "isaacsim_physx,newton_kamino", "", "", "environment-previews/dr_legs.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||"]],
            ["IsaacContrib-DrLegs-Walk", "rsl_rl", "isaacsim_physx,newton_kamino", "", "", "environment-previews/dr_legs.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||"]],
            ["IsaacContrib-ExhaustPipe-GR1T2-Pink-IK-Abs", "", "", "", "", "environment-previews/gr1_exhaust_pipe.jpg"],
            ["IsaacContrib-Factory-Franka", "rsl_rl", "isaacsim_physx", "", "accumulator,choice,gear_mesh_large,gear_mesh_medium,gear_mesh_small,nut_thread_m16,peg_insert_12mm,peg_insert_16mm,peg_insert_4mm,peg_insert_8mm,rod_insert_12mm,rod_insert_16mm,rod_insert_4mm,rod_insert_8mm", "environment-previews/factory_franka_nist.jpg", false, {}, {}, "", null, ["||", "||accumulator", "||choice", "||gear_mesh_large", "||gear_mesh_medium", "||gear_mesh_small", "||nut_thread_m16", "||peg_insert_12mm", "||peg_insert_16mm", "||peg_insert_4mm", "||peg_insert_8mm", "||rod_insert_12mm", "||rod_insert_16mm", "||rod_insert_4mm", "||rod_insert_8mm", "isaacsim_physx||", "isaacsim_physx||accumulator", "isaacsim_physx||choice", "isaacsim_physx||gear_mesh_large", "isaacsim_physx||gear_mesh_medium", "isaacsim_physx||gear_mesh_small", "isaacsim_physx||nut_thread_m16", "isaacsim_physx||peg_insert_12mm", "isaacsim_physx||peg_insert_16mm", "isaacsim_physx||peg_insert_4mm", "isaacsim_physx||peg_insert_8mm", "isaacsim_physx||rod_insert_12mm", "isaacsim_physx||rod_insert_16mm", "isaacsim_physx||rod_insert_4mm", "isaacsim_physx||rod_insert_8mm"]],
            ["IsaacContrib-Factory-GearMesh-Direct", "rl_games", "", "", "", "environment-previews/gear_mesh.jpg"],
            ["IsaacContrib-Factory-NutThread-Direct", "rl_games", "", "", "", "environment-previews/nut_thread.jpg"],
            ["IsaacContrib-Factory-PegInsert-Direct", "rl_games", "", "", "", "environment-previews/peg_insert.jpg"],
            ["IsaacContrib-Forge-GearMesh-Direct", "rl_games", "", "", "", "environment-previews/gear_mesh.jpg"],
            ["IsaacContrib-Forge-NutThread-Direct", "rl_games", "", "", "", "environment-previews/nut_thread.jpg"],
            ["IsaacContrib-Forge-PegInsert-Direct", "rl_games", "", "", "", "environment-previews/peg_insert.jpg"],
            ["IsaacContrib-Franka-Pour", "rsl_rl", "", "", "", "environment-previews/franka_pour.jpg"],
            ["IsaacContrib-Humanoid-AMP-Dance-Direct", "skrl", "", "", "", "environment-previews/humanoid_amp.jpg"],
            ["IsaacContrib-Humanoid-AMP-Run-Direct", "skrl", "", "", "", "environment-previews/humanoid_amp.jpg"],
            ["IsaacContrib-Humanoid-AMP-Walk-Direct", "skrl", "", "", "", "environment-previews/humanoid_amp.jpg"],
            ["IsaacContrib-Keyboard-SO101", "rsl_rl", "isaacsim_physx,newton_mjwarp", "", "", "environment-previews/so101_keyboard.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||"]],
            ["IsaacContrib-Lift-Cube-Franka", "rl_games,rsl_rl,skrl,sb3", "", "", "", "environment-previews/franka_lift.jpg"],
            ["IsaacContrib-Lift-Cube-Franka-IK-Abs", "", "", "", "", "environment-previews/franka_lift.jpg"],
            ["IsaacContrib-Lift-Cube-Franka-IK-Rel", "", "", "", "", "environment-previews/franka_lift.jpg"],
            ["IsaacContrib-Lift-Cube-OpenArm", "rl_games,rsl_rl", "", "", "", "environment-previews/openarm_uni_lift.jpg"],
            ["IsaacContrib-Multitask-Manipulation", "rsl_rl", "", "", "", "environment-previews/multitask_manipulation.jpg"],
            ["IsaacContrib-Navigation-3DObstacles-ARL-Robot-1", "rl_games,rsl_rl,skrl", "", "", "", "tasks/drone_arl/arl_robot_1_navigation.jpg"],
            ["IsaacContrib-Navigation-Flat-AnymalC", "rsl_rl,skrl", "", "", "", "environment-previews/anymal_c_nav.jpg"],
            ["IsaacContrib-NutPour-GR1T2-Pink-IK-Abs", "", "", "", "", "environment-previews/gr1_nut_pour.jpg"],
            ["IsaacContrib-Open-Drawer-Franka-IK-Abs", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "environment-previews/franka_open_drawer.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||", "ovphysx||"]],
            ["IsaacContrib-Open-Drawer-Franka-IK-Rel", "rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "environment-previews/franka_open_drawer.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||", "ovphysx||"]],
            ["IsaacContrib-Open-Drawer-OpenArm", "rl_games,rsl_rl", "", "", "", "environment-previews/openarm_uni_open_drawer.jpg"],
            ["IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs", "", "", "isaacsim_rtx,newton_renderer", "", "environment-previews/g1_pick_place_fixed_base.jpg", false, {}, {}, "", null, ["||", "|isaacsim_rtx|", "|newton_renderer|"]],
            ["IsaacContrib-PickPlace-G1-InspireFTP-Abs", "", "", "", "", "environment-previews/g1_pick_place.jpg"],
            ["IsaacContrib-PickPlace-GR1T2-Abs", "", "", "isaacsim_rtx,newton_renderer", "", "environment-previews/gr_1_pick_place.jpg", false, {}, {}, "", null, ["||", "|isaacsim_rtx|", "|newton_renderer|"]],
            ["IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs", "", "", "", "", "environment-previews/gr_1_pick_place_waist.jpg"],
            ["IsaacContrib-PickPlace-Locomanipulation-G1-Abs", "", "", "isaacsim_rtx,newton_renderer", "", "environment-previews/g1_pick_place_locomanipulation.jpg", false, {}, {}, "", null, ["||", "|isaacsim_rtx|", "|newton_renderer|"]],
            ["IsaacContrib-Place-Mug-Agibot-Left-Arm-RmpFlow", "", "isaacsim_physx", "", "", "environment-previews/agibot_place_mug.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-Place-Toy2Box-Agibot-Right-Arm-RmpFlow", "", "isaacsim_physx", "", "", "environment-previews/agibot_place_toy.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-Reach-OpenArm", "rl_games,rsl_rl,skrl", "", "", "", "environment-previews/openarm_uni_reach.jpg"],
            ["IsaacContrib-Reach-OpenArmBi", "rl_games,rsl_rl", "", "", "", "environment-previews/openarm_bi_reach.jpg"],
            ["IsaacContrib-Reorient-Cube-Shadow-OpenAI-FF-Direct", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "environment-previews/shadow_cube.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||", "ovphysx||"]],
            ["IsaacContrib-Reorient-Cube-Shadow-OpenAI-LSTM-Direct", "rl_games,rsl_rl", "isaacsim_physx,newton_mjwarp,ovphysx", "", "", "environment-previews/shadow_cube.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||", "ovphysx||"]],
            ["IsaacContrib-Shoelace-DualFranka", "rsl_rl", "", "", "", "environment-previews/shoelace_dual_franka.jpg"],
            ["IsaacContrib-Stack-Cube-Bin-Franka-IK-Rel-Mimic", "", "isaacsim_physx", "", "", "environment-previews/franka_bin_stack.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-Stack-Cube-BlueGreen-Franka-IK-Rel", "", "isaacsim_physx", "", "", "environment-previews/franka_stack.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-Stack-Cube-BlueGreenRed-Franka-IK-Rel", "", "isaacsim_physx", "", "", "environment-previews/franka_stack.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-Stack-Cube-Franka", "", "isaacsim_physx", "", "", "environment-previews/franka_stack.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-Stack-Cube-Franka-IK-Abs", "", "isaacsim_physx", "", "", "environment-previews/franka_stack.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-Stack-Cube-Franka-IK-Rel", "", "isaacsim_physx", "", "", "environment-previews/franka_stack.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-Stack-Cube-Franka-IK-Rel-Blueprint", "", "isaacsim_physx,newton_mjwarp", "", "", "environment-previews/franka_stack.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||"]],
            ["IsaacContrib-Stack-Cube-Franka-IK-Rel-Skillgen", "", "isaacsim_physx", "", "", "environment-previews/franka_stack.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor", "", "isaacsim_physx,newton_mjwarp", "", "", "environment-previews/franka_stack.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||"]],
            ["IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos", "", "isaacsim_physx,newton_mjwarp", "", "", "environment-previews/franka_stack.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||"]],
            ["IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow", "", "isaacsim_physx", "", "", "environment-previews/galbot_stack_cube.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor", "", "isaacsim_physx", "isaacsim_rtx,newton_renderer", "", "environment-previews/galbot_stack_cube.jpg", false, {}, {}, "", null, ["||", "|isaacsim_rtx|", "|newton_renderer|", "isaacsim_physx||", "isaacsim_physx|isaacsim_rtx|", "isaacsim_physx|newton_renderer|"]],
            ["IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-Joint-Position", "", "isaacsim_physx", "isaacsim_rtx,newton_renderer", "", "environment-previews/galbot_stack_cube.jpg", false, {}, {}, "", null, ["||", "|isaacsim_rtx|", "|newton_renderer|", "isaacsim_physx||", "isaacsim_physx|isaacsim_rtx|", "isaacsim_physx|newton_renderer|"]],
            ["IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-RmpFlow", "", "isaacsim_physx", "isaacsim_rtx,newton_renderer", "", "environment-previews/galbot_stack_cube.jpg", false, {}, {}, "", null, ["||", "|isaacsim_rtx|", "|newton_renderer|", "isaacsim_physx||", "isaacsim_physx|isaacsim_rtx|", "isaacsim_physx|newton_renderer|"]],
            ["IsaacContrib-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow", "", "isaacsim_physx", "", "", "environment-previews/galbot_right_suction_stack.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-Stack-Cube-Instance-Randomize-Franka", "", "", "", "", "environment-previews/franka_instance_randomize_stack.jpg"],
            ["IsaacContrib-Stack-Cube-Instance-Randomize-Franka-IK-Rel", "", "", "", "", "environment-previews/franka_instance_randomize_stack.jpg"],
            ["IsaacContrib-Stack-Cube-RedGreen-Franka-IK-Rel", "", "isaacsim_physx", "", "", "environment-previews/franka_stack.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-Stack-Cube-RedGreenBlue-Franka-IK-Rel", "", "isaacsim_physx", "", "", "environment-previews/franka_stack.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-Stack-Cube-SO101-IK-Abs-v0", "", "isaacsim_physx", "", "", "environment-previews/so101_stack.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-Stack-Cube-SO101-Joint-Teleop-v0", "", "isaacsim_physx", "", "", "environment-previews/so101_stack.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-Stack-Cube-SO101-v0", "", "isaacsim_physx", "", "", "environment-previews/so101_stack.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-Stack-Cube-UR10-Long-Suction-IK-Rel", "", "isaacsim_physx", "", "", "environment-previews/ur10_stack_surface_gripper.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-Stack-Cube-UR10-Short-Suction-IK-Rel", "", "isaacsim_physx", "", "", "environment-previews/ur10_stack_surface_gripper.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||"]],
            ["IsaacContrib-TrackPositionNoObstacles-ARL-Robot-1", "rl_games,rsl_rl,skrl", "", "", "", "tasks/drone_arl/arl_robot_1_track_position_state_based.jpg"],
            ["IsaacContrib-Tracking-LocoManip-Digit", "rsl_rl", "isaacsim_physx,newton_mjwarp", "", "", "environment-previews/agility_digit_loco_manip.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||"]],
            ["IsaacContrib-UR10-Particle-Push", "rsl_rl", "", "", "", "environment-previews/ur10_particle_push.jpg"],
            ["IsaacContrib-Velocity-Flat-AnymalB", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "environment-previews/anymal_b_flat.jpg", true, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["IsaacContrib-Velocity-Flat-AnymalC-Direct", "rl_games,rsl_rl,skrl", "", "", "", "environment-previews/anymal_c_flat.jpg"],
            ["IsaacContrib-Velocity-Flat-AnymalC", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "environment-previews/anymal_c_flat.jpg", true, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["IsaacContrib-Velocity-Flat-Digit", "rsl_rl", "isaacsim_physx,newton_mjwarp", "", "", "environment-previews/agility_digit_flat.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||"]],
            ["IsaacContrib-Velocity-Flat-MicroDuck", "rsl_rl", "", "", "backlash", "environment-previews/microduck_flat.jpg", false, {}, {}, "", null, ["||", "||backlash"]],
            ["IsaacContrib-Velocity-Flat-Spot", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp", "", "", "tasks/locomotion/spot_flat.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||"]],
            ["IsaacContrib-Velocity-Flat-UnitreeA1", "rsl_rl,skrl,sb3", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "environment-previews/a1_flat.jpg", true, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["IsaacContrib-Velocity-Flat-UnitreeGo1", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "environment-previews/go1_flat.jpg", true, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["IsaacContrib-Velocity-Rough-AnymalB", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/anymal_b_rough.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["IsaacContrib-Velocity-Rough-AnymalC-Direct", "rl_games,rsl_rl,skrl", "", "", "", "tasks/locomotion/anymal_c_rough.jpg"],
            ["IsaacContrib-Velocity-Rough-AnymalC", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/anymal_c_rough.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["IsaacContrib-Velocity-Rough-Digit", "rsl_rl", "isaacsim_physx,newton_mjwarp", "", "", "tasks/locomotion/agility_digit_rough.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_mjwarp||"]],
            ["IsaacContrib-Velocity-Rough-MicroDuck", "rsl_rl", "", "", "backlash", "environment-previews/microduck_flat.jpg", false, {}, {}, "", null, ["||", "||backlash"]],
            ["IsaacContrib-Velocity-Rough-UnitreeA1", "rsl_rl,skrl,sb3", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/a1_rough.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
            ["IsaacContrib-Velocity-Rough-UnitreeGo1", "rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "", "tasks/locomotion/go1_rough.jpg", false, {}, {}, "", null, ["||", "isaacsim_physx||", "newton_kamino||", "newton_mjwarp||", "ovphysx||"]],
        ];
        // END-AUTO-GENERATED: environment-browser-task-rows

    const splitValues = (value) => value ? value.split(",") : [];
    const selectors = ["physics", "renderer", "presets"];
    const runKey = (values) => selectors.map((selector) => values[selector]).join("|");
    const tasks = taskRows.map(([
        task, rl, physics, renderer, presets, previewImage = "", supportsWarpFrontend = false,
        pretrainedCheckpointPresetCompatibility = {},
        defaultAlgorithms = {},
        requiredExtras = "",
        warpPresets = null,
        launchableRuns = null,
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
        requiredExtras: splitValues(requiredExtras),
        warpPresets,
        launchableRuns: launchableRuns && new Set(launchableRuns),
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
        benchmarkChannel: "develop",
    };
    const rlLibraryExtras = {rl_games: "rl-games", sb3: "sb3", skrl: "skrl", rlinf: "rlinf", torchrl: "torchrl"};
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

    const preferredValue = (values, preferred) => preferred.find((value) => value && values.includes(value)) || values[0] || "";

    const populateSelect = (select, values, preferred) => {
        const choices = values.length ? values : [""];
        select.replaceChildren(...choices.map((value) => new Option(value || "default", value)));
        select.value = preferredValue(values, preferred);
        select.disabled = values.length === 0;
        select.preferred = preferred;
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
            [/Lift-Soft-Franka/, "environment-previews/franka_lift_soft.jpg"],
            [/Lift-(Cube-)?Franka/, "tasks/manipulation/franka_lift.jpg"],
            [/Lift-KukaAllegro/, "tasks/manipulation/kuka_allegro_lift.jpg"],
            [/Open-Drawer-Franka/, "tasks/manipulation/franka_open_drawer.jpg"],
            [/Reach-Franka/, "tasks/manipulation/franka_reach.jpg"],
            [/Reach-UR10/, "tasks/manipulation/ur10_reach.jpg"],
            [/Reorient-Cube-Allegro/, "tasks/manipulation/allegro_cube.jpg"],
            [/Reorient-Cube-Shadow/, "tasks/manipulation/shadow_cube.jpg"],
            [/Reorient-KukaAllegro/, "tasks/manipulation/kuka_allegro_reorient.jpg"],
            [/Shadow-Handover/, "tasks/manipulation/shadow_hand_over.jpg"],
            [/AnymalB/, "tasks/locomotion/anymal_b_flat.jpg"],
            [/AnymalC/, "tasks/locomotion/anymal_c_flat.jpg"],
            [/AnymalD/, "tasks/locomotion/anymal_d_flat.jpg"],
            [/Cassie/, "tasks/cassie_flat.jpg"],
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

    const previewImageBaseUrl = "https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/";
    const previewVideoBaseUrl = "https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/tasks/";
    const failedPreviewVideos = new Set();

    const previewImageUrl = (task) => {
        const imagePath = previewImageFor(task);
        return imagePath.startsWith("tasks/")
            ? new URL(imagePath, previewImageBaseUrl).href
            : new URL(`../../_static/${imagePath}`, window.location.href).href;
    };

    // Each preview video shares its image's file name, so tasks without an uploaded clip fall back to the image.
    const previewVideoUrl = (task) => (
        new URL(previewImageFor(task).split("/").pop().replace(/\.[^.]+$/, ".mp4"), previewVideoBaseUrl).href
    );

    const updateTaskControls = () => {
        const task = selectedTask();
        populateSelect(fields.rl, task.rl, [fields.rl.value, "rsl_rl", "rl_games", "skrl", "sb3"]);
        // "" leaves the selector to the task's config; it is listed last so named options are preferred.
        const withDefault = (values) => (values.length ? [...values, ""] : []);
        const physics = state.scope === "warp" ? ["newton_mjwarp"] : withDefault(task.physics);
        populateSelect(fields.physics, physics, [fields.physics.value, "newton_mjwarp", "isaacsim_physx", "ovphysx", "newton_kamino"]);
        const preferredRenderer = fields.physics.value.startsWith("newton") ? "newton_renderer" : "isaacsim_rtx";
        populateSelect(fields.renderer, withDefault(task.renderer), [fields.renderer.value, preferredRenderer, "ovrtx"]);
        const presets = state.scope === "warp" ? (task.warpPresets ?? task.presets) : task.presets;
        populateSelect(fields.presets, withDefault(presets), [fields.presets.value, "joint", "ik", "rgb", "cube", "single_camera"]);
        settleSelectors("physics");
    };

    const updateModeControls = () => {
        const supportsRl = selectedTask().rl.length > 0;
        if (!supportsRl) {
            state.mode = "train";
        }
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

    const selectorValues = () => Object.fromEntries(selectors.map((selector) => [selector, fields[selector].value]));

    // Tasks without selectors carry no run list; their only command is the default one.
    const isLaunchable = (task, values) => !task.launchableRuns || task.launchableRuns.has(runKey(values));

    // Keep ``kept`` and move the other selectors to the first launchable choice, preferring their current values.
    const settleSelectors = (kept) => {
        const task = selectedTask();
        const current = selectorValues();
        if (!isLaunchable(task, current)) {
            // Current value first, then the preferred options, then the rest of the options.
            const choices = (selector) => {
                if (selector === kept) {
                    return [current[selector]];
                }
                const options = [...fields[selector].options].map((option) => option.value);
                const preferred = (fields[selector].preferred || []).filter((value) => options.includes(value));
                return [...new Set([current[selector], ...preferred, ...options])];
            };
            const [physicsChoices, rendererChoices, presetChoices] = selectors.map(choices);
            const settled = physicsChoices.flatMap((physics) => rendererChoices.flatMap((renderer) => (
                presetChoices.map((presets) => ({physics, renderer, presets}))
            ))).find((values) => isLaunchable(task, values));
            if (settled) {
                selectors.forEach((selector) => { fields[selector].value = settled[selector]; });
            }
        }
        const values = selectorValues();
        for (const selector of selectors) {
            for (const option of fields[selector].options) {
                option.disabled = !isLaunchable(task, {...values, [selector]: option.value});
                option.title = option.disabled ? "Not available with the current selection" : "";
            }
        }
    };

    const incompatibleBackends = () => !isLaunchable(selectedTask(), selectorValues());

    const updateCommand = () => {
        const incompatible = incompatibleBackends();
        commandOutput.textContent = currentCommand();
        commandOutput.closest(".environment-command-output").classList.toggle("is-disabled", incompatible);
        builder.querySelector("[data-backend-warning]").hidden = !incompatible;
        copyButton.disabled = incompatible;
        copyStatus.textContent = "";
    };

    const currentCommand = () => {
        const task = selectedTask();
        const extras = [...task.requiredExtras];
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
        const videoUrl = previewVideoUrl(selectedTask());
        previewImage.src = previewImageUrl(selectedTask());
        previewImage.alt = `${state.task} preview`;
        // Keep the image up until the clip has a frame; tasks without a clip keep their image.
        const showVideo = () => {
            if (previewVideo.src === videoUrl) {
                previewVideo.hidden = false;
                previewImage.hidden = true;
                previewVideo.play().catch(() => {});
            }
        };
        if (!failedPreviewVideos.has(videoUrl)) {
            previewVideo.onerror = () => {
                if (previewVideo.src === videoUrl) {
                    failedPreviewVideos.add(videoUrl);
                    previewVideo.hidden = true;
                    previewImage.hidden = false;
                }
            };
            previewVideo.onloadeddata = showVideo;
            previewVideo.setAttribute("aria-label", `${state.task} preview`);
            if (previewVideo.src !== videoUrl) {
                previewVideo.hidden = true;
                previewImage.hidden = false;
                previewVideo.src = videoUrl;
            } else if (previewVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
                showVideo();
            }
        } else {
            previewVideo.onerror = null;
            previewVideo.onloadeddata = null;
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
        const latestVramRow = benchmarkRows
            .filter((row) => row.task === state.task && row.physics_backend === fields.physics.value)
            .filter((row) => !selectedTask().renderer.length || row.rendering_backend === fields.renderer.value)
            .filter((row) => row.rl_library === fields.rl.value)
            .filter((row) => Number.isFinite(Number(row.vram_mean_gb)) && Number(row.vram_mean_gb) > 0)
            .sort((left, right) => right.recorded_at_utc.localeCompare(left.recorded_at_utc))[0];
        const vram = preview.querySelector("[data-preview-vram]");
        vram.textContent = latestVramRow
            ? `${Number(latestVramRow.vram_mean_gb).toFixed(1)}GB (${latestVramRow.num_envs} envs)`
            : "Not available";
        vram.title = latestVramRow
            ? `Peak VRAM: ${Number(latestVramRow.vram_peak_gb).toFixed(2)} GB · measured ${(latestVramRow.measurement_timestamp || latestVramRow.recorded_at_utc).slice(0, 10)}`
            : "";
        updateBenchmark();
    };

    const updateSelection = () => {
        fields.task.value = state.task;
        updateTaskControls();
        updateModeControls();
        updateCommand();
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
            image.src = previewImageUrl(activeTask);
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
        newton_kamino: "kamino",
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
    const isBenchmarkPair = (physics, renderer) => renderer === "none"
        || (physics === "isaacsim_physx" && renderer === "isaacsim_rtx")
        || (physics === "ovphysx" && renderer === "ovrtx")
        || (["newton_mjwarp", "newton_mjwarp_vbd_proxy"].includes(physics)
            && ["newton_renderer", "ovrtx"].includes(renderer));
    // Snapshots contain one fixed camera profile per physics/renderer pair.
    const seriesKey = (row) => JSON.stringify([row.physics_backend, row.rendering_backend]);
    const configurationLabel = (row) => [
        backendLabels[row.physics_backend] || row.physics_backend,
        rendererLabels[row.rendering_backend],
    ].filter(Boolean).join(" + ");
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
        const snapshots = benchmarks.getAttribute(`data-benchmark-${state.benchmarkChannel}-dates`)
            .split(",").map((entry) => entry.split(":"));
        const dateKeys = snapshots.map(([date]) => date);
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
            label.textContent = snapshots[index][1]
                || new Date(`${date}T00:00:00Z`).toLocaleDateString(undefined, {
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
        for (const physics of selectedTask().physics) {
            for (const renderer of selectedTask().renderer.length ? selectedTask().renderer : ["none"]) {
                if (!isBenchmarkPair(physics, renderer)) {
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
            label.append(swatch, configurationLabel(representative));
            configuration.title = representative.missing
                ? `${rendererLabels[representative.rendering_backend] || ""} 8,192-environment matching-profile run unavailable`.trim()
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
        renderBenchmarkTable(rows);
        chart.replaceChildren(...(rows.length ? [renderBenchmarkChart(rows, maximum)] : []));
    };

    const renderBenchmarks = () => {
        if (!benchmarks) {
            return;
        }
        const snapshots = JSON.parse(document.querySelector("[data-environment-benchmark-rows]").textContent);
        benchmarkRows = ["release", "develop"].flatMap((channel) => {
            try {
                return snapshots[channel]
                    .filter((row) => row.data_origin === "measured" && row.workload === "training")
                    .filter((row) => [row.collection_fps_mean, row.total_fps_mean]
                        .every((value) => Number.isFinite(Number(value)) && Number(value) > 0))
                    .filter((row) => row.task.startsWith("Isaac-"))
                    .filter((row) => isBenchmarkPair(row.physics_backend, row.rendering_backend))
                    .map((row) => ({...row, channel}));
            } catch (error) {
                benchmarkErrors.add(channel);
                console.error(error);
                return [];
            }
        });
        updatePreview();
    };

    fields.task.addEventListener("change", () => {
        state.task = fields.task.value;
        updateSelection();
    });
    for (const selector of selectors) {
        fields[selector].addEventListener("change", () => settleSelectors(selector));
    }
    for (const field of [fields.rl, fields.physics, fields.renderer, fields.presets, fields.checkpoint]) {
        field.addEventListener("change", () => {
            updateModeControls();
            updateCommand();
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
            updateCommand();
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
        updateCommand();
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
        if (incompatibleBackends()) {
            return;
        }
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
