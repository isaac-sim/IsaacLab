/*
 * Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

(() => {
    "use strict";

    const initializeEnvironmentBrowser = () => {
        // Generated from the core rows in source/overview/environments.rst.
        const taskRows = [
        ["Isaac-Ant", "rl_games,rsl_rl,skrl,sb3", "newton_kamino,newton_mjwarp,physx", "", ""],
        ["Isaac-Ant-Direct", "rl_games,rsl_rl,skrl", "newton_kamino,newton_mjwarp,ovphysx,physx", "", ""],
        ["Isaac-Ant-Direct-Warp-v0", "rl_games,rsl_rl,skrl", "", "", ""],
        ["Isaac-Ant-Warp-v0", "rl_games,rsl_rl,skrl,sb3", "", "", ""],
        ["Isaac-Cartpole", "rl_games,rsl_rl,skrl,sb3", "newton_kamino,newton_mjwarp,ovphysx,physx", "", ""],
        ["Isaac-Cartpole-Camera", "rl_games,rsl_rl", "newton_kamino,newton_mjwarp,ovphysx,physx", "isaacsim_rtx,newton_renderer,ovrtx,rtx", "albedo,depth,resnet18,rgb,semantic_segmentation,simple_shading_constant_diffuse,simple_shading_diffuse_mdl,simple_shading_full_mdl,theia_tiny"],
        ["Isaac-Cartpole-Camera-Direct", "rl_games,rsl_rl,skrl", "newton_kamino,newton_mjwarp,ovphysx,physx", "isaacsim_rtx,newton_renderer,ovrtx,rtx", "albedo,depth,rgb,semantic_segmentation,simple_shading_constant_diffuse,simple_shading_diffuse_mdl,simple_shading_full_mdl"],
        ["Isaac-Cartpole-Direct", "rl_games,rsl_rl,skrl,sb3", "newton_kamino,newton_mjwarp,ovphysx,physx", "", ""],
        ["Isaac-Cartpole-Direct-Warp-v0", "rl_games,rsl_rl,skrl,sb3", "", "", ""],
        ["Isaac-Cartpole-Warp-v0", "rl_games,rsl_rl,skrl,sb3", "", "", ""],
        ["Isaac-Fourbar-Pole-Swingup", "rsl_rl", "newton_kamino", "", ""],
        ["Isaac-Humanoid", "rl_games,rsl_rl,skrl,sb3", "newton_mjwarp,physx", "", ""],
        ["Isaac-Humanoid-Direct", "rl_games,rsl_rl,skrl", "newton_mjwarp,ovphysx,physx", "", ""],
        ["Isaac-Humanoid-Direct-Warp-v0", "rl_games,rsl_rl,skrl", "", "", ""],
        ["Isaac-Humanoid-Warp-v0", "rl_games,rsl_rl,skrl,sb3", "", "", ""],
        ["Isaac-Lift-Cloth-Franka", "rsl_rl", "newton_mjwarp_vbd_proxy", "", ""],
        ["Isaac-Lift-Cube-Franka", "rl_games,rsl_rl,skrl,sb3", "", "", ""],
        ["Isaac-Lift-Franka", "rsl_rl", "newton_mjwarp,physx", "", "cube,shapes"],
        ["Isaac-Lift-KukaAllegro", "rsl_rl", "newton_mjwarp,ovphysx,physx", "", "cube,shapes"],
        ["Isaac-Lift-KukaAllegro-Camera", "rsl_rl", "newton_mjwarp,ovphysx,physx", "isaacsim_rtx,newton_renderer,ovrtx,rtx", "albedo128,albedo256,albedo64,cube,depth128,depth256,depth64,duo_camera,raycaster_depth128,raycaster_depth256,raycaster_depth64,rgb128,rgb256,rgb64,semantic_segmentation128,semantic_segmentation256,semantic_segmentation64,shapes,simple_shading_constant_diffuse128,simple_shading_constant_diffuse256,simple_shading_constant_diffuse64,simple_shading_diffuse_mdl128,simple_shading_diffuse_mdl256,simple_shading_diffuse_mdl64,simple_shading_full_mdl128,simple_shading_full_mdl256,simple_shading_full_mdl64,single_camera"],
        ["Isaac-Lift-Soft-Franka", "rsl_rl", "newton_mjwarp_vbd_proxy,physx", "", ""],
        ["Isaac-Open-Drawer-Franka", "rl_games,rsl_rl,skrl", "", "", ""],
        ["Isaac-Open-Drawer-Franka-Direct", "rl_games,rsl_rl,skrl", "newton_mjwarp,ovphysx,physx", "", ""],
        ["Isaac-Pendulum-Direct", "rl_games,skrl", "", "", ""],
        ["Isaac-Reach-Franka", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", "diffik,joint_pos,newton_ik"],
        ["Isaac-Reach-Franka-OSC", "rsl_rl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", ""],
        ["Isaac-Reach-Franka-Warp-v0", "rl_games,rsl_rl,skrl", "", "", ""],
        ["Isaac-Reach-UR10", "rl_games,rsl_rl,skrl", "isaacsim_physx,newton_kamino,newton_mjwarp,ovphysx", "", ""],
        ["Isaac-Reorient-Cube-Allegro", "rl_games,rsl_rl,skrl", "", "", ""],
        ["Isaac-Reorient-Cube-Allegro-Direct", "rl_games,rsl_rl,skrl", "newton_mjwarp,ovphysx,physx", "", ""],
        ["Isaac-Reorient-Cube-Allegro-Direct-Warp-v0", "rl_games,rsl_rl,skrl", "", "", ""],
        ["Isaac-Reorient-Cube-Shadow-Camera-Direct", "rl_games,rsl_rl", "newton_kamino,newton_mjwarp,ovphysx,physx", "isaacsim_rtx,newton_renderer,ovrtx,rtx", "albedo,depth,full,rgb,semantic_segmentation,simple_shading_constant_diffuse,simple_shading_diffuse_mdl,simple_shading_full_mdl"],
        ["Isaac-Reorient-Cube-Shadow-Direct", "rl_games,rsl_rl,skrl", "newton_kamino,newton_mjwarp,physx", "", ""],
        ["Isaac-Reorient-Cube-Shadow-OpenAI-FF-Direct", "rl_games,rsl_rl,skrl", "newton_kamino,newton_mjwarp,physx", "", ""],
        ["Isaac-Reorient-Cube-Shadow-OpenAI-LSTM-Direct", "rl_games", "newton_kamino,newton_mjwarp,physx", "", ""],
        ["Isaac-Reorient-Franka", "rsl_rl", "newton_mjwarp,physx", "", "cube,shapes"],
        ["Isaac-Reorient-KukaAllegro", "rsl_rl", "newton_mjwarp,ovphysx,physx", "", "cube,shapes"],
        ["Isaac-Reorient-KukaAllegro-Camera", "rsl_rl", "newton_mjwarp,ovphysx,physx", "isaacsim_rtx,newton_renderer,ovrtx,rtx", "albedo128,albedo256,albedo64,cube,depth128,depth256,depth64,duo_camera,raycaster_depth128,raycaster_depth256,raycaster_depth64,rgb128,rgb256,rgb64,semantic_segmentation128,semantic_segmentation256,semantic_segmentation64,shapes,simple_shading_constant_diffuse128,simple_shading_constant_diffuse256,simple_shading_constant_diffuse64,simple_shading_diffuse_mdl128,simple_shading_diffuse_mdl256,simple_shading_diffuse_mdl64,simple_shading_full_mdl128,simple_shading_full_mdl256,simple_shading_full_mdl64,single_camera"],
        ["Isaac-Shadow-Handover-Direct", "rl_games,skrl", "newton_mjwarp,physx", "", ""],
        ["Isaac-Velocity-Flat-AnymalB-Warp-v0", "rsl_rl,skrl", "", "", ""],
        ["Isaac-Velocity-Flat-AnymalC-Warp-v0", "rl_games,rsl_rl,skrl", "", "", ""],
        ["Isaac-Velocity-Flat-AnymalD", "rsl_rl,skrl", "newton_mjwarp,ovphysx,physx", "", ""],
        ["Isaac-Velocity-Flat-AnymalD-Warp-v0", "rsl_rl,skrl", "", "", ""],
        ["Isaac-Velocity-Flat-Cassie", "rsl_rl,skrl", "newton_kamino,newton_mjwarp,physx", "", ""],
        ["Isaac-Velocity-Flat-Cassie-Warp-v0", "rsl_rl,skrl", "", "", ""],
        ["Isaac-Velocity-Flat-Digit", "rsl_rl", "newton_mjwarp,ovphysx,physx", "", ""],
        ["Isaac-Velocity-Flat-G1", "rsl_rl,skrl", "newton_kamino,newton_mjwarp,physx", "", ""],
        ["Isaac-Velocity-Flat-G1-Warp-v0", "rsl_rl,skrl", "", "", ""],
        ["Isaac-Velocity-Flat-H1", "rsl_rl,skrl", "newton_kamino,newton_mjwarp,physx", "", ""],
        ["Isaac-Velocity-Flat-H1-Warp-v0", "rsl_rl,skrl", "", "", ""],
        ["Isaac-Velocity-Flat-Spot", "rsl_rl,skrl", "newton_kamino,newton_mjwarp,physx", "", ""],
        ["Isaac-Velocity-Flat-UnitreeA1-Warp-v0", "rsl_rl,skrl,sb3", "", "", ""],
        ["Isaac-Velocity-Flat-UnitreeGo1-Warp-v0", "rsl_rl,skrl", "", "", ""],
        ["Isaac-Velocity-Flat-UnitreeGo2", "rsl_rl,skrl", "newton_kamino,newton_mjwarp,physx", "", ""],
        ["Isaac-Velocity-Flat-UnitreeGo2-Warp-v0", "rsl_rl,skrl", "", "", ""],
        ["Isaac-Velocity-Rough-AnymalD", "rsl_rl,skrl", "newton_mjwarp,ovphysx,physx", "", ""],
        ["Isaac-Velocity-Rough-Cassie", "rsl_rl,skrl", "newton_mjwarp,ovphysx,physx", "", ""],
        ["Isaac-Velocity-Rough-Digit", "rsl_rl", "newton_mjwarp,ovphysx,physx", "", ""],
        ["Isaac-Velocity-Rough-G1", "rsl_rl,skrl", "newton_mjwarp,ovphysx,physx", "", ""],
        ["Isaac-Velocity-Rough-H1", "rsl_rl,skrl", "newton_mjwarp,ovphysx,physx", "", ""],
        ["Isaac-Velocity-Rough-UnitreeGo2", "rsl_rl,skrl", "newton_mjwarp,ovphysx,physx", "", ""],
    ];

    const splitValues = (value) => value ? value.split(",") : [];
    const tasks = taskRows.map(([task, rl, physics, renderer, presets]) => ({
        task,
        rl: splitValues(rl),
        physics: splitValues(physics),
        renderer: splitValues(renderer),
        presets: splitValues(presets),
    }));

    const builder = document.querySelector("[data-environment-browser]");
    const preview = document.querySelector("[data-environment-preview]");
    const taskBrowser = document.querySelector("[data-environment-tasks]");
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
    const state = {mode: "train", task: "Isaac-Cartpole-Direct"};

    const categoryFor = (task) => {
        if (task.includes("Velocity")) {
            return "locomotion";
        }
        if (/Lift|Reach|Reorient|Drawer|Handover/.test(task)) {
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

    const previewImageFor = (taskName) => {
        const imageRules = [
            [/Fourbar/, "tasks/classic/fourbar_pole.jpg"],
            [/Cartpole/, "tasks/classic/cartpole.jpg"],
            [/Pendulum/, "tasks/classic/cart_double_pendulum.jpg"],
            [/^Isaac-Ant/, "tasks/classic/ant.jpg"],
            [/^Isaac-Humanoid/, "tasks/classic/humanoid.jpg"],
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

    const updateTaskControls = () => {
        const task = selectedTask();
        populateSelect(fields.rl, task.rl, [fields.rl.value, "rsl_rl", "rl_games", "skrl", "sb3"]);
        populateSelect(fields.physics, task.physics, [fields.physics.value, "newton_mjwarp", "physx", "newton_kamino"]);
        const preferredRenderer = fields.physics.value.startsWith("newton") ? "newton_renderer" : "isaacsim_rtx";
        populateSelect(fields.renderer, task.renderer, [fields.renderer.value, preferredRenderer, "ovrtx", "rtx"]);
        populateSelect(fields.presets, task.presets, [fields.presets.value, "rgb", "cube", "single_camera"]);
    };

    const currentCommand = () => {
        const parts = ["uv", "run", "isaaclab", state.mode, "--rl_library", fields.rl.value, "--task", state.task];
        for (const selector of ["physics", "renderer", "presets"]) {
            if (fields[selector].value) {
                parts.push(`${selector}=${fields[selector].value}`);
            }
        }
        return parts.join(" ");
    };

    const updatePreview = () => {
        const previewImage = preview.querySelector("[data-preview-image]");
        const previewImageName = previewImageFor(state.task).split("/").pop();
        previewImage.src = new URL(`../../_images/${previewImageName}`, window.location.href).href;
        previewImage.alt = `${state.task} preview`;
        preview.querySelector("[data-preview-task]").textContent = state.task;
        preview.querySelector("[data-preview-mode]").textContent = state.mode === "train" ? "Train" : "Play";
        preview.querySelector("[data-preview-rl]").textContent = fields.rl.value || "Default";
        preview.querySelector("[data-preview-physics]").textContent = fields.physics.value || "Default";
        preview.querySelector("[data-preview-renderer]").textContent = fields.renderer.value || "Default";
        preview.querySelector("[data-preview-presets]").textContent = fields.presets.value || "Default";
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
            const matchesCategory = category === "all" || categoryFor(task.task) === category;
            return matchesQuery && matchesCategory;
        });
        taskList.replaceChildren(...visibleTasks.map((task) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "environment-task-row";
            button.dataset.taskName = task.task;
            button.setAttribute("aria-pressed", String(task.task === state.task));
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
        taskCount.textContent = `${visibleTasks.length} core ${visibleTasks.length === 1 ? "task" : "tasks"}`;
        taskEmpty.hidden = visibleTasks.length !== 0;
        taskList.hidden = visibleTasks.length === 0;
        updateSelection();
    };

    fields.task.replaceChildren(...tasks.map((task) => new Option(task.task, task.task)));
    fields.task.value = state.task;
    updateTaskControls();

    fields.task.addEventListener("change", () => {
        state.task = fields.task.value;
        updateSelection();
    });
    for (const field of [fields.rl, fields.physics, fields.renderer, fields.presets]) {
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
            commandOutput.textContent = currentCommand();
            updatePreview();
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

        renderTasks();
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initializeEnvironmentBrowser, {once: true});
    } else {
        initializeEnvironmentBrowser();
    }
})();
