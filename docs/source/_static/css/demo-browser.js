/*
 * Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

(() => {
    "use strict";

    const splitValues = (value) => value ? value.split(",") : [];

    const initializeDemoBrowser = () => {
        const browser = document.querySelector("[data-demo-browser]");
        if (!browser) {
            return;
        }

        const cards = [...browser.querySelectorAll("[data-demo-id], [data-example-id]")];
        const fields = Object.fromEntries(
            [...browser.querySelectorAll("[data-demo-field]")].map((field) => [field.dataset.demoField, field])
        );
        const commandOutput = browser.querySelector("[data-command-output]");
        const copyButton = browser.querySelector("[data-copy-command]");
        const copyStatus = browser.querySelector("[data-copy-status]");
        const selectedName = browser.querySelector("[data-demo-name]:not([data-demo-id]):not([data-example-id])");
        const selectedDescription = browser.querySelector(
            "[data-demo-description]:not([data-demo-id]):not([data-example-id])"
        );
        const optionPanel = browser.querySelector("[data-demo-options]");
        const videoToggle = browser.querySelector("[data-demo-toggle='video']");
        const notes = [...browser.querySelectorAll("[data-demo-note], [data-demo-note-visualizers]")];
        let selectedCard = cards[0];

        const selectedProgramId = () => selectedCard.dataset.demoId || selectedCard.dataset.exampleId;
        const defaultRecording = () => selectedCard.dataset.demoDefaultRecording === fields.physics.value
            && fields.visualizer.value === "default";

        const populateSelect = (select, values, preferredValues) => {
            const previousValue = select.value;
            select.replaceChildren(...values.map((value) => new Option(
                value === "default" ? "Default" : value, value
            )));
            select.value = [previousValue, ...preferredValues].find((value) => values.includes(value)) || values[0];
            select.disabled = values.length === 1;
        };

        const compatibleVisualizers = () => {
            const physicsKey = `demoVisualizers${fields.physics.value
                .split("_")
                .map((part) => part[0].toUpperCase() + part.slice(1))
                .join("")}`;
            return splitValues(selectedCard.dataset[physicsKey] || selectedCard.dataset.demoVisualizers);
        };

        const currentCommand = () => {
            const requiredExtras = new Set(splitValues(selectedCard.dataset.demoExtras));
            const options = splitValues(selectedCard.dataset.demoOptions);
            const recordVideo = defaultRecording() || (options.includes("video") && videoToggle.checked);
            if (fields.physics.value === "isaacsim_physx" || fields.visualizer.value === "kit" || defaultRecording()) {
                requiredExtras.add("isaacsim");
            }
            if (fields.physics.value === "ovphysx") {
                requiredExtras.add("ovphysx");
            }
            if (fields.visualizer.value === "newton_rtx") {
                requiredExtras.add("ovrtx");
            }
            if (["rerun", "viser"].includes(fields.visualizer.value)) {
                requiredExtras.add(fields.visualizer.value);
            }
            if (recordVideo) {
                requiredExtras.add("video");
            }
            const extraOrder = [
                "isaacsim", "ovphysx", "ovrtx", "video", "tetrahedralization", "teleop", "rerun", "viser"
            ];
            const extras = extraOrder.filter((extra) => requiredExtras.has(extra));

            const parts = ["uvx", "--torch-backend", "cu130"];
            for (const dependency of splitValues(selectedCard.dataset.demoWith)) {
                parts.push("--with", `'${dependency}'`);
            }
            if (extras.length) {
                parts.push("--from", `'isaaclab[${extras.join(",")}]'`);
            }
            parts.push("isaaclab", selectedCard.dataset.exampleId ? "example" : "demo", selectedProgramId());
            if (selectedCard.dataset.demoFixedPhysics !== "true") {
                parts.push("--physics", fields.physics.value);
            }
            if (fields.visualizer.value !== "default") {
                parts.push("--viz", fields.visualizer.value);
            }
            if (recordVideo && !defaultRecording()) {
                parts.push("--video");
            }
            parts.push(...(selectedCard.dataset.demoArgs || "").split(" ").filter(Boolean));
            return parts.join(" ");
        };

        const updateCommand = () => {
            const supportsVideo = splitValues(selectedCard.dataset.demoOptions).includes("video");
            optionPanel.hidden = !supportsVideo;
            if (!supportsVideo || videoToggle.disabled) {
                videoToggle.checked = false;
            }
            videoToggle.disabled = defaultRecording();
            videoToggle.checked ||= defaultRecording();
            commandOutput.textContent = currentCommand();
            for (const note of notes) {
                const visualizers = splitValues(note.dataset.demoNoteVisualizers);
                note.hidden = Boolean(
                    (note.dataset.demoNote && note.dataset.demoNote !== selectedProgramId())
                    || (visualizers.length && !visualizers.includes(fields.visualizer.value))
                );
            }
        };

        const updateVisualizer = () => {
            const preferredVisualizers = fields.physics.value.startsWith("newton")
                ? ["default", "newton_gl", "kit"]
                : ["kit"];
            if (selectedCard.dataset.demoDefaultRecording === fields.physics.value) {
                fields.visualizer.value = "";
            }
            populateSelect(fields.visualizer, compatibleVisualizers(), preferredVisualizers);
            updateCommand();
        };

        const selectDemo = (card) => {
            selectedCard = card;
            videoToggle.checked = false;
            for (const candidate of cards) {
                const isSelected = candidate === selectedCard;
                candidate.classList.toggle("is-selected", isSelected);
                candidate.setAttribute("aria-pressed", String(isSelected));
            }
            selectedName.textContent = selectedCard.dataset.demoName;
            selectedDescription.textContent = selectedCard.dataset.demoDescription;
            populateSelect(fields.physics, splitValues(selectedCard.dataset.demoPhysics), ["newton_mjwarp", "isaacsim_physx"]);
            updateVisualizer();
        };

        for (const card of cards) {
            card.addEventListener("click", () => selectDemo(card));
        }
        fields.physics.addEventListener("change", updateVisualizer);
        fields.visualizer.addEventListener("change", updateCommand);
        videoToggle.addEventListener("change", updateCommand);
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

        selectDemo(selectedCard);
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initializeDemoBrowser, {once: true});
    } else {
        initializeDemoBrowser();
    }
})();
