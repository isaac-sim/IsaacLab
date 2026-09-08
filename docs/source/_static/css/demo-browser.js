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

        const cards = [...browser.querySelectorAll("[data-demo-path]")];
        const fields = Object.fromEntries(
            [...browser.querySelectorAll("[data-demo-field]")].map((field) => [field.dataset.demoField, field])
        );
        const commandOutput = browser.querySelector("[data-command-output]");
        const copyButton = browser.querySelector("[data-copy-command]");
        const copyStatus = browser.querySelector("[data-copy-status]");
        const selectedName = browser.querySelector("[data-demo-name]:not([data-demo-path])");
        const selectedDescription = browser.querySelector("[data-demo-description]:not([data-demo-path])");
        let selectedCard = cards[0];

        const populateSelect = (select, values, preferredValues) => {
            const previousValue = select.value;
            select.replaceChildren(...values.map((value) => new Option(value, value)));
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
            if (fields.physics.value === "isaacsim_physx" || fields.visualizer.value === "kit") {
                requiredExtras.add("isaacsim");
            }
            if (fields.physics.value === "ovphysx") {
                requiredExtras.add("ovphysx");
            }
            if (["rerun", "viser"].includes(fields.visualizer.value)) {
                requiredExtras.add(fields.visualizer.value);
            }
            const extraOrder = ["isaacsim", "ovphysx", "tetrahedralization", "teleop", "rerun", "viser"];
            const extras = extraOrder.filter((extra) => requiredExtras.has(extra));

            const parts = ["uv", "run"];
            if (extras.length) {
                parts.push("--extra", extras.join(","));
            }
            parts.push("python", selectedCard.dataset.demoPath);
            if (selectedCard.dataset.demoFixedPhysics !== "true") {
                parts.push("--physics", fields.physics.value);
            }
            parts.push("--viz", fields.visualizer.value);
            parts.push(...(selectedCard.dataset.demoArgs || "").split(" ").filter(Boolean));
            return parts.join(" ");
        };

        const updateCommand = () => {
            commandOutput.textContent = currentCommand();
        };

        const updateVisualizer = () => {
            const preferredVisualizers = fields.physics.value.startsWith("newton")
                ? ["newton_gl", "kit", "none"]
                : fields.physics.value === "ovphysx" ? ["none"] : ["kit", "none"];
            populateSelect(fields.visualizer, compatibleVisualizers(), preferredVisualizers);
            updateCommand();
        };

        const selectDemo = (card) => {
            selectedCard = card;
            for (const candidate of cards) {
                const isSelected = candidate === selectedCard;
                candidate.classList.toggle("is-selected", isSelected);
                candidate.setAttribute("aria-pressed", String(isSelected));
            }
            selectedName.textContent = selectedCard.dataset.demoName;
            selectedDescription.textContent = selectedCard.dataset.demoDescription;
            populateSelect(fields.physics, splitValues(selectedCard.dataset.demoPhysics), ["isaacsim_physx", "newton_mjwarp"]);
            updateVisualizer();
        };

        for (const card of cards) {
            card.addEventListener("click", () => selectDemo(card));
        }
        fields.physics.addEventListener("change", updateVisualizer);
        fields.visualizer.addEventListener("change", updateCommand);
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
