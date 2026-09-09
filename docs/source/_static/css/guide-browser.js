/*
 * Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

(() => {
    "use strict";

    const initializeGuideBrowser = () => {
        const browser = document.querySelector("[data-guide-browser]");
        const guideList = document.querySelector(".guide-list");
        if (!browser || !guideList) {
            return;
        }

        const search = browser.querySelector("[data-guide-search]");
        const typeButtons = [...browser.querySelectorAll("[data-guide-type]")];
        const count = browser.querySelector("[data-guide-count]");
        const empty = document.querySelector("[data-guide-empty]");
        const groups = [...guideList.querySelectorAll(".guide-group")];
        const entries = [...guideList.querySelectorAll(".guide-entry")];
        let selectedType = "all";

        const normalizedTokens = (value) => value.toLowerCase().trim().split(/\s+/).filter(Boolean);

        const render = () => {
            const tokens = normalizedTokens(search.value);
            let visibleCount = 0;
            for (const entry of entries) {
                const matchesType = selectedType === "all" || entry.classList.contains(`guide-${selectedType}`);
                const searchableText = entry.textContent.toLowerCase();
                const matchesSearch = tokens.every((token) => searchableText.includes(token));
                entry.hidden = !(matchesType && matchesSearch);
                visibleCount += entry.hidden ? 0 : 1;
            }
            for (const group of groups) {
                const visibleEntries = [...group.querySelectorAll(".guide-entry:not([hidden])")];
                for (const entry of group.querySelectorAll(".guide-entry")) {
                    entry.classList.remove("is-first-visible", "is-last-visible");
                }
                visibleEntries[0]?.classList.add("is-first-visible");
                visibleEntries.at(-1)?.classList.add("is-last-visible");
                group.hidden = visibleEntries.length === 0;
            }
            count.textContent = `${visibleCount} ${visibleCount === 1 ? "guide" : "guides"}`;
            empty.hidden = visibleCount !== 0;
        };

        for (const button of typeButtons) {
            button.addEventListener("click", () => {
                selectedType = button.dataset.guideType;
                for (const candidate of typeButtons) {
                    const isSelected = candidate === button;
                    candidate.classList.toggle("is-active", isSelected);
                    candidate.setAttribute("aria-pressed", String(isSelected));
                }
                render();
            });
        }
        search.addEventListener("input", render);
        render();
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initializeGuideBrowser, {once: true});
    } else {
        initializeGuideBrowser();
    }
})();
