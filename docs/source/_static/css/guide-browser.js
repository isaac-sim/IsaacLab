/*
 * Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

(() => {
    "use strict";

    const embeddedGuideStyles = `
        html, body {
            overflow: hidden;
            background: transparent;
        }
        #pst-header,
        #pst-primary-sidebar,
        #pst-secondary-sidebar,
        .bd-header-article,
        .onlyprint,
        .prev-next-footer,
        .bd-footer,
        footer.bd-footer-article {
            display: none !important;
        }
        .bd-container,
        .bd-container__inner,
        .bd-page-width,
        .bd-main,
        .bd-content,
        .bd-article-container,
        .bd-article {
            width: 100% !important;
            max-width: none !important;
            margin: 0 !important;
            padding: 0 !important;
        }
    `;

    const initializeGuideBrowser = () => {
        const browser = document.querySelector("[data-guide-browser]");
        const guideList = document.querySelector(".guide-list");
        const viewer = document.querySelector("[data-guide-viewer]");
        if (!browser || !guideList || !viewer) {
            return;
        }

        const intro = document.querySelector(".guide-browser-intro");
        const note = document.querySelector(".guide-browser-note");
        const search = browser.querySelector("[data-guide-search]");
        const count = browser.querySelector("[data-guide-count]");
        const empty = document.querySelector("[data-guide-empty]");
        const backButton = viewer.querySelector("[data-guide-back]");
        const openPage = viewer.querySelector("[data-guide-open-page]");
        const loading = viewer.querySelector("[data-guide-loading]");
        const frame = viewer.querySelector("[data-guide-frame]");
        const groups = [...guideList.querySelectorAll(".guide-group")];
        const entries = [...guideList.querySelectorAll(".guide-entry")];
        let selectedEntry = null;
        let frameObserver = null;

        const normalizedTokens = (value) => value.toLowerCase().trim().split(/\s+/).filter(Boolean);

        const render = () => {
            const tokens = normalizedTokens(search.value);
            let visibleCount = 0;
            for (const entry of entries) {
                const searchableText = entry.textContent.toLowerCase();
                const matchesSearch = tokens.every((token) => searchableText.includes(token));
                entry.hidden = !matchesSearch;
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

        const setBrowserVisible = (isVisible) => {
            intro.hidden = !isVisible;
            browser.hidden = !isVisible;
            guideList.hidden = !isVisible;
            note.hidden = !isVisible;
            empty.hidden = !isVisible || entries.some((entry) => !entry.hidden);
            viewer.hidden = isVisible;
        };

        const closeGuide = () => {
            frameObserver?.disconnect();
            frameObserver = null;
            frame.hidden = true;
            frame.removeAttribute("src");
            setBrowserVisible(true);
            selectedEntry?.focus();
            selectedEntry = null;
        };

        const resizeFrame = () => {
            const frameDocument = frame.contentDocument;
            if (!frameDocument) {
                return;
            }
            const height = Math.ceil(frameDocument.documentElement.scrollHeight);
            if (height > 0 && frame.style.height !== `${height}px`) {
                frame.style.height = `${height}px`;
            }
        };

        const scrollToFrameTarget = (hash) => {
            const frameDocument = frame.contentDocument;
            if (!frameDocument || !hash) {
                return;
            }
            const target = frameDocument.getElementById(decodeURIComponent(hash.slice(1)));
            if (!target) {
                return;
            }
            frame.contentWindow.scrollTo(0, 0);
            const targetTop = window.scrollY + frame.getBoundingClientRect().top + target.getBoundingClientRect().top;
            window.scrollTo({top: Math.max(0, targetTop - 80), behavior: "smooth"});
        };

        const prepareFrame = () => {
            if (viewer.hidden || !frame.contentDocument) {
                return;
            }

            const frameDocument = frame.contentDocument;
            const style = frameDocument.createElement("style");
            style.textContent = embeddedGuideStyles;
            frameDocument.head.append(style);

            frameDocument.addEventListener("click", (event) => {
                const anchor = event.target.closest("a[href]");
                if (!anchor) {
                    return;
                }
                const destination = new URL(anchor.href, frame.contentWindow.location.href);
                if (destination.origin !== window.location.origin) {
                    anchor.target = "_blank";
                    anchor.rel = "noopener";
                } else if (destination.pathname === window.location.pathname) {
                    event.preventDefault();
                    closeGuide();
                } else if (destination.pathname === frame.contentWindow.location.pathname && destination.hash) {
                    event.preventDefault();
                    scrollToFrameTarget(destination.hash);
                }
            });

            const title = frameDocument.querySelector("article.bd-article h1")?.firstChild?.textContent.trim();
            if (title) {
                frame.title = title;
            }
            openPage.href = frame.contentWindow.location.href;
            loading.hidden = true;
            frame.hidden = false;
            resizeFrame();

            frameObserver?.disconnect();
            frameObserver = new frame.contentWindow.ResizeObserver(resizeFrame);
            frameObserver.observe(frameDocument.documentElement);
            frameDocument.fonts?.ready.then(resizeFrame);
            requestAnimationFrame(() => scrollToFrameTarget(frame.contentWindow.location.hash));
        };

        const openGuide = (link) => {
            selectedEntry = link;
            openPage.href = link.href;
            frame.title = link.textContent.trim();
            frame.style.height = "0";
            frame.hidden = true;
            loading.hidden = false;
            setBrowserVisible(false);
            frame.src = link.href;
            viewer.scrollIntoView({block: "start"});
            backButton.focus({preventScroll: true});
        };

        for (const entry of entries) {
            const link = entry.querySelector("a[href]");
            if (!link) {
                continue;
            }
            link.addEventListener("click", (event) => {
                if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
                    return;
                }
                event.preventDefault();
                openGuide(link);
            });
        }

        backButton.addEventListener("click", closeGuide);
        frame.addEventListener("load", prepareFrame);
        search.addEventListener("input", render);
        render();
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initializeGuideBrowser, {once: true});
    } else {
        initializeGuideBrowser();
    }
})();
