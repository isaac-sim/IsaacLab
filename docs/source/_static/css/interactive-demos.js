/*
 * Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

(() => {
    "use strict";

    const namespace = "isaaclab-demos";
    const anchorPrefix = "#interactive-demo-";

    const initialize = () => {
        const embeds = [...document.querySelectorAll("[data-isaaclab-demos]")].map((element) => {
            const url = new URL(element.dataset.baseUrl, document.baseURI);
            url.searchParams.set("embed", "1");
            url.searchParams.set("demos", element.dataset.demoIds || "all");
            url.searchParams.set("parentOrigin", location.origin);
            for (const key of ["physics", "activity"]) {
                if (element.dataset[key]) url.searchParams.set(key, element.dataset[key]);
            }
            return {
                element,
                iframe: element.querySelector("iframe"),
                status: element.querySelector("[role=status]"),
                ids: element.dataset.demoIds ? element.dataset.demoIds.split(",") : [],
                url,
                started: false,
                ready: false,
                selection: "",
                height: 320,
                timer: null,
            };
        });
        if (!embeds.length) return;

        const theme = () => {
            const value = document.documentElement.dataset.theme;
            return value === "dark" || (!value && window.matchMedia("(prefers-color-scheme: dark)").matches)
                ? "dark" : "light";
        };
        const themeMessage = () => {
            const root = getComputedStyle(document.documentElement);
            const body = getComputedStyle(document.body);
            const colors = {
                background: "--pst-color-background",
                surface: "--pst-color-on-background",
                mutedSurface: "--pst-color-surface",
                text: "--pst-color-text-base",
                muted: "--pst-color-text-muted",
                border: "--pst-color-border",
                accent: "--pst-color-primary",
            };
            const styles = Object.fromEntries(Object.entries(colors)
                .map(([key, variable]) => [key, root.getPropertyValue(variable).trim()]));
            return {type: "theme", theme: theme(), styles: {
                ...styles, fontFamily: body.fontFamily, fontSize: body.fontSize, lineHeight: body.lineHeight,
            }};
        };
        const send = (embed, message) => {
            if (embed.ready) {
                embed.iframe.contentWindow.postMessage({namespace, ...message}, embed.url.origin);
            }
        };
        const showError = (embed, message) => {
            window.clearTimeout(embed.timer);
            embed.element.dataset.state = "error";
            embed.iframe.setAttribute("aria-busy", "false");
            embed.status.textContent = `${message} You can still open the demo gallery in a new tab.`;
        };
        const start = (embed) => {
            if (embed.started) return;
            embed.started = true;
            embed.status.textContent = "Loading interactive demos…";
            embed.iframe.src = embed.url.href;
            embed.iframe.addEventListener("error", () => showError(embed, "The embedded gallery could not load."));
            embed.timer = window.setTimeout(() => {
                if (!embed.ready) showError(embed, "The embedded gallery is taking longer than expected to load.");
            }, 20000);
        };

        window.addEventListener("message", (event) => {
            const embed = embeds.find((candidate) => candidate.started
                && event.source === candidate.iframe.contentWindow && event.origin === candidate.url.origin);
            const message = event.data;
            if (!embed || !message || message.namespace !== namespace) return;
            if (message.type === "ready") {
                embed.ready = true;
                window.clearTimeout(embed.timer);
                embed.element.dataset.state = "ready";
                embed.iframe.setAttribute("aria-busy", "false");
                embed.status.textContent = "Choose a demo to start the simulation.";
                send(embed, themeMessage());
                if (embed.selection) send(embed, {type: "select", demo: embed.selection});
            } else if (message.type === "resize" && Number.isFinite(message.height) && message.height > 0) {
                const height = Math.max(1, Math.ceil(message.height));
                if (Math.abs(height - embed.height) > 1) {
                    embed.height = height;
                    embed.iframe.style.height = `${height}px`;
                }
            } else if (message.type === "error" && !embed.ready) {
                const detail = typeof message.message === "string" ? message.message.slice(0, 240) : "";
                showError(embed, detail || "The interactive demo could not load.");
            }
        });

        new MutationObserver(() => {
            const message = themeMessage();
            for (const embed of embeds) send(embed, message);
        }).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});

        const selectAnchor = (hash) => {
            if (!hash.startsWith(anchorPrefix)) return;
            const id = hash.slice(anchorPrefix.length).replaceAll("-", "_");
            const embed = embeds.find((candidate) => candidate.ids.includes(id))
                || embeds.find((candidate) => !candidate.ids.length);
            if (!embed) return;
            embed.selection = id;
            start(embed);
            send(embed, {type: "select", demo: id});
            embed.element.scrollIntoView({block: "start"});
        };
        window.addEventListener("hashchange", () => selectAnchor(location.hash));
        document.addEventListener("click", (event) => {
            if (!(event.target instanceof Element) || event.button || event.ctrlKey || event.metaKey
                || event.shiftKey || event.altKey) return;
            const link = event.target.closest("a[href]");
            if (link && link.origin === location.origin && link.pathname === location.pathname
                && link.hash === location.hash && link.hash.startsWith(anchorPrefix)) {
                event.preventDefault();
                selectAnchor(link.hash);
            }
        });

        if ("IntersectionObserver" in window) {
            const observer = new IntersectionObserver((entries) => {
                for (const entry of entries) {
                    if (entry.isIntersecting) {
                        start(embeds.find((embed) => embed.element === entry.target));
                        observer.unobserve(entry.target);
                    }
                }
            }, {rootMargin: "200px"});
            for (const embed of embeds) observer.observe(embed.element);
        } else {
            for (const embed of embeds) start(embed);
        }
        selectAnchor(location.hash);
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initialize, {once: true});
    } else {
        initialize();
    }
})();
