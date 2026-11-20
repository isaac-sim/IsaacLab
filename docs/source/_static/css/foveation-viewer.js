/*
 * Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Interactive sampling-grid and pooling viewers for the Foveated Camera page.
// The sampling follows fovi's SamplingCoords with spherical field geometry, as the camera configures it.
(() => {
    "use strict";

    const DEG = Math.PI / 180;
    const TAU = 2 * Math.PI;
    // fovi's isotropic pooling neighbourhoods for the pooling viewer, beside this script; fetched only when used.
    const SCRIPT_URL = typeof document !== "undefined" && document.currentScript ? document.currentScript.src : null;
    const TABLES_URL = SCRIPT_URL ? new URL("foveation-pooling-isotropic.json", SCRIPT_URL).href : null;

    // -- Sampling geometry -------------------------------------------------------------------------------------------

    // Warped Cartesian grid: an N x N lattice in the native plane, mapped radially by the integrated CMF.
    // Returns samples in canonical order (x index major), each with normalized visual coordinates s = (x, y) / r_max.
    const warpedGrid = (fov, a, res) => {
        const half = fov / 2;
        const growth = 1 + half / a;
        const samples = [];
        for (let i = 0; i < res; i++) {
            for (let j = 0; j < res; j++) {
                const cx = -1 + (2 * i + 1) / res;
                const cy = -1 + (2 * j + 1) / res;
                const rho = Math.hypot(cx, cy);
                const r = a * (Math.pow(growth, rho) - 1);
                const theta = Math.atan2(cy, cx);
                samples.push({i, j, native: [cx, cy], s: [(r / half) * Math.cos(theta), (r / half) * Math.sin(theta)], valid: rho <= 1 + 1e-12});
            }
        }
        return samples;
    };

    // Isotropic rings: radii even in cortical distance w = log(r + a), angles per ring matched to the radial step.
    const isotropicRings = (fov, a, res) => {
        if (res === 1) {
            return {radius: [0], count: [1]};
        }
        const half = fov / 2;
        const wMin = Math.log(a);
        const wDelta = (Math.log(half + a) - wMin) / (res - 1);
        const radius = Array.from({length: res + 1}, (_, k) => Math.exp(wMin + k * wDelta) - a);
        const count = [1];
        for (let k = 1; k < res; k++) {
            const step = (radius[k + 1] - radius[k - 1]) / 2;
            const tangential = Math.sin(radius[k] * DEG) / DEG;  // spherical field geometry
            count.push(Math.ceil((TAU * tangential) / step));
        }
        return {radius: radius.slice(0, res).map((r) => r / half), count};
    };

    const isotropicCount = (fov, a, res) => isotropicRings(fov, a, res).count.reduce((sum, n) => sum + n, 0);

    // The camera matches an isotropic retina to foveated_res^2 samples (fovi's auto_match_num_coords).
    const matchedIsotropicRes = (fov, a, target) => {
        let left = 1;
        let right = 1000;
        let best = 1;
        let bestDiff = Infinity;
        while (left <= right) {
            const mid = (left + right) >> 1;
            const n = isotropicCount(fov, a, mid);
            const diff = Math.abs(n - target);
            if (diff < bestDiff) {
                bestDiff = diff;
                best = mid;
            }
            if (n < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        while (best > 1 && isotropicCount(fov, a, best) > target) {
            best -= 1;
        }
        return best;
    };

    const isotropicSamples = (fov, a, rings) => {
        const {radius, count} = isotropicRings(fov, a, rings);
        const samples = [];
        radius.forEach((r, ring) => {
            for (let k = 0; k < count[ring]; k++) {
                const angle = (k * TAU) / count[ring];
                samples.push({ring, s: [r * Math.cos(angle), r * Math.sin(angle)], valid: true});
            }
        });
        return samples;
    };

    // Unit ray of a normalized visual coordinate, in the image frame: x right, y down, z forward.
    const ray = (s, fov) => {
        const e = Math.hypot(s[0], s[1]) * (fov / 2) * DEG;
        const theta = Math.atan2(s[1], s[0]);
        return [Math.sin(e) * Math.cos(theta), -Math.sin(e) * Math.sin(theta), Math.cos(e)];
    };

    // Pan-then-tilt gaze rotation (row-major), as isaaclab_fovi's gaze kernels: positive tilt looks up, pan left.
    const gazeRotation = (tilt, pan) => {
        const x = -Math.cos(tilt * DEG) * Math.sin(pan * DEG);
        const y = -Math.sin(tilt * DEG);
        const z = Math.cos(tilt * DEG) * Math.cos(pan * DEG);
        const pitch = Math.atan2(x, z);
        const roll = Math.atan2(-y, Math.hypot(x, z));
        const [cp, sp, cr, sr] = [Math.cos(pitch), Math.sin(pitch), Math.cos(roll), Math.sin(roll)];
        return [[cp, sp * sr, sp * cr], [0, cr, -sr], [-sp, cp * sr, cp * cr]];
    };

    const rotate = (m, v) => m.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);

    // An ideal square camera whose field of view spans its side; the retina spans the same angle.
    const SIDE = 1000;
    const cameraModels = {
        fisheye: {
            label: "Equidistant fisheye",
            fov: {min: 40, max: 240, initial: 180},
            focal: (fov) => SIDE / 2 / ((fov / 2) * DEG),
            project(d, f) {
                const lateral = Math.hypot(d[0], d[1]);
                const theta = Math.atan2(lateral, d[2]);
                if (lateral < 1e-12) {
                    return [SIDE / 2, SIDE / 2];
                }
                return [SIDE / 2 + (f * theta * d[0]) / lateral, SIDE / 2 + (f * theta * d[1]) / lateral];
            },
            unproject(p, f) {
                const dx = p[0] - SIDE / 2;
                const dy = p[1] - SIDE / 2;
                const theta = Math.hypot(dx, dy) / f;
                const lateral = Math.hypot(dx, dy) || 1;
                return [(Math.sin(theta) * dx) / lateral, (Math.sin(theta) * dy) / lateral, Math.cos(theta)];
            },
        },
        pinhole: {
            label: "Pinhole",
            fov: {min: 20, max: 170, initial: 80},
            focal: (fov) => SIDE / 2 / Math.tan((fov / 2) * DEG),
            project(d, f) {
                if (d[2] <= 1e-9) {
                    return null;
                }
                return [SIDE / 2 + (f * d[0]) / d[2], SIDE / 2 + (f * d[1]) / d[2]];
            },
            unproject(p, f) {
                return normalize([(p[0] - SIDE / 2) / f, (p[1] - SIDE / 2) / f, 1]);
            },
        },
    };

    const normalize = (v) => {
        const n = Math.hypot(v[0], v[1], v[2]);
        return [v[0] / n, v[1] / n, v[2] / n];
    };

    const inImage = (p) => p !== null && p[0] >= 0 && p[0] < SIDE && p[1] >= 0 && p[1] < SIDE;

    const imageBorder = (model, f) => {
        const points = [];
        const corners = [[0, 0], [SIDE, 0], [SIDE, SIDE], [0, SIDE], [0, 0]];
        for (let k = 0; k < 4; k++) {
            for (let t = 0; t < 40; t++) {
                const [a, b] = [corners[k], corners[k + 1]];
                points.push(model.unproject([a[0] + ((b[0] - a[0]) * t) / 40, a[1] + ((b[1] - a[1]) * t) / 40], f));
            }
        }
        points.push(points[0]);
        return points;
    };

    // Spacing of neighbouring samples [deg] at the fixation and at the edge of the field.
    const spacing = (style, fov, a, res) => {
        const half = fov / 2;
        if (style === "isotropic") {
            const step = (Math.log(half + a) - Math.log(a)) / Math.max(res - 1, 1);
            return [a * (Math.exp(step) - 1), (half + a) * (1 - Math.exp(-step))];
        }
        const rate = Math.log(1 + half / a) * (2 / res);
        return [a * rate, (half + a) * rate];
    };

    // -- Drawing helpers ---------------------------------------------------------------------------------------------

    const palette = (element) => {
        const style = getComputedStyle(element);
        const get = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
        return {
            ink: get("--fovi-ink", "#1f2a37"),
            muted: get("--fovi-muted", "#8a96a8"),
            faint: get("--fovi-faint", "#d7dee8"),
            surface: get("--fovi-surface", "#ffffff"),
            accent: get("--fovi-accent", "#2458d3"),
            camera: get("--fovi-camera", "#167d72"),
            highlight: get("--fovi-highlight", "#d47b00"),
            body: get("--fovi-body", "#29364b"),
        };
    };

    const fitCanvas = (canvas) => {
        const box = canvas.getBoundingClientRect();
        const ratio = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = Math.max(1, Math.round(box.width * ratio));
        canvas.height = Math.max(1, Math.round(box.height * ratio));
        const ctx = canvas.getContext("2d");
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        return {ctx, w: box.width, h: box.height};
    };

    const polyline = (ctx, points, color, width, dash = []) => {
        if (points.length < 2) {
            return;
        }
        ctx.beginPath();
        ctx.setLineDash(dash);
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        points.forEach((p, k) => (k === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1])));
        ctx.stroke();
        ctx.setLineDash([]);
    };

    const dots = (ctx, points, color, radius, alpha = 1) => {
        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;
        ctx.beginPath();
        for (const p of points) {
            ctx.moveTo(p[0] + radius, p[1]);
            ctx.arc(p[0], p[1], radius, 0, TAU);
        }
        ctx.fill();
        ctx.globalAlpha = 1;
    };

    // Orbiting orthographic view of the viewing sphere; the camera sits at its centre looking along +z.
    const sphereProjector = (view, w, h) => {
        const scale = Math.min(w, h) * 0.4;
        const [cy, sy, cp, sp] = [Math.cos(view.yaw), Math.sin(view.yaw), Math.cos(view.pitch), Math.sin(view.pitch)];
        return ([x, y, z]) => {
            const rx = cy * x - sy * z;
            const rz = sy * x + cy * z;
            const ry = cp * y - sp * rz;
            const depth = sp * y + cp * rz;
            return [w / 2 + rx * scale, h / 2 + ry * scale, depth];
        };
    };

    const bindOrbit = (canvas, view, redraw, initial) => {
        let last = null;
        canvas.addEventListener("pointerdown", (event) => {
            last = [event.clientX, event.clientY];
            canvas.setPointerCapture(event.pointerId);
        });
        canvas.addEventListener("pointermove", (event) => {
            if (!last) {
                return;
            }
            view.yaw += (event.clientX - last[0]) * 0.01;
            view.pitch = Math.max(-1.5, Math.min(1.5, view.pitch - (event.clientY - last[1]) * 0.01));
            last = [event.clientX, event.clientY];
            redraw();
        });
        for (const name of ["pointerup", "pointercancel"]) {
            canvas.addEventListener(name, () => (last = null));
        }
        canvas.addEventListener("dblclick", () => {
            Object.assign(view, initial);
            redraw();
        });
    };

    const drawSphere = (canvas, view, scene, colors) => {
        const {ctx, w, h} = fitCanvas(canvas);
        const project = sphereProjector(view, w, h);
        // The viewer looks from behind the camera, so smaller depths are nearer; nothing is opaque but the camera.
        const segment = (points, color, width) => polyline(ctx, points.map(project), color, width);
        const circle = (polar) => Array.from({length: 121}, (_, k) => {
            const az = (k * TAU) / 120;
            return [Math.sin(polar) * Math.cos(az), Math.sin(polar) * Math.sin(az), Math.cos(polar)];
        });
        for (const polar of [30, 60, 90, 120, 150]) {
            segment(circle(polar * DEG), colors.faint, polar === 90 ? 1.2 : 0.7);
        }
        for (let k = 0; k < 12; k++) {
            const az = (k * TAU) / 12;
            segment(Array.from({length: 61}, (_, t) => {
                const polar = (t * Math.PI) / 60;
                return [Math.sin(polar) * Math.cos(az), Math.sin(polar) * Math.sin(az), Math.cos(polar)];
            }), colors.faint, 0.7);
        }
        segment(scene.border, colors.camera, 2);
        const radius = scene.dotRadius;
        dots(ctx, scene.rays.filter((r) => !r.valid).map((r) => project(r.dir)), colors.muted, radius, 0.5);
        dots(ctx, scene.rays.filter((r) => r.valid).map((r) => project(r.dir)), colors.accent, radius, 0.7);
        for (const line of scene.lines) {
            segment(line, colors.accent, 0.8);
        }
        segment(scene.retinaEdge, colors.accent, 1.6);
        const gaze = project(scene.gaze);
        dots(ctx, [gaze], colors.surface, 4.5);
        ctx.beginPath();
        ctx.arc(gaze[0], gaze[1], 4.5, 0, TAU);
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 2;
        ctx.stroke();
        // Camera housing at the centre.
        const body = [[-0.09, -0.06, -0.22], [0.09, -0.06, -0.22], [0.09, 0.06, -0.22], [-0.09, 0.06, -0.22],
            [-0.09, -0.06, -0.06], [0.09, -0.06, -0.06], [0.09, 0.06, -0.06], [-0.09, 0.06, -0.06]];
        const faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
            .map((face) => face.map((k) => project(body[k])))
            .sort((a, b) => b.reduce((s, p) => s + p[2], 0) - a.reduce((s, p) => s + p[2], 0));
        for (const face of faces) {
            ctx.beginPath();
            face.forEach((p, k) => (k === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1])));
            ctx.closePath();
            ctx.fillStyle = colors.body;
            ctx.fill();
            ctx.strokeStyle = colors.faint;
            ctx.lineWidth = 0.6;
            ctx.stroke();
        }
    };

    // -- Sampling viewer ---------------------------------------------------------------------------------------------

    const control = (label, input, output) => {
        const wrapper = document.createElement("label");
        wrapper.className = "fovi-control";
        const name = document.createElement("span");
        name.className = "fovi-control-name";
        name.innerHTML = label;
        wrapper.append(name, input);
        if (output) {
            wrapper.append(output);
        }
        return wrapper;
    };

    const slider = (min, max, step, value) => {
        const input = document.createElement("input");
        Object.assign(input, {type: "range", min, max, step, value});
        return input;
    };

    const select = (options, value) => {
        const input = document.createElement("select");
        for (const [key, text] of options) {
            input.append(new Option(text, key));
        }
        input.value = value;
        return input;
    };

    const readout = () => {
        const output = document.createElement("output");
        output.className = "fovi-control-value";
        return output;
    };

    const formatA = (a) => (a >= 100 ? a.toFixed(0) : a >= 10 ? a.toFixed(1) : a >= 1 ? a.toFixed(2) : a.toFixed(3));

    const initSamplingViewer = (root) => {
        const defaults = {camera: "fisheye", style: "warped_cartesian_as_grid", logA: Math.log10(0.591), res: 32, tilt: 0, pan: 0};
        const state = {...defaults};
        const inputs = {
            camera: select(Object.entries(cameraModels).map(([key, model]) => [key, model.label]), state.camera),
            style: select([["warped_cartesian_as_grid", "warped_cartesian_as_grid"], ["isotropic", "isotropic"]], state.style),
            logA: slider(-2, 3, 0.01, state.logA),
            res: slider(8, 128, 1, state.res),
            tilt: slider(-60, 60, 1, state.tilt),
            pan: slider(-60, 60, 1, state.pan),
        };
        const outputs = {logA: readout(), res: readout(), tilt: readout(), pan: readout()};
        const controls = document.createElement("div");
        controls.className = "fovi-controls";
        controls.append(
            control("Physical camera", inputs.camera),
            control("<code>foveated_style</code>", inputs.style),
            control("<code>cmf_a_deg</code>", inputs.logA, outputs.logA),
            control("<code>foveated_res</code>", inputs.res, outputs.res),
            control("Gaze tilt", inputs.tilt, outputs.tilt),
            control("Gaze pan", inputs.pan, outputs.pan),
        );
        const reset = document.createElement("button");
        reset.type = "button";
        reset.className = "fovi-reset";
        reset.textContent = "Reset";
        controls.append(reset);
        const summary = document.createElement("p");
        summary.className = "fovi-summary";
        summary.setAttribute("aria-live", "polite");
        const panels = document.createElement("div");
        panels.className = "fovi-panels fovi-panels-stacked";
        const initialView = {yaw: -0.6, pitch: 0.35};
        const cameras = Object.entries(cameraModels).map(([key, model]) => {
            const panel = document.createElement("figure");
            panel.className = "fovi-panel";
            const title = document.createElement("figcaption");
            title.textContent = model.label;
            const sphere = document.createElement("canvas");
            sphere.className = "fovi-canvas fovi-sphere";
            sphere.setAttribute("aria-label", `${model.label}: the retina's rays on the viewing sphere. Drag to rotate; double-click to reset.`);
            const image = document.createElement("canvas");
            image.className = "fovi-canvas fovi-image";
            image.setAttribute("aria-label", `${model.label}: where the retina's samples land in the camera image.`);
            const labels = document.createElement("div");
            labels.className = "fovi-panel-labels";
            labels.innerHTML = "<span>Viewing sphere (drag to rotate)</span><span>Camera image</span>";
            const views = document.createElement("div");
            views.className = "fovi-views";
            views.append(sphere, image);
            const note = document.createElement("p");
            note.className = "fovi-panel-note";
            // Each camera has its own field of view; the retina spans it.
            const fovInput = slider(model.fov.min, model.fov.max, 1, model.fov.initial);
            const fovOutput = readout();
            const fovControl = control("Field of view", fovInput, fovOutput);
            fovControl.classList.add("fovi-panel-control");
            panel.append(title, fovControl, views, labels, note);
            panels.append(panel);
            const view = {...initialView};
            // One camera shows at a time; each keeps its own field of view.
            const camera = {key, model, panel, sphere, image, note, view, fov: model.fov.initial, fovInput, fovOutput};
            fovInput.addEventListener("input", () => {
                camera.fov = Number(fovInput.value);
                sync();
            });
            return camera;
        });
        const legend = document.createElement("p");
        legend.className = "fovi-legend";
        legend.innerHTML = '<span class="fovi-key fovi-key-accent"></span>retinal sample'
            + '<span class="fovi-key fovi-key-muted"></span>outside the camera image (masked with <code>mask_to_sensor_fov</code>)'
            + '<span class="fovi-key fovi-key-camera"></span>camera image border';
        root.append(controls, summary, panels, legend);

        const samplesCache = new Map();
        const samplesFor = (fov) => {
            const a = Math.pow(10, state.logA);
            const key = `${state.style}/${a}/${state.res}/${fov}`;
            if (samplesCache.has(key)) {
                return samplesCache.get(key);
            }
            let samples;
            let description;
            let resolution = state.res;
            if (state.style === "isotropic") {
                resolution = matchedIsotropicRes(fov, a, state.res * state.res);
                samples = isotropicSamples(fov, a, resolution);
                description = `${samples.length.toLocaleString("en-US")} samples on ${resolution} rings, published as (num_envs, ${samples.length}, 1, C)`;
            } else {
                samples = warpedGrid(fov, a, state.res);
                const valid = samples.filter((s) => s.valid).length;
                description = `${state.res} × ${state.res} grid, ${valid.toLocaleString("en-US")} samples inside the circular field, published as (num_envs, ${state.res}, ${state.res}, C)`;
            }
            const [centre, edge] = spacing(state.style, fov, a, resolution);
            const entry = {samples: samples.filter((s) => s.valid), all: samples, description, centre, edge};
            samplesCache.set(key, entry);
            if (samplesCache.size > 4) {
                samplesCache.delete(samplesCache.keys().next().value);  // keep the recent settings of both cameras
            }
            return entry;
        };

        // The retina's rays, outline and warped-grid lattice lines (every few rows and columns) for a field of view.
        const retinaFor = (fov, rotation) => {
            const data = samplesFor(fov);
            const rays = data.samples.map((s) => rotate(rotation, ray(s.s, fov)));
            const edge = Array.from({length: 181}, (_, k) => rotate(rotation, ray([Math.cos((k * TAU) / 180), Math.sin((k * TAU) / 180)], fov)));
            const lines = [];
            if (state.style !== "isotropic") {
                const every = Math.max(1, Math.round(state.res / 16));
                const index = new Map(data.all.map((s) => [s.i * state.res + s.j, s]));
                for (let fixed = Math.floor(every / 2); fixed < state.res; fixed += every) {
                    for (const transpose of [false, true]) {
                        let run = [];
                        for (let moving = 0; moving < state.res; moving++) {
                            const s = index.get(transpose ? moving * state.res + fixed : fixed * state.res + moving);
                            if (s && s.valid) {
                                run.push(rotate(rotation, ray(s.s, fov)));
                            } else if (run.length) {
                                lines.push(run);
                                run = [];
                            }
                        }
                        if (run.length) {
                            lines.push(run);
                        }
                    }
                }
            }
            return {data, rays, edge, lines};
        };

        const redraw = () => {
            const colors = palette(root);
            const rotation = gazeRotation(state.tilt, state.pan);
            const gaze = rotate(rotation, [0, 0, 1]);
            summary.textContent = "The retina spans the physical camera's field of view; switch the camera or move its slider to compare.";
            for (const camera of cameras) {
                if (camera.key !== state.camera) {
                    continue;
                }
                const {data, rays, edge, lines} = retinaFor(camera.fov, rotation);
                const count = rays.length;
                const dotRadius = Math.max(0.6, Math.min(2.4, 70 / Math.sqrt(count)));
                const f = camera.model.focal(camera.fov);
                const projected = rays.map((d) => camera.model.project(d, f));
                const valid = projected.map(inImage);
                const inside = valid.filter(Boolean).length;
                drawSphere(camera.sphere, camera.view, {
                    rays: rays.map((dir, k) => ({dir, valid: valid[k]})),
                    border: imageBorder(camera.model, f),
                    lines,
                    retinaEdge: edge,
                    gaze,
                    dotRadius,
                }, colors);
                drawImagePlane(camera.image, projected, valid, edge.map((d) => camera.model.project(d, f)), camera.model.project(gaze, f), dotRadius, colors);
                camera.note.innerHTML = `${data.description}. Sample spacing: <strong>${data.centre.toPrecision(2)}°</strong> at the fixation, `
                    + `<strong>${data.edge.toPrecision(2)}°</strong> at the edge of the ${camera.fov}° field. `
                    + `${inside.toLocaleString("en-US")} of ${count.toLocaleString("en-US")} samples inside the image (${((100 * inside) / count).toFixed(0)}%).`;
            }
        };

        const drawImagePlane = (canvas, projected, valid, edge, gaze, radius, colors) => {
            const {ctx, w, h} = fitCanvas(canvas);
            const margin = 0.18;
            const scale = Math.min(w, h) / (SIDE * (1 + 2 * margin));
            const ox = (w - SIDE * scale) / 2;
            const oy = (h - SIDE * scale) / 2;
            const map = (p) => [ox + p[0] * scale, oy + p[1] * scale];
            ctx.fillStyle = colors.surface;
            ctx.fillRect(ox, oy, SIDE * scale, SIDE * scale);
            ctx.strokeStyle = colors.camera;
            ctx.lineWidth = 2;
            ctx.strokeRect(ox, oy, SIDE * scale, SIDE * scale);
            polyline(ctx, [map([SIDE / 2, 0]), map([SIDE / 2, SIDE])], colors.faint, 0.7);
            polyline(ctx, [map([0, SIDE / 2]), map([SIDE, SIDE / 2])], colors.faint, 0.7);
            const visible = (p) => p !== null && p[0] > -SIDE && p[0] < 2 * SIDE && p[1] > -SIDE && p[1] < 2 * SIDE;
            ctx.save();
            ctx.beginPath();
            ctx.rect(0, 0, w, h);
            ctx.clip();
            dots(ctx, projected.filter((p, k) => !valid[k] && visible(p)).map(map), colors.muted, radius, 0.5);
            dots(ctx, projected.filter((p, k) => valid[k]).map(map), colors.accent, radius, 0.75);
            const edgeRuns = [];
            let run = [];
            for (const p of edge) {
                if (visible(p)) {
                    run.push(map(p));
                } else if (run.length) {
                    edgeRuns.push(run);
                    run = [];
                }
            }
            edgeRuns.push(run);
            for (const r of edgeRuns) {
                polyline(ctx, r, colors.accent, 1.4);
            }
            if (gaze && visible(gaze)) {
                const g = map(gaze);
                ctx.beginPath();
                ctx.arc(g[0], g[1], 4.5, 0, TAU);
                ctx.fillStyle = colors.surface;
                ctx.fill();
                ctx.strokeStyle = colors.accent;
                ctx.lineWidth = 2;
                ctx.stroke();
            }
            ctx.restore();
        };

        const sync = () => {
            const a = Math.pow(10, state.logA);
            outputs.logA.textContent = `${formatA(a)}°`;
            outputs.res.textContent = state.res;
            for (const camera of cameras) {
                camera.fovOutput.textContent = `${camera.fov}°`;
                camera.panel.hidden = camera.key !== state.camera;
            }
            outputs.tilt.textContent = `${state.tilt}°`;
            outputs.pan.textContent = `${state.pan}°`;
            redraw();
        };
        for (const [name, input] of Object.entries(inputs)) {
            input.addEventListener("input", () => {
                state[name] = name === "style" || name === "camera" ? input.value : Number(input.value);
                sync();
            });
        }
        reset.addEventListener("click", () => {
            Object.assign(state, defaults);
            for (const [name, input] of Object.entries(inputs)) {
                input.value = state[name];
            }
            for (const camera of cameras) {
                Object.assign(camera.view, initialView);
                camera.fov = camera.model.fov.initial;
                camera.fovInput.value = camera.fov;
            }
            sync();
        });
        for (const camera of cameras) {
            bindOrbit(camera.sphere, camera.view, redraw, initialView);
        }
        new ResizeObserver(() => redraw()).observe(panels);
        sync();
    };

    // -- Pooling viewer ----------------------------------------------------------------------------------------------

    // Upright layout of a canonical warped grid (fovi's sampling_as_grid): row = res - 1 - j, column = i.
    // Pooling windows follow fovi's GridPoolingLayer: stride = resolution multiple, SAME-style padding.
    const poolingWindow = (lowRes, mult, side, row, col) => {
        const stride = mult;
        const before = Math.floor((side - stride) / 2);
        const cells = [];
        for (let dr = 0; dr < side; dr++) {
            for (let dc = 0; dc < side; dc++) {
                const r = row * stride - before + dr;
                const c = col * stride - before + dc;
                if (r >= 0 && r < lowRes * mult && c >= 0 && c < lowRes * mult) {
                    cells.push([r, c]);
                }
            }
        }
        return cells;
    };

    const initPoolingViewer = (root) => {
        const FOV = Number(root.dataset.fov || 90);
        const A = Number(root.dataset.cmfA || 2);
        const LOW = Number(root.dataset.res || 16);
        const defaults = {style: "warped_cartesian_as_grid", mult: 2, side: 2, row: 8, col: 11, k: 4, space: "geodesic", unit: 120};
        const state = {...defaults};
        const inputs = {
            style: select([["warped_cartesian_as_grid", "warped_cartesian_as_grid"], ["isotropic", "isotropic"]], state.style),
            mult: select([["2", "2"], ["3", "3"], ["4", "4"]], String(state.mult)),
            side: slider(1, 9, 1, state.side),
            row: slider(0, LOW - 1, 1, state.row),
            col: slider(0, LOW - 1, 1, state.col),
            k: slider(1, 32, 1, state.k),
            space: select([["geodesic", "geodesic (default)"], ["visual", "visual field (False)"]], state.space),
            unit: slider(0, 1, 1, state.unit),
        };
        const outputs = {side: readout(), row: readout(), col: readout(), k: readout(), unit: readout()};
        const controls = document.createElement("div");
        controls.className = "fovi-controls";
        const gridControls = [
            control("Window side (<code>pool_k</code> = side²)", inputs.side, outputs.side),
            control("Pooled sample row", inputs.row, outputs.row),
            control("Pooled sample column", inputs.col, outputs.col),
        ];
        const isotropicControls = [
            control("<code>pool_k</code> (neighbours)", inputs.k, outputs.k),
            control("<code>pool_sample_cortex</code>", inputs.space),
            control("Pooled sample", inputs.unit, outputs.unit),
        ];
        controls.append(control("<code>foveated_style</code>", inputs.style), control("<code>pool_res_mult</code>", inputs.mult), ...gridControls, ...isotropicControls);
        const reset = document.createElement("button");
        reset.type = "button";
        reset.className = "fovi-reset";
        reset.textContent = "Reset";
        controls.append(reset);
        const summary = document.createElement("p");
        summary.className = "fovi-summary";
        summary.setAttribute("aria-live", "polite");
        const panels = document.createElement("div");
        panels.className = "fovi-panels fovi-panels-three";
        const makePanel = (title, label) => {
            const panel = document.createElement("figure");
            panel.className = "fovi-panel";
            const caption = document.createElement("figcaption");
            caption.textContent = title;
            const canvas = document.createElement("canvas");
            canvas.className = "fovi-canvas fovi-square";
            canvas.setAttribute("aria-label", label);
            panel.append(caption, canvas);
            panels.append(panel);
            return {panel, canvas};
        };
        const nativeView = makePanel("Published grid (click a sample)", "The pooled grid and its photoreceptor grid in the published image layout. Click a pooled sample to select it.");
        const fieldView = makePanel("Visual field (click a sample)", "Photoreceptor and pooled samples by visual angle. Click a pooled sample to select it.");
        const zoomView = makePanel("Around the selected sample", "The selected pooled sample's neighbourhood, enlarged.");
        const legend = document.createElement("p");
        legend.className = "fovi-legend";
        legend.innerHTML = '<span class="fovi-key fovi-key-muted"></span>photoreceptor'
            + '<span class="fovi-key fovi-key-highlight"></span>pooled into the selected sample'
            + '<span class="fovi-key fovi-key-accent"></span>pooled sample'
            + `<span class="fovi-legend-note">Fixed: ${FOV}° field, <code>cmf_a_deg=${A}</code>, <code>foveated_res=${LOW}</code>.</span>`;
        root.append(controls, summary, panels, legend);

        // fovi's pooling neighbourhoods for the isotropic retina, exported with the camera's own helpers.
        let isotropicTables = null;
        if (TABLES_URL) {
            fetch(TABLES_URL)
                .then((response) => (response.ok ? response.json() : Promise.reject(new Error(response.statusText))))
                .then((data) => {
                    if (data.fov !== FOV || data.cmfA !== A || data.res !== LOW) {
                        throw new Error("the exported tables describe another retina");
                    }
                    isotropicTables = data;
                    if (state.style === "isotropic") {
                        sync();
                    }
                })
                .catch((error) => {
                    isotropicTables = {error: String(error.message || error)};
                    if (state.style === "isotropic") {
                        sync();
                    }
                });
        }

        // Grid retina: samples keyed by their upright (row, column) in the published image.
        const gridLayout = (res) => {
            const samples = warpedGrid(FOV, A, res).map((s) => ({...s, row: res - 1 - s.j, col: s.i}));
            return {samples, at: new Map(samples.map((s) => [`${s.row},${s.col}`, s]))};
        };
        const layouts = new Map();
        const layoutFor = () => {
            const key = `${state.style}/${state.mult}`;
            if (layouts.has(key)) {
                return layouts.get(key);
            }
            let layout;
            if (state.style === "isotropic") {
                const rings = matchedIsotropicRes(FOV, A, LOW * LOW);
                const highRings = Math.round(state.mult * rings);
                layout = {
                    rings,
                    highRings,
                    low: isotropicSamples(FOV, A, rings),
                    high: isotropicSamples(FOV, A, highRings),
                };
            } else {
                layout = {low: gridLayout(LOW), high: gridLayout(LOW * state.mult)};
            }
            layouts.set(key, layout);
            return layout;
        };

        // The selected pooled sample, the photoreceptors it pools, and a description, for the current style.
        const selection = (layout) => {
            if (state.style !== "isotropic") {
                const unit = layout.low.at.get(`${state.row},${state.col}`);
                const cells = poolingWindow(LOW, state.mult, state.side, state.row, state.col);
                const members = cells.map(([r, c]) => layout.high.at.get(`${r},${c}`)).filter((s) => s.valid);
                const k = state.side * state.side;
                const text = `<code>pool_k=${k}</code> (window side ${state.side}, stride ${state.mult}); the photoreceptor grid is `
                    + `${LOW * state.mult} × ${LOW * state.mult}`
                    + (k < state.mult * state.mult ? "; a window smaller than the stride skips photoreceptors." : ".");
                return {unit, members, valid: unit.valid && members.length > 0, text, label: `(${state.row}, ${state.col})`};
            }
            const unit = layout.low[state.unit];
            const table = isotropicTable(layout);
            if (!table) {
                return {unit, members: [], valid: false, pending: true, text: "", label: `${state.unit}`};
            }
            const members = table.slice(0, state.k).filter((index) => index >= 0).map((index) => layout.high[index]);
            const text = `<code>pool_k=${state.k}</code> nearest photoreceptors ${state.space === "geodesic" ? "along the cortex" : "in the visual field"}; `
                + `${layout.high.length.toLocaleString("en-US")} photoreceptors on ${layout.highRings} rings pool into ${layout.low.length} samples on ${layout.rings} rings`
                + ` (default <code>pool_k</code> ${Math.round(layout.high.length / layout.low.length)}).`;
            return {unit, members, valid: members.length > 0, text, label: `${state.unit}`};
        };

        // The selected sample's neighbours, nearest first, from fovi's table for the chosen space.
        const isotropicTable = (layout) => {
            const entry = isotropicTables && !isotropicTables.error && isotropicTables.mult[String(state.mult)];
            if (!entry) {
                return null;
            }
            if (entry.count !== layout.high.length || isotropicTables.count !== layout.low.length) {
                isotropicTables = {error: "the exported tables do not match this retina"};
                return null;
            }
            return entry[state.space].slice(state.unit * entry.kMax, (state.unit + 1) * entry.kMax);
        };

        const fieldTransform = (w, h, centre, radius) => {
            const scale = (Math.min(w, h) / 2 - 8) / radius;
            return (s) => [w / 2 + (s[0] - centre[0]) * scale, h / 2 - (s[1] - centre[1]) * scale];
        };

        const redraw = () => {
            const colors = palette(root);
            const layout = layoutFor();
            const isotropic = state.style === "isotropic";
            const lowSamples = isotropic ? layout.low : layout.low.samples;
            const highSamples = isotropic ? layout.high : layout.high.samples;
            const chosen = selection(layout);
            const members = new Set(chosen.members);
            const {unit} = chosen;
            nativeView.panel.hidden = isotropic;
            // Published-grid view: photoreceptor cells, pooled cell borders, and the window.
            if (!isotropic) {
                const highRes = LOW * state.mult;
                const {ctx, w, h} = fitCanvas(nativeView.canvas);
                const size = Math.min(w, h) - 8;
                const ox = (w - size) / 2;
                const oy = (h - size) / 2;
                const cell = size / highRes;
                const windowCells = new Set(poolingWindow(LOW, state.mult, state.side, state.row, state.col).map(([r, c]) => `${r},${c}`));
                for (const s of highSamples) {
                    const member = windowCells.has(`${s.row},${s.col}`);
                    ctx.fillStyle = member ? colors.highlight : s.valid ? colors.faint : colors.surface;
                    ctx.globalAlpha = member && !s.valid ? 0.35 : 1;
                    ctx.fillRect(ox + s.col * cell + 0.5, oy + s.row * cell + 0.5, Math.max(cell - 1, 0.5), Math.max(cell - 1, 0.5));
                }
                ctx.globalAlpha = 1;
                for (let k = 0; k <= LOW; k++) {
                    polyline(ctx, [[ox + (k * size) / LOW, oy], [ox + (k * size) / LOW, oy + size]], colors.muted, 0.6);
                    polyline(ctx, [[ox, oy + (k * size) / LOW], [ox + size, oy + (k * size) / LOW]], colors.muted, 0.6);
                }
                ctx.strokeStyle = colors.accent;
                ctx.lineWidth = 2.4;
                ctx.strokeRect(ox + (state.col * size) / LOW, oy + (state.row * size) / LOW, size / LOW, size / LOW);
                nativeView.canvas.onclick = (event) => {
                    const box = nativeView.canvas.getBoundingClientRect();
                    const col = Math.floor(((event.clientX - box.left - ox) / size) * LOW);
                    const row = Math.floor(((event.clientY - box.top - oy) / size) * LOW);
                    if (row >= 0 && row < LOW && col >= 0 && col < LOW) {
                        choose({row, col});
                    }
                };
            }
            const zoomRegion = () => {
                const points = chosen.members.map((s) => s.s);
                points.push(unit.s);
                const xs = points.map((p) => p[0]);
                const ys = points.map((p) => p[1]);
                const centre = [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2];
                const extent = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 0.002);
                return [centre, extent * 1.1];
            };
            // Visual-field views: the whole retina, and the selected sample's neighbourhood.
            const drawField = (canvas, centre, radius, showZoomBox) => {
                const {ctx, w, h} = fitCanvas(canvas);
                const map = fieldTransform(w, h, centre, radius);
                const scale = (Math.min(w, h) / 2 - 8) / radius;
                const dotHigh = Math.max(0.7, Math.min(3, scale * 0.004 * Math.sqrt(1024 / highSamples.length) * 6));
                const highPoints = highSamples.filter((s) => s.valid);
                dots(ctx, highPoints.filter((s) => !members.has(s)).map((s) => map(s.s)), colors.muted, dotHigh, 0.55);
                dots(ctx, highPoints.filter((s) => members.has(s)).map((s) => map(s.s)), colors.highlight, dotHigh * 1.4, 1);
                dots(ctx, lowSamples.filter((s) => s.valid).map((s) => map(s.s)), colors.accent, dotHigh * 1.2, 0.9);
                const u = map(unit.s);
                ctx.beginPath();
                ctx.arc(u[0], u[1], dotHigh * 1.2 + 4, 0, TAU);
                ctx.strokeStyle = colors.accent;
                ctx.lineWidth = 2;
                ctx.stroke();
                polyline(ctx, Array.from({length: 181}, (_, k) => map([Math.cos((k * TAU) / 180), Math.sin((k * TAU) / 180)])), colors.camera, 1.2);
                if (showZoomBox) {
                    const [zc, zr] = zoomRegion();
                    const corner = map([zc[0] - zr, zc[1] + zr]);
                    ctx.strokeStyle = colors.ink;
                    ctx.setLineDash([3, 3]);
                    ctx.lineWidth = 1;
                    ctx.strokeRect(corner[0], corner[1], 2 * zr * scale, 2 * zr * scale);
                    ctx.setLineDash([]);
                }
                return map;
            };
            const map = drawField(fieldView.canvas, [0, 0], 1, true);
            fieldView.canvas.onclick = (event) => {
                const box = fieldView.canvas.getBoundingClientRect();
                const x = event.clientX - box.left;
                const y = event.clientY - box.top;
                let best = null;
                lowSamples.forEach((s, index) => {
                    if (!s.valid) {
                        return;
                    }
                    const p = map(s.s);
                    const d = Math.hypot(p[0] - x, p[1] - y);
                    if (!best || d < best.d) {
                        best = {d, s, index};
                    }
                });
                if (best) {
                    choose(isotropic ? {unit: best.index} : {row: best.s.row, col: best.s.col});
                }
            };
            const [zc, zr] = zoomRegion();
            drawField(zoomView.canvas, zc, zr, false);
            const ecc = Math.hypot(unit.s[0], unit.s[1]) * (FOV / 2);
            const span = chosen.members.length
                ? Math.max(...chosen.members.map((s) => Math.hypot(s.s[0] - unit.s[0], s.s[1] - unit.s[1]))) * (FOV / 2) * 2
                : 0;
            if (chosen.pending) {
                const reason = isotropicTables && isotropicTables.error ? `could not be loaded (${isotropicTables.error})` : "are loading";
                summary.innerHTML = `fovi's pooling neighbourhoods ${reason}.`;
                return;
            }
            summary.innerHTML = `Pooled sample ${chosen.label} at <strong>${ecc.toFixed(2)}°</strong> eccentricity`
                + (chosen.valid
                    ? ` averages <strong>${chosen.members.length}</strong> photoreceptors spanning about <strong>${span.toPrecision(2)}°</strong>. `
                    : " lies outside the retina, so the camera publishes zero there. ")
                + chosen.text;
        };

        const choose = (selected) => {
            Object.assign(state, selected);
            for (const name of Object.keys(selected)) {
                inputs[name].value = state[name];
            }
            sync();
        };

        const sync = () => {
            const isotropic = state.style === "isotropic";
            for (const element of gridControls) {
                element.hidden = isotropic;
            }
            for (const element of isotropicControls) {
                element.hidden = !isotropic;
            }
            if (isotropic) {
                const layout = layoutFor();
                inputs.unit.max = layout.low.length - 1;
                state.unit = Math.min(state.unit, layout.low.length - 1);
                inputs.unit.value = state.unit;
                inputs.k.max = 2 * state.mult * state.mult;
                state.k = Math.min(state.k, Number(inputs.k.max));
                inputs.k.value = state.k;
            }
            outputs.side.textContent = `${state.side} (pool_k=${state.side * state.side})`;
            outputs.row.textContent = state.row;
            outputs.col.textContent = state.col;
            outputs.k.textContent = state.k;
            outputs.unit.textContent = state.unit;
            redraw();
        };
        inputs.style.addEventListener("input", () => {
            state.style = inputs.style.value;
            sync();
        });
        inputs.space.addEventListener("input", () => {
            state.space = inputs.space.value;
            sync();
        });
        inputs.mult.addEventListener("input", () => {
            // Keep the default neighbourhood when it was in use: pool_res_mult squared photoreceptors.
            const defaultSide = state.side === state.mult;
            const defaultK = state.k === state.mult * state.mult;
            state.mult = Number(inputs.mult.value);
            if (defaultSide) {
                state.side = state.mult;
                inputs.side.value = state.side;
            }
            if (defaultK) {
                state.k = state.mult * state.mult;
            }
            sync();
        });
        for (const name of ["side", "row", "col", "k", "unit"]) {
            inputs[name].addEventListener("input", () => {
                state[name] = Number(inputs[name].value);
                sync();
            });
        }
        reset.addEventListener("click", () => {
            Object.assign(state, defaults);
            for (const [name, input] of Object.entries(inputs)) {
                input.value = state[name];
            }
            sync();
        });
        new ResizeObserver(() => redraw()).observe(panels);
        sync();
    };

    const initialize = () => {
        document.querySelectorAll("[data-fovi-sampling-viewer]").forEach(initSamplingViewer);
        document.querySelectorAll("[data-fovi-pooling-viewer]").forEach(initPoolingViewer);
    };

    // Exposed for the documentation's geometry checks.
    window.FoviViewerGeometry = {warpedGrid, isotropicRings, matchedIsotropicRes, isotropicSamples, ray, gazeRotation, poolingWindow};

    if (typeof document !== "undefined") {
        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", initialize);
        } else {
            initialize();
        }
    }
})();
