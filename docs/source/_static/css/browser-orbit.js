// Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

export class OrbitCamera {
  constructor(canvas, camera, azimuth, elevation, distance) {
    this.canvas = canvas;
    this.camera = camera;
    this.azimuth = azimuth;
    this.elevation = elevation;
    this.distance = distance;
    this.onPointerDown = (event) => {
      this.drag = { x: event.clientX, y: event.clientY, id: event.pointerId };
      canvas.setPointerCapture(event.pointerId);
    };
    this.onPointerMove = (event) => {
      if (!this.drag || event.pointerId !== this.drag.id) return;
      this.azimuth -= (event.clientX - this.drag.x) * 0.006;
      this.elevation = Math.max(0.05, Math.min(1.25, this.elevation + (event.clientY - this.drag.y) * 0.006));
      this.drag.x = event.clientX;
      this.drag.y = event.clientY;
    };
    this.onPointerUp = () => { this.drag = null; };
    this.onWheel = (event) => {
      event.preventDefault();
      this.distance = Math.max(1.25, Math.min(6, this.distance * Math.exp(event.deltaY * 0.001)));
    };
    canvas.addEventListener('pointerdown', this.onPointerDown);
    canvas.addEventListener('pointermove', this.onPointerMove);
    canvas.addEventListener('pointerup', this.onPointerUp);
    canvas.addEventListener('pointercancel', this.onPointerUp);
    canvas.addEventListener('wheel', this.onWheel, { passive: false });
  }

  update(target) {
    const horizontal = this.distance * Math.cos(this.elevation);
    this.camera.position.set(
      target.x + horizontal * Math.cos(this.azimuth),
      target.y + this.distance * Math.sin(this.elevation),
      target.z + horizontal * Math.sin(this.azimuth),
    );
    this.camera.lookAt(target);
  }

  dispose() {
    this.canvas.removeEventListener('pointerdown', this.onPointerDown);
    this.canvas.removeEventListener('pointermove', this.onPointerMove);
    this.canvas.removeEventListener('pointerup', this.onPointerUp);
    this.canvas.removeEventListener('pointercancel', this.onPointerUp);
    this.canvas.removeEventListener('wheel', this.onWheel);
  }
}
