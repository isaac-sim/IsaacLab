// Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

export class JointPdPlot {
  constructor(viewport) {
    this.samples = [];
    this.container = document.createElement('div');
    this.container.className = 'browser-demo-tracking';
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'browser-demo-tracking-canvas';
    this.canvas.setAttribute('aria-label', 'Last four seconds of commanded and measured joint angles');
    this.caption = document.createElement('div');
    this.caption.className = 'browser-demo-tracking-caption';
    const target = document.createElement('span');
    target.className = 'browser-demo-tracking-target';
    target.textContent = 'Target ';
    this.targetValue = document.createElement('output');
    target.append(this.targetValue);
    const measured = document.createElement('span');
    measured.className = 'browser-demo-tracking-measured';
    measured.textContent = 'Measured ';
    this.measuredValue = document.createElement('output');
    measured.append(this.measuredValue);
    this.error = document.createElement('output');
    this.caption.append(target, measured, this.error);
    this.container.append(this.canvas, this.caption);
    viewport.after(this.container);
    this.draw();
  }

  reset() {
    this.samples.length = 0;
    this.draw();
  }

  add(time, target, measured) {
    this.samples.push({ time, target, measured });
    while (this.samples.length && this.samples[0].time < time - 4) this.samples.shift();
  }

  draw() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const pixelWidth = Math.round(width * ratio);
    const pixelHeight = Math.round(height * ratio);
    if (this.canvas.width !== pixelWidth || this.canvas.height !== pixelHeight) {
      this.canvas.width = pixelWidth;
      this.canvas.height = pixelHeight;
    }
    const context = this.canvas.getContext('2d');
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const left = 42;
    const right = width - 16;
    const top = 12;
    const bottom = height - 25;
    const seconds = 4;
    const end = Math.max(seconds, this.samples.at(-1)?.time || 0);
    const start = end - seconds;
    const x = (time) => left + (time - start) / seconds * (right - left);
    const visible = this.samples.filter((sample) => sample.time >= start);
    const extent = visible.reduce((maximum, sample) =>
      Math.max(maximum, Math.abs(sample.target), Math.abs(sample.measured)), 1.4);
    const range = Math.ceil(extent * 2) / 2;
    const y = (angle) => top + (range - angle) / (2 * range) * (bottom - top);

    context.font = '11px system-ui, sans-serif';
    context.textBaseline = 'middle';
    context.strokeStyle = '#e5edf0';
    context.lineWidth = 1;
    for (const angle of [-range, 0, range]) {
      context.beginPath();
      context.moveTo(left, y(angle));
      context.lineTo(right, y(angle));
      context.stroke();
      context.fillStyle = '#70858e';
      context.fillText(`${angle.toFixed(1)} rad`, 4, y(angle));
    }
    for (let second = 0; second <= seconds; second += 1) {
      const position = left + second / seconds * (right - left);
      context.beginPath();
      context.moveTo(position, top);
      context.lineTo(position, bottom);
      context.stroke();
    }
    context.fillStyle = '#70858e';
    context.fillText('4 s history', right - 64, height - 10);
    for (const [key, color] of [['target', '#d97825'], ['measured', '#236eb9']]) {
      context.strokeStyle = color;
      context.lineWidth = 2.5;
      context.lineJoin = 'round';
      context.beginPath();
      visible.forEach((sample, index) => {
        if (index) context.lineTo(x(sample.time), y(sample[key]));
        else context.moveTo(x(sample.time), y(sample[key]));
      });
      context.stroke();
    }
    const squaredError = visible.reduce((sum, sample) => sum + (sample.target - sample.measured) ** 2, 0);
    const latest = visible.at(-1);
    this.targetValue.value = latest ? `${latest.target.toFixed(2)} rad` : '—';
    this.measuredValue.value = latest ? `${latest.measured.toFixed(2)} rad` : '—';
    this.error.value = `RMS error ${Math.sqrt(squaredError / Math.max(visible.length, 1)).toFixed(2)} rad`;
  }

  dispose() {
    this.container.remove();
  }
}
