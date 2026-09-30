// Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

const arrayTypes = {
  float32: [Float32Array, 'HEAPF32'], float64: [Float64Array, 'HEAPF64'],
  int32: [Int32Array, 'HEAP32'], uint32: [Uint32Array, 'HEAPU32'],
};

class BrowserSimulation {
  static async load(source) {
    const url = new URL(source, document.baseURI);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Manifest request failed (${response.status})`);
    const manifest = await response.json();
    if (manifest.bundleVersion !== 1 || manifest.abiVersion !== 1) throw new Error('Unsupported simulation bundle');
    const factory = (await import(new URL(manifest.module, url).href)).default;
    const options = { locateFile: (name) => new URL(name.endsWith('.wasm') ? manifest.wasm : name, url).href };
    if (manifest.wasm.endsWith('.gz')) {
      if (!('DecompressionStream' in window)) throw new Error('This browser cannot decompress the simulation bundle');
      const wasmResponse = await fetch(new URL(manifest.wasm, url));
      if (!wasmResponse.ok || !wasmResponse.body) throw new Error(`WebAssembly request failed (${wasmResponse.status})`);
      const decompressed = wasmResponse.body.pipeThrough(new DecompressionStream('gzip'));
      options.wasmBinary = new Uint8Array(await new Response(decompressed).arrayBuffer());
    }
    const module = await factory(options);
    const simulation = new BrowserSimulation(manifest, module);
    if (!simulation.init()) {
      simulation.dispose();
      throw new Error('Could not initialize WebAssembly simulation');
    }
    simulation.url = url;
    return simulation;
  }

  constructor(manifest, module) {
    this.manifest = manifest;
    this.module = module;
    this.init = module.cwrap('newton_init', 'number', []);
    this.stepNative = module.cwrap('newton_step', 'number', []);
    this.resetNative = module.cwrap('newton_reset', 'number', []);
    this.disposeNative = module.cwrap('newton_dispose', null, []);
    this.pointer = module.cwrap('newton_get_param_ptr', 'number', ['string']);
    this.size = module.cwrap('newton_get_param_size', 'number', ['string']);
  }

  binding(name) {
    const metadata = this.manifest.bindings[name];
    if (!metadata) throw new Error(`Missing binding: ${name}`);
    const entry = arrayTypes[metadata.dtype];
    if (!entry) throw new Error(`Unsupported binding type: ${metadata.dtype}`);
    const [ArrayType, heap] = entry;
    const pointer = Number(this.pointer(name));
    if (!pointer || Number(this.size(name)) !== metadata.byteLength) throw new Error(`Invalid binding: ${name}`);
    return new ArrayType(this.module[heap].buffer, pointer, metadata.byteLength / ArrayType.BYTES_PER_ELEMENT);
  }

  step() {
    if (!this.stepNative()) throw new Error('Simulation step failed');
  }

  reset() {
    if (!this.resetNative()) throw new Error('Simulation reset failed');
  }

  dispose() {
    this.disposeNative();
  }
}

function rotateInverse(q, v) {
  const [x, y, z, w] = q;
  const tx = 2 * (-y * v[2] + z * v[1]);
  const ty = 2 * (-z * v[0] + x * v[2]);
  const tz = 2 * (-x * v[1] + y * v[0]);
  return [v[0] + w * tx - y * tz + z * ty,
    v[1] + w * ty - z * tx + x * tz,
    v[2] + w * tz - x * ty + y * tx];
}

class DensePolicy {
  static async load(url, description) {
    const response = await fetch(new URL(description.file, url));
    if (!response.ok) throw new Error(`Policy request failed (${response.status})`);
    const weights = new Float32Array(await response.arrayBuffer());
    const final = description.layers.at(-1);
    if (weights.length !== final.offset + final.rows * (final.columns + 1)) {
      throw new Error('Policy weight size does not match its manifest');
    }
    return new DensePolicy(weights, description.layers);
  }

  constructor(weights, layers) {
    this.weights = weights;
    this.layers = layers;
  }

  run(observation) {
    let input = observation;
    for (let index = 0; index < this.layers.length; index += 1) {
      const { rows, columns, offset } = this.layers[index];
      if (input.length !== columns) throw new Error('Policy observation shape mismatch');
      const output = new Float32Array(rows);
      const biasOffset = offset + rows * columns;
      for (let row = 0; row < rows; row += 1) {
        let sum = this.weights[biasOffset + row];
        for (let column = 0; column < columns; column += 1) {
          sum += this.weights[offset + row * columns + column] * input[column];
        }
        output[row] = index === this.layers.length - 1 ? sum : (sum > 0 ? sum : Math.expm1(sum));
      }
      input = output;
    }
    return input;
  }
}

class IsaacLabBrowserDemo extends HTMLElement {
  connectedCallback() {
    if (this.observer) return;
    this.generation = (this.generation || 0) + 1;
    const generation = this.generation;
    this.visible = false;
    this.observer = new IntersectionObserver(([entry]) => {
      this.visible = entry.isIntersecting;
      if (this.visible && !this.loading) this.loading = this.load(generation);
      else if (this.visible && this.simulation && !this.failed && !this.frameId) {
        this.previousTime = performance.now();
        this.frameId = requestAnimationFrame((time) => this.frame(time));
      } else if (!this.visible && this.frameId) {
        cancelAnimationFrame(this.frameId);
        this.frameId = 0;
      }
    });
    this.observer.observe(this);
  }

  disconnectedCallback() {
    this.generation += 1;
    this.observer?.disconnect();
    this.observer = null;
    cancelAnimationFrame(this.frameId);
    this.frameId = 0;
    this.plot?.dispose();
    this.plot = null;
    this.viewer?.dispose();
    this.viewer = null;
    this.simulation?.dispose();
    this.simulation = null;
    this.loading = null;
  }

  async load(generation) {
    this.failed = false;
    this.innerHTML = '<section class="browser-demo"><div class="browser-demo-head"><strong>Newton simulation</strong><span role="status">Loading…</span></div><canvas aria-label="Interactive Isaac Lab simulation"></canvas><div class="browser-demo-controls"></div></section>';
    this.status = this.querySelector('[role="status"]');
    this.canvas = this.querySelector('canvas');
    try {
      const simulation = await BrowserSimulation.load(this.getAttribute('src'));
      if (this.generation !== generation) {
        simulation.dispose();
        return;
      }
      this.simulation = simulation;
      this.demo = this.simulation.manifest.isaacLabDemo;
      if (!this.demo || !['stiffness', 'cloth_bending', 'rigid_friction', 'joint_pd', 'cartpole', 'g1', 'anymal'].includes(this.demo.kind)) throw new Error('Unknown Isaac Lab demo');
      this.querySelector('.browser-demo').classList.add(`browser-demo-${this.demo.kind}`);
      const title = this.getAttribute('demo-title') || this.demo.title;
      this.querySelector('.browser-demo-head strong').textContent = title;
      this.canvas.setAttribute('aria-label', `${title} interactive simulation`);
      if (this.demo.kind === 'rigid_friction' || this.demo.kind === 'joint_pd') {
        const { RigidViewer } = await import('./rigid-viewer.js');
        const viewer = await RigidViewer.load(this.canvas, this.simulation);
        if (this.generation !== generation) {
          viewer.dispose();
          return;
        }
        this.viewer = viewer;
        if (this.demo.kind === 'joint_pd') {
          const { JointPdPlot } = await import('./joint-pd-plot.js');
          if (this.generation !== generation) return;
          this.plot = new JointPdPlot(this.canvas);
        }
      } else if (this.demo.kind === 'cartpole') {
        const [{ CartpoleViewer }, policy] = await Promise.all([
          import('./cartpole-viewer.js'), DensePolicy.load(this.simulation.url, this.demo.policy),
        ]);
        if (this.generation !== generation) return;
        this.policy = policy;
        const viewer = await CartpoleViewer.load(this.canvas, this.simulation);
        if (this.generation !== generation) {
          viewer.dispose();
          return;
        }
        this.viewer = viewer;
      } else if (!['stiffness', 'cloth_bending'].includes(this.demo.kind)) {
        const [{ LocomotionViewer }, policy] = await Promise.all([
          import('./locomotion-viewer.js'), DensePolicy.load(this.simulation.url, this.demo.policy),
        ]);
        if (this.generation !== generation) return;
        this.policy = policy;
        const viewer = await LocomotionViewer.load(this.canvas, this.simulation);
        if (this.generation !== generation) {
          viewer.dispose();
          return;
        }
        this.viewer = viewer;
        this.command = new Float32Array(3);
        this.previousAction = new Float32Array(this.demo.actionIndices?.length || this.demo.jointNames.length);
      } else {
        const { StiffnessViewer } = await import('./stiffness-viewer.js');
        const viewer = await StiffnessViewer.load(this.canvas, this.simulation);
        if (this.generation !== generation) {
          viewer.dispose();
          return;
        }
        this.viewer = viewer;
      }
      this.tick = 0;
      this.elapsed = 0;
      this.setupControls();
      this.running = true;
      this.previousTime = performance.now();
      this.status.textContent = 'Running in your browser';
      this.render();
      if (this.visible) this.frameId = requestAnimationFrame((time) => this.frame(time));
    } catch (error) {
      if (this.generation !== generation) return;
      this.failed = true;
      this.plot?.dispose();
      this.plot = null;
      this.viewer?.dispose();
      this.viewer = null;
      this.simulation?.dispose();
      this.simulation = null;
      this.status.textContent = `Could not start: ${error.message}`;
      this.status.classList.add('browser-demo-error');
      const retry = document.createElement('button');
      retry.textContent = 'Retry loading';
      retry.addEventListener('click', () => { this.loading = this.load(this.generation); });
      this.querySelector('.browser-demo-controls').replaceChildren(retry);
      console.error(error);
    }
  }

  setupControls() {
    const panel = this.querySelector('.browser-demo-controls');
    if (this.demo.kind === 'stiffness' || this.demo.kind === 'cloth_bending' || this.demo.kind === 'rigid_friction') {
      const legend = document.createElement('div');
      legend.className = 'browser-demo-legend';
      const references = this.demo.kind === 'cloth_bending'
        ? [['Reference · 0.001 N·m', '#297ad9'], ['Adjustable · 1.00 N·m', '#9154d9'], ['Reference · 10 N·m', '#e37a38']]
        : this.demo.kind === 'rigid_friction'
          ? [['Reference · μ 0.05', '#236bdb'], ['Adjustable · μ 0.15', '#8648ce'], ['Reference · μ 0.8', '#e34c31']]
          : [['Reference · μ = λ = 2 kPa', '#76b900'], ['Adjustable · μ = λ = 20 kPa', '#55880a'], ['Reference · μ = λ = 100 kPa', '#88cc22']];
      for (const [label, color] of references) {
        const item = document.createElement('span');
        const dot = document.createElement('i');
        dot.style.background = color;
        const value = document.createElement('span');
        value.textContent = label;
        item.append(dot, value);
        legend.append(item);
        if (label.startsWith('Adjustable')) this.middleLegend = value;
      }
      panel.append(legend);
      const selected = (this.getAttribute('parameters') || (this.demo.kind === 'stiffness'
        ? 'shear,volume,damping,gravity' : this.demo.kind === 'cloth_bending' ? 'bending,gravity' : 'middle_friction')).split(',');
      for (const name of selected) {
        const parameter = this.simulation.manifest.parameters.find((item) => item.binding === name);
        if (!parameter) throw new Error(`Unknown simulation parameter: ${name}`);
        const logarithmic = name === 'bending' && this.demo.bendingScale === 'log10';
        const physical = (value) => logarithmic ? 10 ** value : value;
        const format = ['shear', 'volume'].includes(name) ? (value) => `${(value / 1000).toFixed(0)} kPa`
          : name === 'bending' ? (value) => `${value < 0.01 ? value.toFixed(3) : value.toFixed(2)} N·m`
            : name === 'middle_friction' ? (value) => `μ ${value.toFixed(2)}`
          : name === 'gravity' ? (value) => `${value.toFixed(1)} m/s²`
            : (value) => value.toFixed(1);
        this.addSlider(panel, parameter.label,
          logarithmic ? Math.log10(parameter.minimum) : parameter.minimum,
          logarithmic ? Math.log10(parameter.maximum) : parameter.maximum,
          logarithmic ? 0.01 : parameter.step,
          logarithmic ? Math.log10(this.simulation.binding(parameter.binding)[parameter.index])
            : this.simulation.binding(parameter.binding)[parameter.index], (input) => {
            const value = physical(input);
            this.simulation.binding(parameter.binding)[parameter.index] = value;
            if (['shear', 'volume', 'bending', 'middle_friction'].includes(name)) {
              this.middleLegend.textContent = this.demo.kind === 'stiffness'
                ? `Adjustable · μ ${format(this.simulation.binding('shear')[0])} · λ ${format(this.simulation.binding('volume')[0])}`
                : `Adjustable · ${format(value)}`;
            }
          }, (value) => format(physical(value)));
      }
    } else if (this.demo.kind === 'joint_pd') {
      this.setupJointPdControls(panel);
    } else if (this.demo.kind === 'cartpole') {
      this.policyForce = 0;
      this.perturbation = 0;
      const forceCard = document.createElement('div');
      forceCard.className = 'browser-demo-force';
      const slider = this.addSlider(forceCard, 'Push cart', -this.demo.perturbationLimit,
        this.demo.perturbationLimit, 5, 0, (value) => {
          this.perturbation = -value;
          this.updateCartpoleForce();
        }, (value) => `${value < 0 ? '+' : value > 0 ? '−' : ''}${Math.abs(value).toFixed(0)} N`);
      slider.setAttribute('aria-label', 'Cart perturbation force; release to stop pushing');
      const scale = document.createElement('div');
      scale.className = 'browser-demo-force-scale';
      for (const label of [`+${this.demo.perturbationLimit} N`, '0', `−${this.demo.perturbationLimit} N`]) {
        const mark = document.createElement('span');
        mark.textContent = label;
        scale.append(mark);
      }
      forceCard.append(scale);
      const release = () => {
        slider.value = 0;
        slider.dispatchEvent(new Event('input'));
      };
      for (const name of ['pointerup', 'pointercancel', 'keyup', 'blur']) slider.addEventListener(name, release);
      this.resetSticks = release;
      const hint = document.createElement('small');
      hint.textContent = 'Hold to perturb the cart. Release to see the policy respond.';
      forceCard.append(hint);
      panel.append(forceCard);
    } else {
      this.addVelocityControls(panel);
    }
    const buttons = document.createElement('div');
    buttons.className = 'browser-demo-buttons';
    const pause = document.createElement('button');
    pause.textContent = 'Pause';
    pause.addEventListener('click', () => {
      this.running = !this.running;
      pause.textContent = this.running ? 'Pause' : 'Play';
    });
    const reset = document.createElement('button');
    reset.textContent = 'Reset';
    reset.addEventListener('click', () => this.resetSimulation());
    buttons.append(pause, reset);
    panel.append(buttons);
  }

  setupJointPdControls(panel) {
    this.selectedJoint = 0;
    this.driveMode = 'step';
    this.commandAmplitude = this.simulation.binding('target_q')[0];
    this.frequency = 0.7;
    const toolbar = document.createElement('div');
    toolbar.className = 'browser-demo-pd-toolbar';
    const addSelect = (label, options, onChange) => {
      const row = document.createElement('label');
      const select = document.createElement('select');
      for (const [value, title] of options) {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = title;
        select.append(option);
      }
      select.addEventListener('change', () => onChange(select.value));
      row.append(label, select);
      toolbar.append(row);
    };
    addSelect('Tune joint', this.demo.jointNames.map((name, index) => [String(index), name]), (value) => {
      this.selectedJoint = Number(value);
      for (const name of ['stiffness', 'damping']) {
        const slider = this.gainSliders[name];
        const gain = this.simulation.binding(name)[this.selectedJoint];
        slider.value = gain;
        slider.parentElement.querySelector('output').value = gain.toFixed(name === 'damping' ? 1 : 0);
      }
      this.resetSimulation();
    });
    addSelect('Command', [['step', 'Step'], ['sine', 'Sine wave']], (value) => {
      this.driveMode = value;
      this.frequencyRow.hidden = value !== 'sine';
      this.resetSimulation();
    });
    panel.append(toolbar);
    const command = this.simulation.manifest.parameters.find((item) => item.binding === 'target_q');
    this.addSlider(panel, 'Target / amplitude [rad]', command.minimum, command.maximum, command.step, this.commandAmplitude,
      (value) => { this.commandAmplitude = value; this.updateJointPdTarget(); },
      (value) => value.toFixed(2));
    const frequency = this.addSlider(panel, 'Frequency [Hz]', 0.2, 2.0, 0.1, this.frequency,
      (value) => { this.frequency = value; }, (value) => value.toFixed(1));
    this.frequencyRow = frequency.parentElement;
    this.frequencyRow.hidden = true;
    this.gainSliders = {};
    for (const name of ['stiffness', 'damping']) {
      const parameter = this.simulation.manifest.parameters.find((item) => item.binding === name);
      this.gainSliders[name] = this.addSlider(panel, parameter.label, parameter.minimum, parameter.maximum,
        parameter.step, this.simulation.binding(name)[0],
        (value) => { this.simulation.binding(name)[this.selectedJoint] = value; },
        (value) => value.toFixed(name === 'damping' ? 1 : 0));
    }
    this.updateJointPdTarget();
  }

  updateJointPdTarget() {
    const target = this.simulation.binding('target_q');
    target.fill(0);
    target[this.selectedJoint] = this.driveMode === 'sine'
      ? this.commandAmplitude * Math.sin(2 * Math.PI * this.frequency * this.tick * this.simulation.manifest.timestep)
      : this.commandAmplitude;
  }

  resetSimulation() {
    this.simulation.reset();
    this.previousAction?.fill(0);
    if (this.demo.kind === 'cartpole') this.policyForce = 0;
    this.resetSticks?.();
    this.tick = 0;
    this.elapsed = 0;
    if (this.demo.kind === 'joint_pd') {
      this.plot.reset();
      this.updateJointPdTarget();
    }
    this.render();
  }

  addSlider(panel, label, min, max, step, initial, onInput, formatValue = (value) => Number(value).toFixed(step < 1 ? 2 : 0)) {
    const row = document.createElement('label');
    row.className = 'browser-demo-slider';
    const text = document.createElement('span');
    const value = document.createElement('output');
    text.textContent = label;
    value.value = formatValue(initial);
    const slider = document.createElement('input');
    slider.type = 'range';
    Object.assign(slider, { min, max, step, value: initial });
    slider.addEventListener('input', () => {
      const number = Number(slider.value);
      value.value = formatValue(number);
      onInput(number);
    });
    row.append(text, slider, value);
    panel.append(row);
    return slider;
  }

  addVelocityControls(panel) {
    const controls = document.createElement('div');
    controls.className = 'browser-demo-joysticks';
    const linear = this.addStick(controls, 'Travel', 'Forward / back · left / right', true,
      (x, y, readout) => {
        this.command[0] = -y * (this.demo.policyType === 'wbc_agile_g1' ? (y < 0 ? 0.8 : 0.6) : 1);
        this.command[1] = -x * (this.demo.policyType === 'wbc_agile_g1' ? 0.5 : 1);
        readout.value = `x ${this.command[0].toFixed(2)} · y ${this.command[1].toFixed(2)} m/s`;
      });
    const yaw = this.addStick(controls, 'Turn', 'Yaw left / right', false,
      (x, _y, readout) => {
        this.command[2] = -x;
        readout.value = `${this.command[2].toFixed(2)} rad/s`;
      });
    this.resetSticks = () => { linear(); yaw(); };
    panel.append(controls);
  }

  addStick(panel, label, hint, linear, apply) {
    const section = document.createElement('section');
    section.className = 'browser-demo-stick-section';
    const heading = document.createElement('div');
    heading.className = 'browser-demo-stick-heading';
    const readout = document.createElement('output');
    heading.append(document.createTextNode(label), readout);
    const pad = document.createElement('div');
    pad.className = `browser-demo-stick ${linear ? 'browser-demo-stick-linear' : 'browser-demo-stick-yaw'}`;
    pad.tabIndex = 0;
    pad.setAttribute('role', 'group');
    pad.setAttribute('aria-label', `${label}: drag or use arrow keys; release to stop`);
    for (const [side, glyph] of linear
      ? [['top', '↑'], ['right', '→'], ['bottom', '↓'], ['left', '←']]
      : [['left', '↶'], ['right', '↷']]) {
      const mark = document.createElement('span');
      mark.className = `browser-demo-stick-mark browser-demo-stick-mark-${side}`;
      mark.textContent = glyph;
      pad.append(mark);
    }
    const knob = document.createElement('span');
    knob.className = 'browser-demo-stick-knob';
    pad.append(knob);
    const description = document.createElement('small');
    description.textContent = hint;
    section.append(heading, pad, description);
    panel.append(section);
    let activePointer = null;
    const keys = new Set();
    const move = (x, y) => {
      x = Math.max(-1, Math.min(1, x));
      y = linear ? Math.max(-1, Math.min(1, y)) : 0;
      knob.style.transform = `translate(${x * pad.clientWidth * 0.32}px, ${y * pad.clientWidth * 0.32}px)`;
      apply(x, y, readout);
    };
    const reset = () => {
      activePointer = null;
      keys.clear();
      pad.classList.remove('active');
      move(0, 0);
    };
    const pointerMove = (event) => {
      const box = pad.getBoundingClientRect();
      const travel = box.width * 0.32;
      move((event.clientX - box.left - box.width / 2) / travel,
        (event.clientY - box.top - box.height / 2) / travel);
    };
    pad.addEventListener('pointerdown', (event) => {
      if (activePointer !== null) return;
      event.preventDefault();
      activePointer = event.pointerId;
      pad.setPointerCapture(event.pointerId);
      pad.classList.add('active');
      pointerMove(event);
    });
    pad.addEventListener('pointermove', (event) => { if (event.pointerId === activePointer) pointerMove(event); });
    for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
      pad.addEventListener(name, (event) => { if (event.pointerId === activePointer) reset(); });
    }
    const applyKeys = () => move(
      Number(keys.has('ArrowRight')) - Number(keys.has('ArrowLeft')),
      Number(keys.has('ArrowDown')) - Number(keys.has('ArrowUp')),
    );
    pad.addEventListener('keydown', (event) => {
      if (!event.key.startsWith('Arrow')) return;
      event.preventDefault();
      keys.add(event.key);
      pad.classList.add('active');
      applyKeys();
    });
    pad.addEventListener('keyup', (event) => {
      if (!event.key.startsWith('Arrow')) return;
      event.preventDefault();
      keys.delete(event.key);
      keys.size ? applyKeys() : reset();
    });
    pad.addEventListener('blur', reset);
    reset();
    return reset;
  }

  applyPolicy() {
    if (this.demo.policyType === 'wbc_agile_g1') {
      this.applyAgileG1Policy();
      return;
    }
    const { jointNames, jointDefaults, rootBody, actionScale } = this.demo;
    const count = jointNames.length;
    const pose = this.simulation.binding('body_q');
    const velocity = this.simulation.binding('body_qd');
    const joints = this.simulation.binding('joint_q');
    const jointVelocity = this.simulation.binding('joint_qd');
    const base = rootBody * 7;
    const speed = rootBody * 6;
    const quaternion = pose.subarray(base + 3, base + 7);
    const observation = new Float32Array(12 + 3 * count);
    observation.set(rotateInverse(quaternion, velocity.subarray(speed, speed + 3)), 0);
    observation.set(rotateInverse(quaternion, velocity.subarray(speed + 3, speed + 6)), 3);
    observation.set(rotateInverse(quaternion, [0, 0, -1]), 6);
    observation.set(this.command, 9);
    for (let index = 0; index < count; index += 1) {
      observation[12 + index] = joints[7 + index] - jointDefaults[index];
      observation[12 + count + index] = jointVelocity[6 + index];
    }
    observation.set(this.previousAction, 12 + 2 * count);
    const action = this.policy.run(observation);
    const target = this.simulation.binding('target_q');
    for (let index = 0; index < count; index += 1) {
      if (!Number.isFinite(action[index])) throw new Error('Policy produced a non-finite action');
      target[7 + index] = jointDefaults[index] + actionScale * action[index];
    }
    this.previousAction.set(action);
  }

  applyAgileG1Policy() {
    const { jointDefaults, observationIndices, actionIndices, rootBody, actionScale } = this.demo;
    const pose = this.simulation.binding('body_q');
    const velocity = this.simulation.binding('body_qd');
    const joints = this.simulation.binding('joint_q');
    const jointVelocity = this.simulation.binding('joint_qd');
    const quaternion = pose.subarray(rootBody * 7 + 3, rootBody * 7 + 7);
    const speed = rootBody * 6;
    const observation = new Float32Array(83);
    observation.set(this.command, 0);
    observation.set(rotateInverse(quaternion, velocity.subarray(speed, speed + 3)), 4);
    observation.set(rotateInverse(quaternion, velocity.subarray(speed + 3, speed + 6)), 7);
    observation.set(rotateInverse(quaternion, [0, 0, -1]), 10);
    for (let index = 0; index < observationIndices.length; index += 1) {
      const joint = observationIndices[index];
      observation[13 + index] = joints[7 + joint] - jointDefaults[joint];
      observation[42 + index] = 0.1 * jointVelocity[6 + joint];
    }
    for (let index = 0; index < this.previousAction.length; index += 1) {
      observation[71 + index] = Math.max(-10, Math.min(10, this.previousAction[index]));
    }
    const action = this.policy.run(observation);
    const target = this.simulation.binding('target_q');
    for (let index = 0; index < actionIndices.length; index += 1) {
      if (!Number.isFinite(action[index])) throw new Error('Policy produced a non-finite action');
      const joint = actionIndices[index];
      target[7 + joint] = Math.max(-4, Math.min(4, jointDefaults[joint] + actionScale * action[index]));
    }
    this.previousAction.set(action);
  }

  applyCartpolePolicy() {
    const position = this.simulation.binding('joint_q');
    const velocity = this.simulation.binding('joint_qd');
    const action = this.policy.run(new Float32Array([position[0], position[1], velocity[0], velocity[1]]))[0];
    if (!Number.isFinite(action)) throw new Error('Cartpole policy produced a non-finite action');
    this.policyForce = this.demo.actionScale * action;
    this.updateCartpoleForce();
  }

  updateCartpoleForce() {
    this.simulation.binding('force')[0] = this.policyForce + this.perturbation;
  }

  frame(time) {
    if (!this.isConnected || !this.visible) {
      this.frameId = 0;
      return;
    }
    try {
      if (this.running && this.visible) {
        this.elapsed = Math.min(this.elapsed + Math.min((time - this.previousTime) / 1000, 0.05), 0.06);
        const steps = Math.min(12, Math.floor(this.elapsed / this.simulation.manifest.timestep));
        let resetDuringFrame = false;
        for (let index = 0; index < steps; index += 1) {
          if (this.policy && this.tick % this.demo.decimation === 0) {
            this.demo.kind === 'cartpole' ? this.applyCartpolePolicy() : this.applyPolicy();
          }
          if (this.demo.kind === 'joint_pd') this.updateJointPdTarget();
          this.simulation.step();
          this.tick += 1;
          if (this.demo.kind === 'joint_pd') {
            this.plot.add(this.tick * this.simulation.manifest.timestep,
              this.simulation.binding('target_q')[this.selectedJoint],
              this.simulation.binding('joint_q')[this.selectedJoint]);
          }
          if (this.demo.cycleSteps && this.tick >= this.demo.cycleSteps) {
            this.resetSimulation();
            resetDuringFrame = true;
            break;
          }
          if (this.demo.kind === 'cartpole'
              && Math.abs(this.simulation.binding('joint_q')[0]) > this.demo.maxCartPosition) {
            this.resetSimulation();
            resetDuringFrame = true;
            break;
          }
        }
        if (!resetDuringFrame) this.elapsed -= steps * this.simulation.manifest.timestep;
      }
      this.render();
      this.previousTime = time;
      this.frameId = requestAnimationFrame((next) => this.frame(next));
    } catch (error) {
      this.failed = true;
      this.frameId = 0;
      this.status.textContent = `Simulation stopped: ${error.message}`;
      this.status.classList.add('browser-demo-error');
      console.error(error);
    }
  }

  render() {
    this.viewer.render();
    this.plot?.draw();
  }

}

customElements.define('isaaclab-browser-demo', IsaacLabBrowserDemo);
