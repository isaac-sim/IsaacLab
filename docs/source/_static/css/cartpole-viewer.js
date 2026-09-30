// Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

import * as THREE from '../vendor/three.module.min.js';
import { addIsaacLabGround } from './browser-ground.js';
import { OrbitCamera } from './browser-orbit.js';

export class CartpoleViewer {
  static async load(canvas, simulation) {
    const viewer = new CartpoleViewer(canvas, simulation);
    try {
      viewer.disposeGround = await addIsaacLabGround(viewer.scene);
      return viewer;
    } catch (error) {
      viewer.dispose();
      throw error;
    }
  }

  constructor(canvas, simulation) {
    this.canvas = canvas;
    this.simulation = simulation;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'low-power' });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.5;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xdde8ed);
    this.scene.fog = new THREE.Fog(0xdde8ed, 9, 30);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x91a0a3, 2.0));
    const light = new THREE.DirectionalLight(0xffffff, 3.5);
    light.position.set(4, 9, 6);
    light.castShadow = true;
    light.shadow.mapSize.set(1024, 1024);
    light.shadow.camera.left = -6;
    light.shadow.camera.right = 6;
    light.shadow.camera.top = 6;
    light.shadow.camera.bottom = -6;
    this.scene.add(light);

    this.stage = new THREE.Group();
    this.stage.rotation.x = -Math.PI / 2;
    this.scene.add(this.stage);
    this.bodies = [new THREE.Group(), new THREE.Group()];
    this.bodies.forEach((body) => this.stage.add(body));
    const colors = [0x455961, 0x182e38, 0x76b900];
    this.geometry = [];
    this.materials = colors.map((color, index) => new THREE.MeshStandardMaterial({
      color, metalness: index === 2 ? 0.28 : 0.52, roughness: index === 2 ? 0.32 : 0.44,
    }));
    simulation.manifest.isaacLabDemo.shapes.forEach((shape, index) => {
      const [x, y, z] = shape.halfExtents;
      const geometry = new THREE.BoxGeometry(2 * x, 2 * y, 2 * z);
      const mesh = new THREE.Mesh(geometry, this.materials[index]);
      mesh.position.fromArray(shape.position);
      mesh.quaternion.fromArray(shape.quaternion);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      (shape.body < 0 ? this.stage : this.bodies[shape.body]).add(mesh);
      this.geometry.push(geometry);
    });
    for (const railPosition of [-1.6, 1.6]) {
      for (const [size, position] of [
        [[0.06, 0.06, 2.0], [0, railPosition, 1.0]],
        [[0.26, 0.26, 0.04], [0, railPosition, 0.02]],
      ]) {
        const geometry = new THREE.BoxGeometry(...size);
        const support = new THREE.Mesh(geometry, this.materials[0]);
        support.position.set(...position);
        support.castShadow = true;
        this.stage.add(support);
        this.geometry.push(geometry);
      }
    }
    this.camera = new THREE.PerspectiveCamera(48, 1, 0.05, 70);
    this.orbit = new OrbitCamera(canvas, this.camera, 0.2, 0.3, 3.7);
    this.target = new THREE.Vector3(0, 1.75, 0);
  }

  render() {
    const poses = this.simulation.binding('body_q');
    this.bodies.forEach((body, index) => {
      const offset = index * 7;
      body.position.set(poses[offset], poses[offset + 1], poses[offset + 2]);
      body.quaternion.set(poses[offset + 3], poses[offset + 4], poses[offset + 5], poses[offset + 6]);
    });
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (width && height && (this.canvas.width !== width || this.canvas.height !== height)) {
      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }
    this.orbit.update(this.target);
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.orbit?.dispose();
    this.disposeGround?.();
    this.geometry?.forEach((geometry) => geometry.dispose());
    this.materials?.forEach((material) => material.dispose());
    this.renderer?.dispose();
  }
}
