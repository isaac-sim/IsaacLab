// Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

import * as THREE from '../vendor/three.module.min.js';
import { addIsaacLabGround } from './browser-ground.js';
import { OrbitCamera } from './browser-orbit.js';

export class RigidViewer {
  static async load(canvas, simulation) {
    const viewer = new RigidViewer(canvas, simulation);
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
    this.renderer.toneMappingExposure = 1.35;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#eef3f5');
    this.scene.fog = new THREE.Fog('#eef3f5', 9, 24);
    this.scene.add(new THREE.HemisphereLight('#ffffff', '#a2b1b9', 1.4));
    const key = new THREE.DirectionalLight('#fffdf7', 2.6);
    key.position.set(3, 7, 5);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = -4;
    key.shadow.camera.right = 4;
    key.shadow.camera.top = 4;
    key.shadow.camera.bottom = -4;
    this.scene.add(key);

    this.stage = new THREE.Group();
    this.stage.rotation.x = -Math.PI / 2;
    this.scene.add(this.stage);
    const visualization = simulation.manifest.visualization;
    this.bodies = Array.from({ length: visualization.bodyCount }, () => new THREE.Group());
    this.bodies.forEach((body) => this.stage.add(body));
    this.geometries = [];
    this.materials = [];
    if (visualization.ground) {
      const ground = visualization.ground;
      const thickness = 0.08;
      const geometry = new THREE.BoxGeometry(ground.size[0], ground.size[1], thickness);
      const material = new THREE.MeshStandardMaterial({ color: '#87959b', roughness: 0.67, metalness: 0.08 });
      const ramp = new THREE.Mesh(geometry, material);
      ramp.position.fromArray(ground.transform.slice(0, 3));
      ramp.quaternion.fromArray(ground.transform.slice(3));
      ramp.translateZ(-thickness / 2);
      ramp.castShadow = true;
      ramp.receiveShadow = true;
      this.stage.add(ramp);
      this.geometries.push(geometry);
      this.materials.push(material);
    }
    for (const shape of visualization.shapes) {
      if (shape.type !== 'box') throw new Error(`Unsupported rigid shape: ${shape.type}`);
      const geometry = new THREE.BoxGeometry(...shape.size);
      const material = new THREE.MeshStandardMaterial({ color: shape.color, roughness: 0.35, metalness: 0.12 });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.fromArray(shape.transform.slice(0, 3));
      mesh.quaternion.fromArray(shape.transform.slice(3));
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.bodies[shape.body].add(mesh);
      this.geometries.push(geometry);
      this.materials.push(material);
    }
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 35);
    this.target = new THREE.Vector3(0, 0.85, 0);
    this.orbit = new OrbitCamera(canvas, this.camera, 1.25, 0.35, 5.2);
  }

  render() {
    const poses = this.simulation.binding('body_q');
    this.bodies.forEach((body, index) => {
      const offset = index * 7;
      const position = poses.subarray(offset, offset + 3);
      if (!position.every(Number.isFinite)) throw new Error('Non-finite rigid body position');
      body.position.fromArray(position);
      body.quaternion.fromArray(poses.subarray(offset + 3, offset + 7));
    });
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    if (this.canvas.width !== Math.round(width * ratio) || this.canvas.height !== Math.round(height * ratio)) {
      this.renderer.setPixelRatio(ratio);
      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }
    this.orbit.update(this.target);
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.disposeGround?.();
    this.orbit?.dispose();
    this.geometries?.forEach((geometry) => geometry.dispose());
    this.materials?.forEach((material) => material.dispose());
    this.renderer?.dispose();
  }
}
