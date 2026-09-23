// Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

import * as THREE from '../vendor/three.module.min.js';
import { addIsaacLabGround } from './browser-ground.js';
import { OrbitCamera } from './browser-orbit.js';

export class LocomotionViewer {
  static async load(canvas, simulation) {
    const { visuals } = simulation.manifest.isaacLabDemo;
    const response = await fetch(new URL(visuals.file, simulation.url));
    if (!response.ok) throw new Error(`Robot visual request failed (${response.status})`);
    const data = await response.arrayBuffer();
    if (data.byteLength !== visuals.byteLength) throw new Error('Robot visual bundle size does not match its manifest');
    const viewer = new LocomotionViewer(canvas, simulation, data);
    try {
      viewer.disposeGround = await addIsaacLabGround(viewer.scene);
      return viewer;
    } catch (error) {
      viewer.dispose();
      throw error;
    }
  }

  constructor(canvas, simulation, data) {
    this.canvas = canvas;
    this.simulation = simulation;
    this.isAnymal = simulation.manifest.isaacLabDemo.kind === 'anymal';
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'low-power' });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.65;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#f3f6f7');
    this.scene.fog = new THREE.Fog('#f3f6f7', 7, 15);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 50);
    this.target = new THREE.Vector3();
    this.orbit = new OrbitCamera(canvas, this.camera, 0.55, 0.28, this.isAnymal ? 2.0 : 2.55);

    this.scene.add(new THREE.HemisphereLight('#ffffff', '#acb9be', 2.2));
    const key = new THREE.DirectionalLight('#ffffff', 3.0);
    key.position.set(3, 5, 4);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = -2.5;
    key.shadow.camera.right = 2.5;
    key.shadow.camera.top = 2.5;
    key.shadow.camera.bottom = -2.5;
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 12;
    key.shadow.bias = -0.0001;
    this.key = key;
    this.scene.add(key);
    this.scene.add(key.target);
    const fill = new THREE.DirectionalLight('#d7e6ed', 1.4);
    fill.position.set(-3, 2, -2);
    this.scene.add(fill);

    this.robot = new THREE.Group();
    this.robot.rotation.x = -Math.PI / 2;
    this.scene.add(this.robot);
    const materials = {
      shell: new THREE.MeshStandardMaterial({ color: this.isAnymal ? '#c5cbd1' : '#d8e0e5', roughness: 0.42, metalness: 0.22 }),
      structure: new THREE.MeshStandardMaterial({ color: '#364650', roughness: 0.5, metalness: 0.28 }),
      accent: new THREE.MeshStandardMaterial({ color: this.isAnymal ? '#386f91' : '#76b900', roughness: 0.46, metalness: 0.15 }),
    };
    this.materials = Object.values(materials);
    const pose = simulation.binding('body_q');
    this.bodies = Array.from({ length: pose.length / 7 }, () => {
      const group = new THREE.Group();
      this.robot.add(group);
      return group;
    });
    this.geometries = [];
    for (const visual of simulation.manifest.isaacLabDemo.visuals.meshes) {
      const vertices = new Float32Array(data, visual.vertexOffset, visual.vertexCount * 3);
      const indices = new Uint16Array(data, visual.indexOffset, visual.indexCount);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
      geometry.setIndex(new THREE.BufferAttribute(indices, 1));
      geometry.computeVertexNormals();
      const mesh = new THREE.Mesh(geometry, materials[visual.material]);
      mesh.castShadow = !/_(zero|one|two|three|four|five|six)_link$/.test(visual.name);
      mesh.receiveShadow = true;
      this.bodies[visual.body].add(mesh);
      this.geometries.push(geometry);
    }
  }

  render() {
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
    const pose = this.simulation.binding('body_q');
    for (let index = 0; index < this.bodies.length; index += 1) {
      const offset = index * 7;
      const values = pose.subarray(offset, offset + 7);
      if (!values.every(Number.isFinite)) throw new Error('Non-finite robot body transform');
      this.bodies[index].position.set(values[0], values[1], values[2]);
      this.bodies[index].quaternion.set(values[3], values[4], values[5], values[6]);
    }
    const root = this.simulation.manifest.isaacLabDemo.rootBody * 7;
    this.target.set(pose[root], this.isAnymal ? 0.6 : 0.71, -pose[root + 1]);
    this.key.position.set(pose[root] + 3, 5, -pose[root + 1] + 4);
    this.key.target.position.set(pose[root], 0, -pose[root + 1]);
    this.key.target.updateMatrixWorld();
    this.orbit.update(this.target);
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.disposeGround?.();
    this.orbit.dispose();
    this.geometries.forEach((geometry) => geometry.dispose());
    this.materials.forEach((material) => material.dispose());
    this.renderer.dispose();
  }
}
