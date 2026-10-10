// Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

import * as THREE from '../vendor/three.module.min.js';
import { addIsaacLabGround } from './browser-ground.js';
import { OrbitCamera } from './browser-orbit.js';

export class StiffnessViewer {
  static async load(canvas, simulation) {
    const viewer = new StiffnessViewer(canvas, simulation);
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
    this.renderer.toneMappingExposure = 1.3;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#f4f6f7');
    this.scene.fog = new THREE.Fog('#f4f6f7', 8, 18);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 30);
    const demo = simulation.manifest.isaacLabDemo;
    const cloth = demo.kind === 'cloth_bending';
    const mpm = demo.kind === 'mpm';
    this.target = new THREE.Vector3(0, cloth || mpm ? 0.55 : 0.65, 0);
    this.orbit = new OrbitCamera(canvas, this.camera, cloth || mpm ? Math.PI / 2 : 1.3,
      mpm ? 0.18 : cloth ? 0.22 : 0.26, mpm ? 2.8 : cloth ? 4.8 : 3.6);

    this.scene.add(new THREE.AmbientLight('#ffffff', 0.75));
    this.scene.add(new THREE.HemisphereLight('#ffffff', '#aebbc1', 0.7));
    const key = new THREE.DirectionalLight('#fffdf8', 2.2);
    key.position.set(3.5, 6, 4.5);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = -3;
    key.shadow.camera.right = 3;
    key.shadow.camera.top = 3;
    key.shadow.camera.bottom = -3;
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 12;
    key.shadow.bias = -0.0001;
    this.scene.add(key);

    const supports = mpm ? [demo.cylinder] : (demo.supports || []);
    this.supports = supports.map((support) => {
      const geometry = new THREE.CylinderGeometry(support.radius, support.radius, support.height, 32);
      const material = new THREE.MeshStandardMaterial({ color: '#a7b7bd', roughness: 0.28, metalness: 0.6 });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(support.position[0], support.position[2], -support.position[1]);
      mesh.rotation.x = Math.PI / 2;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.scene.add(mesh);
      return mesh;
    });
    if (mpm && demo.tub) {
      const { halfWidth, halfDepth, height, thickness } = demo.tub;
      const panels = [
        { size: [2 * (halfWidth + thickness), thickness, 2 * (halfDepth + thickness)], position: [0, -thickness / 2, 0] },
        ...[-1, 1].flatMap((side) => [
          { size: [thickness, height, 2 * (halfDepth + thickness)], position: [side * (halfWidth + thickness / 2), height / 2, 0] },
          { size: [2 * halfWidth, height, thickness], position: [0, height / 2, side * (halfDepth + thickness / 2)] },
        ]),
      ];
      for (const [index, panel] of panels.entries()) {
        const geometry = new THREE.BoxGeometry(...panel.size);
        const material = new THREE.MeshStandardMaterial({
          color: index === 0 ? '#bdcbd0' : '#bed0d7',
          roughness: index === 0 ? 0.65 : 0.2,
          transparent: index !== 0,
          opacity: index === 0 ? 1 : 0.2,
          depthWrite: index === 0,
          polygonOffset: index === 0,
          polygonOffsetFactor: -1,
          polygonOffsetUnits: -1,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(...panel.position);
        mesh.receiveShadow = index === 0;
        const outline = new THREE.LineSegments(new THREE.EdgesGeometry(geometry),
          new THREE.LineBasicMaterial({ color: '#829aa5', transparent: true, opacity: 0.65 }));
        mesh.add(outline);
        this.scene.add(mesh);
        this.supports.push(mesh, outline);
      }
    }
    this.meshes = simulation.manifest.visualization.softMeshes.map((description) => {
      const geometry = new THREE.BufferGeometry();
      const position = new THREE.BufferAttribute(new Float32Array(description.particles.length * 3), 3);
      position.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('position', position);
      geometry.setIndex(description.indices);
      const material = new THREE.MeshStandardMaterial({
        color: description.color,
        roughness: 0.42,
        metalness: 0.1,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.frustumCulled = false;
      this.scene.add(mesh);
      return { mesh, particles: description.particles };
    });
    this.particleClouds = (simulation.manifest.visualization.particleClouds || []).map((description) => {
      const geometry = new THREE.SphereGeometry(demo.particleRadius, 8, 6);
      const material = new THREE.MeshStandardMaterial({ color: description.color, roughness: 0.8, metalness: 0 });
      const mesh = new THREE.InstancedMesh(geometry, material, description.particles.length);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.frustumCulled = false;
      const color = new THREE.Color();
      for (let index = 0; index < description.particles.length; index += 1) {
        color.setScalar(0.8 + 0.35 * ((index * 37) % 101) / 100);
        mesh.setColorAt(index, color);
      }
      this.scene.add(mesh);
      return { mesh, particles: description.particles };
    });
    this.particleMatrix = new THREE.Matrix4();
  }

  setParticleColor(color) {
    for (const { mesh } of this.particleClouds) mesh.material.color.set(color);
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
    const particles = this.simulation.binding('particle_q');
    for (const { mesh, particles: particleIds } of this.meshes) {
      const positions = mesh.geometry.attributes.position.array;
      for (let index = 0; index < particleIds.length; index += 1) {
        const source = particleIds[index] * 3;
        const x = particles[source];
        const y = particles[source + 1];
        const z = particles[source + 2];
        if (![x, y, z].every(Number.isFinite)) throw new Error('Non-finite VBD particle position');
        const target = index * 3;
        positions[target] = x;
        positions[target + 1] = z;
        positions[target + 2] = -y;
      }
      mesh.geometry.attributes.position.needsUpdate = true;
      mesh.geometry.computeVertexNormals();
    }
    for (const { mesh, particles: particleIds } of this.particleClouds) {
      for (let index = 0; index < particleIds.length; index += 1) {
        const source = particleIds[index] * 3;
        const x = particles[source];
        const y = particles[source + 1];
        const z = particles[source + 2];
        if (![x, y, z].every(Number.isFinite)) throw new Error('Non-finite MPM particle position');
        this.particleMatrix.makeTranslation(x, z, -y);
        mesh.setMatrixAt(index, this.particleMatrix);
      }
      mesh.instanceMatrix.needsUpdate = true;
    }
    this.orbit.update(this.target);
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.disposeGround?.();
    this.orbit.dispose();
    for (const support of this.supports) {
      support.geometry.dispose();
      support.material.dispose();
    }
    for (const { mesh } of [...this.meshes, ...this.particleClouds]) {
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    this.renderer.dispose();
  }
}
