// Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

import * as THREE from '../vendor/three.module.min.js';
import { addIsaacLabGround } from './browser-ground.js';
import { OrbitCamera } from './browser-orbit.js';

export class RobotViewer {
  static async load(canvas, simulation) {
    const { visuals } = simulation.manifest.isaacLabDemo;
    const assets = await Promise.all((Array.isArray(visuals) ? visuals : [visuals]).map(async (description) => {
      const chunks = await Promise.all((description.files || [description.file]).map(async (file) => {
        const response = await fetch(new URL(file, simulation.url));
        if (!response.ok) throw new Error(`Robot visual request failed (${response.status})`);
        return response.arrayBuffer();
      }));
      let stream = new Blob(chunks).stream();
      if (description.file.endsWith('.gz')) {
        stream = stream.pipeThrough(new DecompressionStream('gzip'));
      }
      const data = await new Response(stream).arrayBuffer();
      if (data.byteLength !== description.byteLength) throw new Error('Robot visual bundle size does not match its manifest');
      return { description, data };
    }));
    const viewer = new RobotViewer(canvas, simulation, assets);
    try {
      if (!['courtyard', 'warehouse'].includes(viewer.demo.scene)) {
        viewer.disposeGround = await addIsaacLabGround(viewer.scene, simulation.manifest.isaacLabDemo.groundHeight || 0);
      }
      const textures = new Map();
      viewer.textures = [];
      viewer.textureAbort = new AbortController();
      await Promise.all(viewer.materials.filter((material) => material.userData.texture).map(async (material) => {
        const url = new URL(material.userData.texture, simulation.url).href;
        if (!textures.has(url)) textures.set(url, (async () => {
          const response = await fetch(url, { signal: viewer.textureAbort.signal });
          if (!response.ok) throw new Error(`Scene texture request failed (${response.status})`);
          const bitmap = await createImageBitmap(await response.blob(), { imageOrientation: 'flipY' });
          if (viewer.disposed) { bitmap.close(); throw new Error('Scene was disposed'); }
          const texture = new THREE.Texture(bitmap);
          viewer.textures.push(texture);
          texture.needsUpdate = true;
          return texture;
        })());
        const texture = await textures.get(url);
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
        material.map = texture;
        material.needsUpdate = true;
      }));
      return viewer;
    } catch (error) {
      viewer.dispose();
      throw error;
    }
  }

  constructor(canvas, simulation, assets) {
    this.canvas = canvas;
    this.simulation = simulation;
    this.isAnymal = simulation.manifest.isaacLabDemo.kind === 'anymal';
    this.demo = simulation.manifest.isaacLabDemo;
    this.isManipulation = this.demo.kind.startsWith('franka_');
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'low-power' });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = this.demo.toneMappingExposure ?? (this.demo.scene === 'warehouse' ? 1.25 : this.demo.kind === 'microduck' ? 1.05 : 1.65);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#f3f6f7');
    this.scene.fog = new THREE.Fog('#f3f6f7', 7, 15);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 50);
    this.target = new THREE.Vector3();
    this.orbit = new OrbitCamera(canvas, this.camera, this.demo.cameraAzimuth ?? (this.isManipulation ? -2.2 : Math.PI),
      this.isManipulation ? 0.45 : 0.28, this.demo.cameraRadius || (this.isAnymal ? 2.4 : 3.0));

    const microduck = this.demo.kind === 'microduck';
    this.scene.add(new THREE.HemisphereLight('#ffffff', '#acb9be', microduck ? 1.3 : 2.2));
    const key = new THREE.DirectionalLight('#ffffff', microduck ? 2.5 : 3.0);
    key.position.set(3, 5, 4);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    const shadowExtent = microduck ? 0.75 : 2.5;
    key.shadow.camera.left = -shadowExtent;
    key.shadow.camera.right = shadowExtent;
    key.shadow.camera.top = shadowExtent;
    key.shadow.camera.bottom = -shadowExtent;
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 12;
    key.shadow.bias = microduck ? -0.000005 : -0.0001;
    key.shadow.normalBias = microduck ? 0.0003 : 0;
    this.key = key;
    this.scene.add(key);
    this.scene.add(key.target);
    const fill = new THREE.DirectionalLight('#d7e6ed', microduck ? 0.55 : 1.4);
    fill.position.set(-3, 2, -2);
    this.scene.add(fill);

    this.environment = new THREE.Group();
    this.environment.rotation.x = -Math.PI / 2;
    this.scene.add(this.environment);
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
    for (const { description, data } of assets) for (const visual of description.meshes) {
      const vertices = new Float32Array(data, visual.vertexOffset, visual.vertexCount * 3);
      const IndexArray = visual.indexType === 'uint32' ? Uint32Array : Uint16Array;
      const indices = new IndexArray(data, visual.indexOffset, visual.indexCount);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
      geometry.setIndex(new THREE.BufferAttribute(indices, 1));
      if (visual.uvOffset !== undefined) geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(data, visual.uvOffset, visual.vertexCount * 2), 2));
      geometry.computeVertexNormals();
      let material = materials[visual.material];
      if (visual.color) {
        const name = JSON.stringify([visual.color, visual.texture, visual.roughness, visual.opacity, visual.flatShading]);
        if (!materials[name]) {
          materials[name] = new THREE.MeshStandardMaterial({
            color: new THREE.Color().setRGB(...visual.color),
            roughness: visual.roughness ?? (microduck ? 0.5 : 0.6), metalness: microduck ? 0 : 0.05,
            flatShading: visual.flatShading ?? microduck,
            opacity: visual.opacity ?? 1, transparent: (visual.opacity ?? 1) < 1,
            side: visual.body < 0 ? THREE.DoubleSide : THREE.FrontSide,
          });
          materials[name].userData.texture = visual.texture;
          this.materials.push(materials[name]);
        }
        material = materials[name];
      }
      const mesh = visual.instances
        ? new THREE.InstancedMesh(geometry, material, visual.instances.length)
        : new THREE.Mesh(geometry, material);
      visual.instances?.forEach((matrix, index) => mesh.setMatrixAt(index, new THREE.Matrix4().fromArray(matrix)));
      mesh.castShadow = (visual.opacity ?? 1) > 0.95 && !/_(zero|one|two|three|four|five|six)_link$/.test(visual.name);
      mesh.receiveShadow = true;
      // Free warehouse props share the solver's pose array, but are not robot drag targets.
      if (visual.environment && visual.body >= 0) this.environment.add(this.bodies[visual.body]);
      (visual.body < 0 ? this.environment : this.bodies[visual.body]).add(mesh);
      this.geometries.push(geometry);
    }
    if (this.demo.commandRange) {
      const geometry = new THREE.SphereGeometry(0.025, 20, 12);
      const material = new THREE.MeshStandardMaterial({ color: '#76b900', emissive: '#3a6000', transparent: true, opacity: 0.8 });
      this.goal = new THREE.Mesh(geometry, material);
      this.robot.add(this.goal);
      this.geometries.push(geometry);
      this.materials.push(material);
    }
    if (this.demo.scene === 'courtyard') this.addCourtyard();
    if (this.demo.scene === 'warehouse') this.scene.fog = new THREE.Fog('#f3f6f7', 14, 28);
    if (this.demo.pushMaxForce) this.addMousePush();
  }

  addCourtyard() {
    // Ground markings add scale and depth without changing the task's flat contact surface.
    this.scene.background.set('#dce8ed');
    this.scene.fog = new THREE.Fog('#dce8ed', 4, 12);
    const grassMaterial = new THREE.MeshStandardMaterial({ color: '#9fac87', roughness: 1 });
    const lawn = new THREE.Shape([
      new THREE.Vector2(-10, -10), new THREE.Vector2(10, -10),
      new THREE.Vector2(10, 10), new THREE.Vector2(-10, 10),
    ]);
    lawn.holes.push(new THREE.Path([
      new THREE.Vector2(-2.5, -2.5), new THREE.Vector2(-2.5, 2.5),
      new THREE.Vector2(2.5, 2.5), new THREE.Vector2(2.5, -2.5),
    ]));
    const grassGeometry = new THREE.ShapeGeometry(lawn);
    const grass = new THREE.Mesh(grassGeometry, grassMaterial);
    grass.rotation.x = -Math.PI / 2;
    grass.receiveShadow = true;
    this.scene.add(grass);
    this.materials.push(grassMaterial);
    this.geometries.push(grassGeometry);
    const material = new THREE.MeshStandardMaterial({ color: '#ddd4bf', roughness: 1 });
    const geometry = new THREE.PlaneGeometry(5, 5);
    const court = new THREE.Mesh(geometry, material);
    court.rotation.x = -Math.PI / 2;
    court.receiveShadow = true;
    this.scene.add(court);
    this.materials.push(material);
    this.geometries.push(geometry);
    const lines = new THREE.GridHelper(5, 20, '#bfb299', '#cbbfa9');
    lines.position.y = 0.001;
    this.scene.add(lines);
    this.geometries.push(lines.geometry);
    this.materials.push(lines.material);
    const green = new THREE.MeshStandardMaterial({ color: '#637e5a', roughness: 1 });
    const wood = new THREE.MeshStandardMaterial({ color: '#aa8565', roughness: 0.9 });
    this.materials.push(green, wood);
    for (const [x, z] of [[-2.9, -2.9], [2.9, -2.9], [-2.9, 2.9], [2.9, 2.9]]) {
      const trunkGeometry = new THREE.CylinderGeometry(0.025, 0.035, 0.4, 8);
      const trunk = new THREE.Mesh(trunkGeometry, wood);
      trunk.position.set(x, 0.2, z);
      const crownGeometry = new THREE.IcosahedronGeometry(0.22, 1);
      const crown = new THREE.Mesh(crownGeometry, green);
      crown.scale.set(0.85, 1.2, 0.85);
      crown.position.set(x, 0.5, z);
      trunk.castShadow = crown.castShadow = true;
      this.scene.add(trunk, crown);
      this.geometries.push(trunkGeometry, crownGeometry);
    }
  }

  addMousePush() {
    const raycaster = new THREE.Raycaster();
    const point = new THREE.Vector3();
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -(this.demo.cameraHeight || 0.13));
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.2, '#e37a40', 0.04, 0.025);
    arrow.visible = false;
    this.scene.add(arrow);
    this.pushArrow = arrow;
    let active = null;
    const project = (event) => {
      const rect = this.canvas.getBoundingClientRect();
      raycaster.setFromCamera(new THREE.Vector2(
        2 * (event.clientX - rect.left) / rect.width - 1,
        1 - 2 * (event.clientY - rect.top) / rect.height,
      ), this.camera);
      return raycaster.ray.intersectPlane(plane, point);
    };
    const down = (event) => {
      if (active !== null) {
        event.stopImmediatePropagation();
        return;
      }
      if (event.button !== 0 || !project(event)) return;
      if (!raycaster.intersectObject(this.robot, true).length) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      active = { id: event.pointerId, start: point.clone() };
      this.canvas.setPointerCapture(event.pointerId);
      this.canvas.classList.add('browser-demo-pushing');
    };
    const move = (event) => {
      if (event.pointerId !== active?.id || !project(event)) return;
      const direction = point.clone().sub(active.start);
      direction.y = 0;
      const force = this.simulation.binding('force');
      const length = direction.length();
      const magnitude = length > 0.003 ? this.demo.pushMaxForce : 0;
      direction.normalize();
      force[0] = direction.x * magnitude;
      force[1] = -direction.z * magnitude;
      arrow.setDirection(direction);
      arrow.setLength(Math.max(0.03, Math.min(length, 0.3)), 0.035, 0.02);
      arrow.visible = magnitude > 0;
    };
    const release = (event) => {
      if (event?.pointerId !== undefined && event.pointerId !== active?.id) return;
      const id = active?.id;
      active = null;
      if (id !== undefined && this.canvas.hasPointerCapture(id)) this.canvas.releasePointerCapture(id);
      this.simulation.binding('force').fill(0);
      arrow.visible = false;
      this.canvas.classList.remove('browser-demo-pushing');
    };
    this.releasePush = release;
    this.canvas.addEventListener('pointerdown', down, { capture: true });
    this.canvas.addEventListener('pointermove', move);
    for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) this.canvas.addEventListener(name, release);
    window.addEventListener('blur', release);
    this.disposePush = () => {
      release();
      this.canvas.removeEventListener('pointerdown', down, { capture: true });
      this.canvas.removeEventListener('pointermove', move);
      for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) this.canvas.removeEventListener(name, release);
      window.removeEventListener('blur', release);
      arrow.line.geometry.dispose(); arrow.line.material.dispose();
      arrow.cone.geometry.dispose(); arrow.cone.material.dispose();
    };
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
    if (this.isManipulation) {
      this.target.fromArray(this.demo.cameraTarget);
      if (this.goal) {
        const command = this.simulation.binding('command');
        this.goal.position.set(command[0], command[1], command[2]);
        this.goal.position.applyQuaternion(this.bodies[this.demo.rootBody].quaternion);
        this.goal.position.add(this.bodies[this.demo.rootBody].position);
      }
    } else this.target.set(pose[root], this.demo.cameraHeight ?? (this.isAnymal ? 0.6 : 0.71), -pose[root + 1]);
    if (this.pushArrow) this.pushArrow.position.set(pose[root], pose[root + 2] + 0.03, -pose[root + 1]);
    this.key.position.set(pose[root] + 3, 5, -pose[root + 1] + 4);
    this.key.target.position.set(pose[root], 0, -pose[root + 1]);
    this.key.target.updateMatrixWorld();
    this.orbit.update(this.target);
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.disposed = true;
    this.textureAbort?.abort();
    this.disposePush?.();
    this.disposeGround?.();
    this.orbit.dispose();
    this.geometries.forEach((geometry) => geometry.dispose());
    this.materials.forEach((material) => material.dispose());
    this.textures?.forEach((texture) => { texture.dispose(); texture.image.close(); });
    this.renderer.dispose();
  }
}
