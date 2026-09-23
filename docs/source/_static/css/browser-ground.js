// Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

import * as THREE from '../vendor/three.module.min.js';

const assetRoot = new URL('../browser_demos/shared/', import.meta.url);

export async function addIsaacLabGround(scene) {
  const loader = new THREE.TextureLoader();
  const [albedo, roughness] = await Promise.all([
    loader.loadAsync(new URL('default_ground_plane_albedo.png', assetRoot).href),
    loader.loadAsync(new URL('default_ground_plane_roughness.png', assetRoot).href),
  ]);
  const size = 100;
  const tileSize = 2;
  for (const texture of [albedo, roughness]) {
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(size / tileSize, size / tileSize);
    texture.anisotropy = 4;
  }
  albedo.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.MeshStandardMaterial({
    map: albedo,
    roughnessMap: roughness,
    roughness: 1,
    metalness: 0,
  });
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(size, size), material);
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);
  return () => {
    scene.remove(ground);
    ground.geometry.dispose();
    material.dispose();
    albedo.dispose();
    roughness.dispose();
  };
}
