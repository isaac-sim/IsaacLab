# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX view provenance for fixed scene export; target authoring lives in core."""

from isaaclab.sim.usd_export import AssetPaths


class SceneAdapter:
    """Map PhysX tensor identities to the public asset data order."""

    def __init__(self, scene):
        self.scene = scene

    def paths(self, asset) -> AssetPaths:
        view = asset.root_view
        if asset in self.scene.articulations.values():
            return AssetPaths(
                [
                    (str(path), asset.body_names.index(name))
                    for path, name in zip(view.link_paths[0], asset.backend_body_names)
                ],
                [
                    (str(path), asset.joint_names.index(name))
                    for path, name in zip(view.dof_paths[0], asset.backend_joint_names)
                ],
            )
        return AssetPaths([(str(path), row) for row, path in enumerate(view.prim_paths)], [])

    def write_extensions(self, writer) -> None:
        """PhysX fixed configuration is already authored by spawners and scene initialization."""
