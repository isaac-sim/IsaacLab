# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

parser = argparse.ArgumentParser(description="Generate terrains using trimesh")
parser.add_argument(
    "--headless", action="store_true", default=False, help="Don't create a window to display each output."
)
args_cli = parser.parse_args()

from isaaclab.app import AppLauncher

# launch omniverse app
# note: we only need to do this because of `TerrainImporter` which uses Omniverse functions
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import argparse
import os
import trimesh

import isaaclab.terrains.trimesh as mesh_gen
from isaaclab.terrains.utils import color_meshes_by_height


def test_flat_terrain(difficulty: float, output_dir: str, headless: bool):
    # parameters for the terrain
    cfg = mesh_gen.MeshPlaneTerrainCfg(size=(8.0, 8.0))
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # write the image to a file
    with open(os.path.join(output_dir, "flat_terrain.jpg"), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption="Flat Terrain")


def test_pyramid_stairs_terrain(difficulty: float, holes: bool, output_dir: str, headless: bool):
    # parameters for the terrain
    cfg = mesh_gen.MeshPyramidStairsTerrainCfg(
        size=(8.0, 8.0),
        border_width=0.2,
        step_width=0.3,
        step_height_range=(0.05, 0.23),
        platform_width=1.5,
        holes=holes,
    )
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # resolve file name
    if holes:
        caption = "Pyramid Stairs Terrain with Holes"
        filename = "pyramid_stairs_terrain_with_holes.jpg"
    else:
        caption = "Pyramid Stairs Terrain"
        filename = "pyramid_stairs_terrain.jpg"
    # write the image to a file
    with open(os.path.join(output_dir, filename), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption=caption)


def test_inverted_pyramid_stairs_terrain(difficulty: float, holes: bool, output_dir: str, headless: bool):
    # parameters for the terrain
    cfg = mesh_gen.MeshInvertedPyramidStairsTerrainCfg(
        size=(8.0, 8.0),
        border_width=0.2,
        step_width=0.3,
        step_height_range=(0.05, 0.23),
        platform_width=1.5,
        holes=holes,
    )
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # resolve file name
    if holes:
        caption = "Inverted Pyramid Stairs Terrain with Holes"
        filename = "inverted_pyramid_stairs_terrain_with_holes.jpg"
    else:
        caption = "Inverted Pyramid Stairs Terrain"
        filename = "inverted_pyramid_stairs_terrain.jpg"
    # write the image to a file
    with open(os.path.join(output_dir, filename), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption=caption)


def test_random_grid_terrain(difficulty: float, holes: bool, output_dir: str, headless: bool):
    # parameters for the terrain
    cfg = mesh_gen.MeshRandomGridTerrainCfg(
        size=(8.0, 8.0),
        platform_width=1.5,
        grid_width=0.75,
        grid_height_range=(0.025, 0.2),
        holes=holes,
    )
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # resolve file name
    if holes:
        caption = "Random Grid Terrain with Holes"
        filename = "random_grid_terrain_with_holes.jpg"
    else:
        caption = "Random Grid Terrain"
        filename = "random_grid_terrain.jpg"
    # write the image to a file
    with open(os.path.join(output_dir, filename), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption=caption)


def test_rails_terrain(difficulty: float, output_dir: str, headless: bool):
    # parameters for the terrain
    cfg = mesh_gen.MeshRailsTerrainCfg(
        size=(8.0, 8.0),
        platform_width=1.5,
        rail_thickness_range=(0.05, 0.1),
        rail_height_range=(0.05, 0.3),
    )
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # write the image to a file
    with open(os.path.join(output_dir, "rails_terrain.jpg"), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption="Rail Terrain")


def test_pit_terrain(difficulty: float, double_pit: bool, output_dir: str, headless: bool):
    # parameters for the terrain
    cfg = mesh_gen.MeshPitTerrainCfg(
        size=(8.0, 8.0), platform_width=1.5, pit_depth_range=(0.05, 1.1), double_pit=double_pit
    )
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # resolve file name
    if double_pit:
        caption = "Pit Terrain with Two Levels"
        filename = "pit_terrain_with_two_levels.jpg"
    else:
        caption = "Pit Terrain"
        filename = "pit_terrain.jpg"
    # write the image to a file
    with open(os.path.join(output_dir, filename), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption=caption)


def test_box_terrain(difficulty: float, double_box: bool, output_dir: str, headless: bool):
    # parameters for the terrain
    cfg = mesh_gen.MeshBoxTerrainCfg(
        size=(8.0, 8.0),
        platform_width=1.5,
        box_height_range=(0.05, 0.2),
        double_box=double_box,
    )
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # resolve file name
    if double_box:
        caption = "Box Terrain with Two Levels"
        filename = "box_terrain_with_two_boxes.jpg"
    else:
        caption = "Box Terrain"
        filename = "box_terrain.jpg"
    # write the image to a file
    with open(os.path.join(output_dir, filename), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption=caption)


def test_gap_terrain(difficulty: float, output_dir: str, headless: bool):
    # parameters for the terrain
    cfg = mesh_gen.MeshGapTerrainCfg(
        size=(8.0, 8.0),
        platform_width=1.5,
        gap_width_range=(0.05, 1.1),
    )
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # write the image to a file
    with open(os.path.join(output_dir, "gap_terrain.jpg"), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption="Gap Terrain")


def test_floating_ring_terrain(difficulty: float, output_dir: str, headless: bool):
    # parameters for the terrain
    cfg = mesh_gen.MeshFloatingRingTerrainCfg(
        size=(8.0, 8.0),
        platform_width=1.5,
        ring_height_range=(0.4, 1.0),
        ring_width_range=(0.5, 1.0),
        ring_thickness=0.05,
    )
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # write the image to a file
    with open(os.path.join(output_dir, "floating_ring_terrain.jpg"), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption="Floating Ring Terrain")


def test_star_terrain(difficulty: float, output_dir: str, headless: bool):
    # parameters for the terrain
    cfg = mesh_gen.MeshStarTerrainCfg(
        size=(8.0, 8.0),
        platform_width=1.5,
        num_bars=5,
        bar_width_range=(0.5, 1.0),
        bar_height_range=(0.05, 0.2),
    )
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # write the image to a file
    with open(os.path.join(output_dir, "star_terrain.jpg"), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption="Star Terrain")


def test_repeated_objects_terrain(
    difficulty: float, object_type: str, output_dir: str, headless: bool, provide_as_string: bool = False
):
    # parameters for the terrain
    if object_type == "pyramid":
        cfg = mesh_gen.MeshRepeatedPyramidsTerrainCfg(
            size=(8.0, 8.0),
            platform_width=1.5,
            abs_height_noise=(-0.5, 0.5),
            object_params_start=mesh_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=40, height=0.05, radius=0.6, max_yx_angle=0.0, degrees=True
            ),
            object_params_end=mesh_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=80, height=0.15, radius=0.6, max_yx_angle=60.0, degrees=True
            ),
        )
    elif object_type == "box":
        cfg = mesh_gen.MeshRepeatedBoxesTerrainCfg(
            size=(8.0, 8.0),
            platform_width=1.5,
            abs_height_noise=(-0.5, 0.5),
            object_params_start=mesh_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=40, height=0.05, size=(0.6, 0.6), max_yx_angle=0.0, degrees=True
            ),
            object_params_end=mesh_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=80, height=0.15, size=(0.6, 0.6), max_yx_angle=60.0, degrees=True
            ),
        )
    elif object_type == "cylinder":
        cfg = mesh_gen.MeshRepeatedCylindersTerrainCfg(
            size=(8.0, 8.0),
            platform_width=1.5,
            abs_height_noise=(-0.5, 0.5),
            object_params_start=mesh_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=40, height=0.05, radius=0.6, max_yx_angle=0.0, degrees=True
            ),
            object_params_end=mesh_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=80, height=0.15, radius=0.6, max_yx_angle=60.0, degrees=True
            ),
        )
    else:
        raise ValueError(f"Invalid object type for repeated objects terrain: {object_type}")

    # provide object_type as string (check that the import works)
    if provide_as_string:
        cfg.object_type = object_type

    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # write the image to a file
    with open(os.path.join(output_dir, f"repeated_objects_{object_type}_terrain.jpg"), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption=f"Repeated Objects Terrain: {object_type}")


def main():
    # Create directory to dump results
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "terrains", "trimesh")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # Read headless mode
    headless = args_cli.headless
    # generate terrains
    test_flat_terrain(difficulty=0.0, output_dir=output_dir, headless=headless)
    test_pyramid_stairs_terrain(difficulty=0.75, holes=False, output_dir=output_dir, headless=headless)
    test_pyramid_stairs_terrain(difficulty=0.75, holes=True, output_dir=output_dir, headless=headless)
    test_inverted_pyramid_stairs_terrain(difficulty=0.75, holes=False, output_dir=output_dir, headless=headless)
    test_inverted_pyramid_stairs_terrain(difficulty=0.75, holes=True, output_dir=output_dir, headless=headless)
    test_random_grid_terrain(difficulty=0.75, holes=False, output_dir=output_dir, headless=headless)
    test_random_grid_terrain(difficulty=0.75, holes=True, output_dir=output_dir, headless=headless)
    test_star_terrain(difficulty=0.75, output_dir=output_dir, headless=headless)
    test_repeated_objects_terrain(difficulty=0.75, object_type="pyramid", output_dir=output_dir, headless=headless)
    test_repeated_objects_terrain(difficulty=0.75, object_type="cylinder", output_dir=output_dir, headless=headless)
    test_repeated_objects_terrain(difficulty=0.75, object_type="box", output_dir=output_dir, headless=headless)
    test_repeated_objects_terrain(
        difficulty=0.75, object_type="cylinder", provide_as_string=True, output_dir=output_dir, headless=headless
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
