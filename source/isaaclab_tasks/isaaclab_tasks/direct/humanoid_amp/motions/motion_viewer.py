# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import torch

import mpl_toolkits.mplot3d  # noqa: F401
from motion_loader import MotionLoader


class MotionViewer:
    """
    Helper class to visualize motion data from NumPy-file format.
    """

    def __init__(self, motion_file: str, device: torch.device | str = "cpu", render_scene: bool = False) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.
            render_scene: Whether the scene (space occupied by the skeleton during movement)
                is rendered instead of a reduced view of the skeleton.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        self._figure = None
        self._figure_axes = None
        self._render_scene = render_scene

        # load motions
        self._motion_loader = MotionLoader(motion_file=motion_file, device=device)

        self._num_frames = self._motion_loader.num_frames
        self._current_frame = 0
        self._body_positions = self._motion_loader.body_positions.cpu().numpy()

        print("\nBody")
        for i, name in enumerate(self._motion_loader.body_names):
            minimum = np.min(self._body_positions[:, i], axis=0).round(decimals=2)
            maximum = np.max(self._body_positions[:, i], axis=0).round(decimals=2)
            print(f"  |-- [{name}] minimum position: {minimum}, maximum position: {maximum}")

    def _drawing_callback(self, frame: int) -> None:
        """Drawing callback called each frame"""
        # get current motion frame
        # get data
        vertices = self._body_positions[self._current_frame]
        # draw skeleton state
        self._figure_axes.clear()
        self._figure_axes.scatter(*vertices.T, color="black", depthshade=False)
        # adjust exes according to motion view
        # - scene
        if self._render_scene:
            # compute axes limits
            minimum = np.min(self._body_positions.reshape(-1, 3), axis=0)
            maximum = np.max(self._body_positions.reshape(-1, 3), axis=0)
            center = 0.5 * (maximum + minimum)
            diff = 0.75 * (maximum - minimum)
        # - skeleton
        else:
            # compute axes limits
            minimum = np.min(vertices, axis=0)
            maximum = np.max(vertices, axis=0)
            center = 0.5 * (maximum + minimum)
            diff = np.array([0.75 * np.max(maximum - minimum).item()] * 3)
        # scale view
        self._figure_axes.set_xlim((center[0] - diff[0], center[0] + diff[0]))
        self._figure_axes.set_ylim((center[1] - diff[1], center[1] + diff[1]))
        self._figure_axes.set_zlim((center[2] - diff[2], center[2] + diff[2]))
        self._figure_axes.set_box_aspect(aspect=diff / diff[0])
        # plot ground plane
        x, y = np.meshgrid([center[0] - diff[0], center[0] + diff[0]], [center[1] - diff[1], center[1] + diff[1]])
        self._figure_axes.plot_surface(x, y, np.zeros_like(x), color="green", alpha=0.2)
        # print metadata
        self._figure_axes.set_xlabel("X")
        self._figure_axes.set_ylabel("Y")
        self._figure_axes.set_zlabel("Z")
        self._figure_axes.set_title(f"frame: {self._current_frame}/{self._num_frames}")
        # increase frame counter
        self._current_frame += 1
        if self._current_frame >= self._num_frames:
            self._current_frame = 0

    def show(self) -> None:
        """Show motion"""
        # create a 3D figure
        self._figure = plt.figure()
        self._figure_axes = self._figure.add_subplot(projection="3d")
        # matplotlib animation (the instance must live as long as the animation will run)
        self._animation = matplotlib.animation.FuncAnimation(
            fig=self._figure,
            func=self._drawing_callback,
            frames=self._num_frames,
            interval=1000 * self._motion_loader.dt,
        )
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    parser.add_argument(
        "--render-scene",
        action="store_true",
        default=False,
        help=(
            "Whether the scene (space occupied by the skeleton during movement) is rendered instead of a reduced view"
            " of the skeleton."
        ),
    )
    parser.add_argument("--matplotlib-backend", type=str, default="TkAgg", help="Matplotlib interactive backend")
    args, _ = parser.parse_known_args()

    # https://matplotlib.org/stable/users/explain/figure/backends.html#interactive-backends
    matplotlib.use(args.matplotlib_backend)

    viewer = MotionViewer(args.file, render_scene=args.render_scene)
    viewer.show()
