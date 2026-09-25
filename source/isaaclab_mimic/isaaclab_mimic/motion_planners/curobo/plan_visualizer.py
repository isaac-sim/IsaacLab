# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Utility for visualizing motion plans using Rerun.

This module provides tools to visualize motion plans, robot poses, and collision spheres
using Rerun's visualization capabilities. It helps in debugging and validating collision-free paths.
"""

import atexit
import os
import signal
import subprocess
import threading
import time
import weakref
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

# Check if rerun is installed
try:
    import rerun as rr
except ImportError:
    raise ImportError("Rerun is not installed!")

_RR_HAS_TRANSFORM_AXES = hasattr(rr, "TransformAxes3D")

from curobo.types.state import JointState

import isaaclab.utils.math as PoseUtils

# Import psutil for process management
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Process monitoring will be limited.")

if TYPE_CHECKING:  # For type hints only
    import trimesh


# Global registry to track all PlanVisualizer instances for cleanup
_GLOBAL_PLAN_VISUALIZERS: list["PlanVisualizer"] = []


def _cleanup_all_plan_visualizers():
    """Enhanced global cleanup function with better process killing."""
    global _GLOBAL_PLAN_VISUALIZERS

    if PSUTIL_AVAILABLE:
        killed_count = 0
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            # Check if it's a rerun process
            if (proc.info["name"] and "rerun" in proc.info["name"].lower()) or (
                proc.info["cmdline"] and any("rerun" in str(arg).lower() for arg in proc.info["cmdline"])
            ):
                proc.kill()
                killed_count += 1

        print(f"Killed {killed_count} Rerun viewer processes on script exit")
    else:
        # Fallback to pkill
        subprocess.run(["pkill", "-f", "rerun"], stderr=subprocess.DEVNULL, check=False)
        print("Used pkill fallback to close Rerun processes")

    # Also clean up individual instances
    for visualizer in _GLOBAL_PLAN_VISUALIZERS[:]:
        if not visualizer._closed:
            visualizer.close()

    _GLOBAL_PLAN_VISUALIZERS.clear()


# Register global cleanup on module import
atexit.register(_cleanup_all_plan_visualizers)


class PlanVisualizer:
    """Visualizes motion plans using Rerun.

    This class provides methods to visualize:
    1. Robot poses along a planned trajectory
    2. Attached objects and their collision spheres
    3. Robot collision spheres
    4. Target poses and waypoints
    """

    def __init__(
        self,
        robot_name: str = "panda",
        recording_id: str | None = None,
        debug: bool = False,
        save_path: str | None = None,
        base_translation: np.ndarray | None = None,
    ):
        """Initialize the plan visualizer.

        Args:
            robot_name: Name of the robot for visualization
            recording_id: Optional ID for the Rerun recording
            debug: Whether to print debug information
            save_path: Optional path to save the recording
            base_translation: Optional base translation to apply to all visualized entities
        """
        self.robot_name = robot_name
        self.debug = debug
        self.recording_id = recording_id or f"motion_plan_{robot_name}"
        self.save_path = save_path
        self._closed = False
        # Translation offset applied to all visualized entities (for multi-env setups)
        self._base_translation = (
            np.array(base_translation, dtype=float) if base_translation is not None else np.zeros(3)
        )

        # Enhanced process management
        self._parent_pid = os.getpid()
        self._monitor_thread = None
        self._monitor_active = False

        # Motion generator reference for sphere animation (set by CuroboPlanner)
        self._motion_gen_ref = None

        # Register this instance globally for cleanup
        global _GLOBAL_PLAN_VISUALIZERS
        _GLOBAL_PLAN_VISUALIZERS.append(self)

        # Initialize Rerun
        rr.init(self.recording_id, spawn=False)

        # Spawn viewer and keep handle if provided so we can terminate it later
        try:
            self._rerun_process = rr.spawn()
        except Exception:
            # Older versions of Rerun may not return a process handle
            self._rerun_process = None

        # Set up coordinate system
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)

        # Store visualization state
        self._current_frame = 0
        self._sphere_entities: dict[str, list[str]] = {"robot": [], "attached": [], "target": []}

        # Start enhanced parent process monitoring
        self._start_parent_process_monitoring()

        # Use weakref.finalize for cleanup when object is garbage collected
        self._finalizer = weakref.finalize(
            self, self._cleanup_class_resources, self.recording_id, self.save_path, debug
        )

        # Also register atexit handler as backup for normal script termination
        # Store values locally to avoid capturing self in the closure
        recording_id = self.recording_id
        save_path = self.save_path
        debug_flag = debug
        atexit.register(self._cleanup_class_resources, recording_id, save_path, debug_flag)

        # Store original signal handlers so we can restore them after cleanup
        self._original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_DFL)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, signal.SIG_DFL)

        # Handle Ctrl+C (SIGINT) and termination (SIGTERM) signals
        def signal_handler(signum, frame):
            if self.debug:
                print(f"Received signal {signum}, closing Rerun viewer...")
            self._cleanup_on_exit()

            # Restore original signal handler and re-raise the signal
            if signum == signal.SIGINT:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
            elif signum == signal.SIGTERM:
                signal.signal(signal.SIGTERM, self._original_sigterm_handler)
            os.kill(os.getpid(), signum)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if self.debug:
            print(f"Initialized Rerun visualization with recording ID: {self.recording_id}")
            if np.linalg.norm(self._base_translation) > 0:
                print(f"Applying translation offset: {self._base_translation}")
            if PSUTIL_AVAILABLE:
                print("Enhanced process monitoring enabled")

    def _start_parent_process_monitoring(self) -> None:
        """Start monitoring the parent process and cleanup when it dies."""
        if not PSUTIL_AVAILABLE:
            if self.debug:
                print("psutil not available, skipping parent process monitoring")
            return

        self._monitor_active = True

        def monitor_parent_process() -> None:
            """Monitor thread function that watches the parent process."""
            if self.debug:
                print(f"Starting parent process monitor for PID {self._parent_pid}")

            # Get parent process handle
            parent_process = psutil.Process(self._parent_pid)

            # Monitor parent process
            while self._monitor_active:
                try:
                    if not parent_process.is_running():
                        if self.debug:
                            print(f"Parent process {self._parent_pid} died, cleaning up Rerun...")
                        self._kill_rerun_processes()
                        break

                    # Check every 2 seconds
                    time.sleep(2)

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    if self.debug:
                        print(f"Parent process {self._parent_pid} no longer accessible, cleaning up...")
                    self._kill_rerun_processes()
                    break
                except Exception as e:
                    if self.debug:
                        print(f"Error in parent process monitor: {e}")
                    break

        # Start monitor thread
        self._monitor_thread = threading.Thread(target=monitor_parent_process, daemon=True)
        self._monitor_thread.start()

    def _kill_rerun_processes(self) -> None:
        """Enhanced method to kill Rerun viewer processes using psutil."""
        try:
            if PSUTIL_AVAILABLE:
                killed_count = 0
                for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                    try:
                        # Check if it's a rerun process
                        is_rerun = False

                        # Check process name
                        if proc.info["name"] and "rerun" in proc.info["name"].lower():
                            is_rerun = True

                        # Check command line arguments
                        if proc.info["cmdline"] and any("rerun" in str(arg).lower() for arg in proc.info["cmdline"]):
                            is_rerun = True

                        if is_rerun:
                            proc.kill()
                            killed_count += 1
                            if self.debug:
                                print(f"Killed Rerun process {proc.info['pid']} ({proc.info['name']})")

                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        # Process already dead or inaccessible
                        pass
                    except Exception as e:
                        if self.debug:
                            print(f"Error killing process: {e}")

                if self.debug:
                    print(f"Killed {killed_count} Rerun processes using psutil")

            else:
                # Fallback to pkill if psutil not available
                result = subprocess.run(["pkill", "-f", "rerun"], stderr=subprocess.DEVNULL, check=False)
                if self.debug:
                    print(f"Used pkill fallback (return code: {result.returncode})")

        except Exception as e:
            if self.debug:
                print(f"Error killing rerun processes: {e}")

    @staticmethod
    def _cleanup_class_resources(recording_id: str, save_path: str | None, debug: bool) -> None:
        """Static method for cleanup that doesn't hold references to the instance.

        This is called by weakref.finalize when the object is garbage collected.
        """
        if debug:
            print(f"Cleaning up Rerun visualization for {recording_id}")

        # Disconnect from Rerun
        rr.disconnect()

        # Save to file if requested
        if save_path is not None:
            rr.save(save_path)
            if debug:
                print(f"Saved Rerun recording to {save_path}")

        # Enhanced process killing
        if PSUTIL_AVAILABLE:
            killed_count = 0
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                if (proc.info["name"] and "rerun" in proc.info["name"].lower()) or (
                    proc.info["cmdline"] and any("rerun" in str(arg).lower() for arg in proc.info["cmdline"])
                ):
                    proc.kill()
                    killed_count += 1

            if debug:
                print(f"Killed {killed_count} Rerun processes during cleanup")
        else:
            subprocess.run(["pkill", "-f", "rerun"], stderr=subprocess.DEVNULL, check=False)

        if debug:
            print("Cleanup completed")

    def _cleanup_on_exit(self) -> None:
        """Manual cleanup method for signal handlers."""
        if not self._closed:
            # Stop monitoring thread
            self._monitor_active = False

            self.close()
            self._kill_rerun_processes()

    def close(self) -> None:
        """Close the Rerun visualization with enhanced cleanup."""
        if self._closed:
            return

        # Stop parent process monitoring
        self._monitor_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            # Give the thread a moment to stop gracefully
            time.sleep(0.1)

        # Disconnect from Rerun (closes connections, servers, and files)
        rr.disconnect()

        # Save to file if requested
        if self.save_path is not None:
            rr.save(self.save_path)
            if self.debug:
                print(f"Saved Rerun recording to {self.save_path}")

        self._closed = True

        # Terminate viewer process if we have a handle
        try:
            process = getattr(self, "_rerun_process", None)
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
        except Exception:
            pass

        # Enhanced process killing
        self._kill_rerun_processes()

        # Remove from global registry
        global _GLOBAL_PLAN_VISUALIZERS
        if self in _GLOBAL_PLAN_VISUALIZERS:
            _GLOBAL_PLAN_VISUALIZERS.remove(self)

        if self.debug:
            print("Closed Rerun visualization with enhanced cleanup")

    def visualize_plan(
        self,
        plan: JointState,
        target_pose: torch.Tensor,
        robot_spheres: list[Any] | None = None,
        attached_spheres: list[Any] | None = None,
        ee_positions: np.ndarray | None = None,
        world_scene: Optional["trimesh.Scene"] = None,
    ) -> None:
        """Visualize a complete motion plan with all components.

        Args:
            plan: Joint state trajectory to visualize
            target_pose: Target end-effector pose
            robot_spheres: Optional list of robot collision spheres
            attached_spheres: Optional list of attached object spheres
            ee_positions: Optional end-effector positions
            world_scene: Optional world scene to visualize
        """
        if self.debug:
            robot_count = len(robot_spheres) if robot_spheres else 0
            attached_count = len(attached_spheres) if attached_spheres else 0
            offset_info = (
                f"offset={self._base_translation}" if np.linalg.norm(self._base_translation) > 0 else "no offset"
            )
            print(
                f"Visualizing plan: {len(plan.position)} waypoints, {robot_count} robot spheres (with offset),"
                f" {attached_count} attached spheres (no offset), {offset_info}"
            )

        # Set timeline for static visualization (separate from animation)
        rr.set_time("static_plan", sequence=self._current_frame)
        self._current_frame += 1

        # Clear previous visualization of dynamic entities (keep static meshes)
        self._clear_visualization()

        # If a scene is supplied and not yet logged, draw it once
        if world_scene is not None:
            self._visualize_world_scene(world_scene)

        # Visualize target pose
        self._visualize_target_pose(target_pose)

        # Visualize trajectory (end-effector positions if provided)
        self._visualize_trajectory(plan, ee_positions)

        # Visualize spheres if provided
        if robot_spheres:
            self._visualize_robot_spheres(robot_spheres)
        if attached_spheres:
            self._visualize_attached_spheres(attached_spheres)
        else:
            # Clear any existing attached sphere visualization when no objects are attached
            self._clear_attached_spheres()

    def _clear_visualization(self) -> None:
        """Clear all visualization entities."""
        # Clear dynamic trajectory, target, and finger logs to avoid artifacts between visualizations
        dynamic_paths = [
            "trajectory",
            "target",
            "anim",
        ]

        for path in dynamic_paths:
            rr.log(f"world/{path}", rr.Clear(recursive=True))

        for entity_type, entities in self._sphere_entities.items():
            for entity in entities:
                rr.log(f"world/{entity_type}/{entity}", rr.Clear(recursive=True))
            self._sphere_entities[entity_type] = []
        self._current_frame = 0

    def clear_visualization(self) -> None:
        """Public method to clear the visualization."""
        self._clear_visualization()

    def _visualize_target_pose(self, target_pose: torch.Tensor) -> None:
        """Visualize the target end-effector pose.

        Args:
            target_pose: Target pose as 4x4 transformation matrix
        """
        pos, rot = PoseUtils.unmake_pose(target_pose)

        # Convert to numpy arrays
        pos_np = pos.detach().cpu().numpy() if torch.is_tensor(pos) else np.array(pos)
        rot_np = rot.detach().cpu().numpy() if torch.is_tensor(rot) else np.array(rot)

        # Ensure arrays are the right shape
        pos_np = pos_np.reshape(-1)
        rot_np = rot_np.reshape(3, 3)

        # Apply translation offset
        pos_np += self._base_translation

        # Log target position
        rr.log(
            "world/target/position",
            rr.Points3D(
                positions=np.array([pos_np]),
                colors=[[255, 0, 0]],  # Red
                radii=[0.02],
            ),
        )

        # Log target orientation as coordinate frame
        rr.log(
            "world/target/frame",
            rr.Transform3D(
                translation=pos_np,
                mat3x3=rot_np,
            ),
        )

    def _visualize_trajectory(
        self,
        plan: JointState,
        ee_positions: np.ndarray | None = None,
    ) -> None:
        """Visualize the robot trajectory.

        Args:
            plan: Joint state trajectory
            ee_positions: Optional end-effector positions
        """
        if ee_positions is None:
            raw = plan.position.detach().cpu().numpy() if torch.is_tensor(plan.position) else np.array(plan.position)
            if raw.shape[1] >= 3:
                positions = raw[:, :3]
            else:
                raise ValueError("ee_positions not provided and joint positions are not 3-D")
        else:
            positions = ee_positions

        # Apply translation offset
        positions = positions + self._base_translation

        # Log trajectory points
        rr.log(
            "world/trajectory",
            rr.LineStrips3D(
                [positions],  # single strip consisting of all waypoints
                colors=[[0, 100, 255]],  # Blue
                radii=[0.005],
            ),
            static=True,
        )

        # Log keyframes
        for i, pos in enumerate(positions):
            rr.log(
                f"world/trajectory/keyframe_{i}",
                rr.Points3D(
                    positions=np.array([pos]),
                    colors=[[0, 100, 255]],  # Blue
                    radii=[0.01],
                ),
                static=True,
            )

    def _visualize_robot_spheres(self, spheres: list[Any]) -> None:
        """Visualize robot collision spheres.

        Args:
            spheres: List of robot collision spheres
        """
        self._log_spheres(
            spheres=spheres,
            entity_type="robot",
            color=[0, 255, 100, 128],  # Semi-transparent green
            apply_offset=True,
        )

    def _visualize_attached_spheres(self, spheres: list[Any]) -> None:
        """Visualize attached object collision spheres.

        Args:
            spheres: List of attached object spheres
        """
        self._log_spheres(
            spheres=spheres,
            entity_type="attached",
            color=[255, 0, 0, 128],  # Semi-transparent red
            apply_offset=False,
        )

    def _clear_attached_spheres(self) -> None:
        """Clear all attached object spheres."""
        for entity_id in self._sphere_entities.get("attached", []):
            rr.log(f"world/attached/{entity_id}", rr.Clear(recursive=True))
        self._sphere_entities["attached"] = []

    # ---------------------------------------------------------------------
    # PRIVATE UTILITIES
    # ---------------------------------------------------------------------

    def _log_spheres(
        self,
        spheres: list[Any],
        entity_type: str,
        color: list[int],
        apply_offset: bool = False,
    ) -> None:
        """Generic helper for sphere visualization.

        Args:
            spheres: List of CuRobo ``Sphere`` objects.
            entity_type: Log path prefix (``robot`` or ``attached``).
            color: RGBA color for the spheres.
            apply_offset: Whether to add ``self._base_translation`` to sphere positions.
        """
        for i, sphere in enumerate(spheres):
            entity_id = f"sphere_{i}"
            # Track entities so we can clear them later
            self._sphere_entities.setdefault(entity_type, []).append(entity_id)

            # Convert position to numpy and optionally apply offset
            pos = (
                sphere.position.detach().cpu().numpy()
                if torch.is_tensor(sphere.position)
                else np.array(sphere.position)
            )
            if apply_offset:
                pos = pos + self._base_translation
            pos = pos.reshape(-1)  # Ensure 1-D

            # Log sphere via Rerun
            rr.log(
                f"world/{entity_type}/{entity_id}",
                rr.Points3D(positions=np.array([pos]), colors=[color], radii=[float(sphere.radius)]),
            )

    def _visualize_world_scene(self, scene: "trimesh.Scene") -> None:
        """Log world geometry and dynamic transforms each call.

        Geometry is sent once (cached), but transforms are updated every invocation
        so objects that move (cubes after randomization) appear at the correct
        pose per episode/frame.
        """
        import trimesh  # local import to avoid hard dependency at top

        def _to_rerun_mesh(mesh: "trimesh.Trimesh") -> "rr.Mesh3D":
            # Basic conversion without materials
            return rr.Mesh3D(
                vertex_positions=mesh.vertices,
                triangle_indices=mesh.faces,
                vertex_normals=mesh.vertex_normals if mesh.vertex_normals is not None else None,
            )

        if not hasattr(self, "_logged_geometry"):
            self._logged_geometry = set()

        for node in scene.graph.nodes_geometry:
            tform, geom_key = scene.graph.get(node)
            mesh = scene.geometry.get(geom_key)
            if mesh is None:
                continue
            rr_path = f"world/scene/{node.replace('/', '_')}"

            # Always update transform (objects may move between calls)
            # NOTE: World scene objects are already in correct world coordinates, no offset needed
            if _RR_HAS_TRANSFORM_AXES:
                # rerun >= 0.28: axis_length moved to TransformAxes3D
                rr.log(rr_path, rr.Transform3D(translation=tform[:3, 3], mat3x3=tform[:3, :3]), static=False)
                rr.log(rr_path, rr.TransformAxes3D(axis_length=0.25), static=False)
            else:
                # rerun <= 0.27: axis_length on Transform3D directly
                rr.log(
                    rr_path,
                    rr.Transform3D(translation=tform[:3, 3], mat3x3=tform[:3, :3], axis_length=0.25),
                    static=False,
                )

            # Geometry: send only once per node to avoid duplicates
            if rr_path not in self._logged_geometry:
                if isinstance(mesh, trimesh.Trimesh):
                    rr_mesh = _to_rerun_mesh(mesh)
                elif isinstance(mesh, trimesh.PointCloud):
                    rr_mesh = rr.Points3D(positions=mesh.vertices, colors=mesh.colors)
                else:
                    continue

                rr.log(rr_path, rr_mesh, static=True)
                self._logged_geometry.add(rr_path)

        if self.debug:
            print(f"Logged/updated {len(scene.graph.nodes_geometry)} world nodes in Rerun")

    def animate_plan(
        self,
        ee_positions: np.ndarray,
        object_positions: dict[str, np.ndarray] | None = None,
        timeline: str = "plan",
        point_radius: float = 0.01,
    ) -> None:
        """Animate robot end-effector and (optionally) attached object positions over time using Rerun.

        This helper logs a single 3-D point per timestep so that Rerun can play back the
        trajectory on the provided ``timeline``.  It is intentionally lightweight and does
        not attempt to render the full robot geometry—only key points—keeping the data
        transfer to the viewer minimal.

        Args:
            ee_positions: Array of shape (T, 3) with end-effector world positions.
            object_positions: Mapping from object name to an array (T, 3) with that
                object's world positions.  Each trajectory must be at least as long as
                ``ee_positions``; extra entries are ignored.
            timeline: Name of the Rerun timeline used for the animation frames.
            point_radius: Visual radius (in metres) of the rendered points.
        """
        if ee_positions is None or len(ee_positions) == 0:
            return

        # Iterate over timesteps and log a frame on the chosen timeline
        for idx, pos in enumerate(ee_positions):
            # Set time on the chosen timeline (creates it if needed)
            rr.set_time(timeline, sequence=idx)

            # Log end-effector marker (needs offset for multi-env)
            rr.log(
                "world/anim/ee",
                rr.Points3D(
                    positions=np.array([pos + self._base_translation]),
                    colors=[[0, 100, 255]],  # Blue
                    radii=[point_radius],
                ),
            )

            # Optionally log attached object markers
            # NOTE: Object positions are already in world coordinates, no offset needed
            if object_positions is not None:
                for name, traj in object_positions.items():
                    if idx >= len(traj):
                        continue
                    rr.log(
                        f"world/anim/{name}",
                        rr.Points3D(
                            positions=np.array([traj[idx]]),
                            colors=[[255, 128, 0]],  # Orange
                            radii=[point_radius],
                        ),
                    )

    def animate_spheres_along_path(
        self,
        plan: JointState,
        robot_spheres_at_start: list[Any] | None = None,
        attached_spheres_at_start: list[Any] | None = None,
        timeline: str = "sphere_animation",
        interpolation_steps: int = 10,
    ) -> None:
        """Animate robot and attached object spheres along the planned trajectory with smooth interpolation.

        This method creates a dense, interpolated trajectory and computes sphere positions
        at many intermediate points to create smooth animation of the robot moving along the path.

        Args:
            plan: Joint state trajectory to animate spheres along
            robot_spheres_at_start: Initial robot collision spheres (for reference)
            attached_spheres_at_start: Initial attached object spheres (for reference)
            timeline: Name of the Rerun timeline for the animation
            interpolation_steps: Number of interpolated steps between each waypoint pair
        """
        if plan is None or len(plan.position) == 0:
            if self.debug:
                print("No plan available for sphere animation")
            return

        if self.debug:
            robot_count = len(robot_spheres_at_start) if robot_spheres_at_start else 0
            attached_count = len(attached_spheres_at_start) if attached_spheres_at_start else 0
            print(f"Creating smooth animation for {robot_count} robot spheres and {attached_count} attached spheres")
            print(
                f"Original plan: {len(plan.position)} waypoints, interpolating with {interpolation_steps} steps between"
                " waypoints"
            )

        # We need access to the motion generator to compute spheres at each waypoint
        if not hasattr(self, "_motion_gen_ref") or self._motion_gen_ref is None:
            if self.debug:
                print("Motion generator reference not available for sphere animation")
            return

        motion_gen = self._motion_gen_ref

        # Validate motion generator has required attributes
        if not hasattr(motion_gen, "kinematics") or motion_gen.kinematics is None:
            if self.debug:
                print("Motion generator kinematics not available for sphere animation")
            return

        # Clear static spheres to avoid visual clutter during animation
        self._hide_static_spheres_for_animation()

        # Count robot link spheres (excluding attached objects) for consistent splitting
        robot_link_count = 0
        if robot_spheres_at_start:
            robot_link_count = len(robot_spheres_at_start)

        # Create interpolated trajectory for smooth animation
        interpolated_positions = self._create_interpolated_trajectory(plan, interpolation_steps)

        if self.debug:
            print(f"Created interpolated trajectory with {len(interpolated_positions)} total frames")

        # Animate spheres along the interpolated trajectory
        for frame_idx, joint_positions in enumerate(interpolated_positions):
            # Set time on the animation timeline with consistent timing
            rr.set_time(timeline, sequence=frame_idx)

            # Create joint state for this interpolated position
            if isinstance(joint_positions, torch.Tensor):
                sphere_position = joint_positions
            else:
                sphere_position = torch.tensor(joint_positions)

            # Ensure tensor is on the right device for CuRobo
            if hasattr(motion_gen, "tensor_args") and motion_gen.tensor_args is not None:
                sphere_position = motion_gen.tensor_args.to_device(sphere_position)

            # Get spheres at this configuration
            try:
                sphere_list = motion_gen.kinematics.get_robot_as_spheres(sphere_position)[0]
            except Exception as e:
                if self.debug:
                    print(f"Failed to compute spheres for frame {frame_idx}: {e}")
                continue

            # Handle sphere_list as either a list or single Sphere object
            if hasattr(sphere_list, "__iter__") and not hasattr(sphere_list, "position"):
                sphere_items = list(sphere_list)
            else:
                sphere_items = [sphere_list]

            # Separate robot and attached object spheres with different colors
            robot_sphere_positions = []
            robot_sphere_radii = []
            attached_sphere_positions = []
            attached_sphere_radii = []

            for i, sphere in enumerate(sphere_items):
                # Convert position to numpy
                pos = (
                    sphere.position.detach().cpu().numpy()
                    if torch.is_tensor(sphere.position)
                    else np.array(sphere.position)
                )
                pos = pos.reshape(-1)
                radius = float(sphere.radius)

                if i < robot_link_count:
                    # Robot sphere - needs base translation offset
                    robot_sphere_positions.append(pos + self._base_translation)
                    robot_sphere_radii.append(radius)
                else:
                    # Attached object sphere - already in world coordinates
                    attached_sphere_positions.append(pos)
                    attached_sphere_radii.append(radius)

            # Log robot spheres with green color
            if robot_sphere_positions:
                rr.log(
                    "world/robot_animation",
                    rr.Points3D(
                        positions=np.array(robot_sphere_positions),
                        colors=[[0, 255, 100, 220]] * len(robot_sphere_positions),  # Bright green
                        radii=robot_sphere_radii,
                    ),
                )

            # Log attached object spheres with orange color (or clear if no attached objects)
            if attached_sphere_positions:
                rr.log(
                    "world/attached_animation",
                    rr.Points3D(
                        positions=np.array(attached_sphere_positions),
                        colors=[[255, 150, 0, 220]] * len(attached_sphere_positions),  # Bright orange
                        radii=attached_sphere_radii,
                    ),
                )
            else:
                # Clear attached object spheres when no objects are attached
                rr.log("world/attached_animation", rr.Clear(recursive=True))

        if self.debug:
            print(
                f"Completed smooth sphere animation with {len(interpolated_positions)} frames on timeline '{timeline}'"
            )

    def _hide_static_spheres_for_animation(self) -> None:
        """Hide static sphere visualization during animation to reduce visual clutter."""
        # Clear static robot spheres
        for entity_id in self._sphere_entities.get("robot", []):
            rr.log(f"world/robot/{entity_id}", rr.Clear(recursive=True))

        # Clear static attached spheres
        for entity_id in self._sphere_entities.get("attached", []):
            rr.log(f"world/attached/{entity_id}", rr.Clear(recursive=True))

        if self.debug:
            print("Hidden static spheres for cleaner animation view")

    def _create_interpolated_trajectory(self, plan: JointState, interpolation_steps: int) -> list[torch.Tensor]:
        """Create a smooth interpolated trajectory between waypoints.

        Args:
            plan: Original joint state trajectory
            interpolation_steps: Number of interpolation steps between each waypoint pair

        Returns:
            List of interpolated joint positions
        """
        if len(plan.position) < 2:
            # If only one waypoint, just return it
            return [plan.position[0] if isinstance(plan.position[0], torch.Tensor) else torch.tensor(plan.position[0])]  # type: ignore

        interpolated_positions = []

        # Convert plan positions to tensors if needed
        waypoints = []
        for i in range(len(plan.position)):
            pos = plan.position[i]
            if isinstance(pos, torch.Tensor):
                waypoints.append(pos)
            else:
                waypoints.append(torch.tensor(pos))

        # Interpolate between each pair of consecutive waypoints
        for i in range(len(waypoints) - 1):
            start_pos = waypoints[i]
            end_pos = waypoints[i + 1]

            # Create interpolation steps between start and end
            for step in range(interpolation_steps):
                alpha = step / interpolation_steps
                interpolated_pos = start_pos * (1 - alpha) + end_pos * alpha
                interpolated_positions.append(interpolated_pos)

        # Add the final waypoint
        interpolated_positions.append(waypoints[-1])

        return interpolated_positions

    def set_motion_generator_reference(self, motion_gen: Any) -> None:
        """Set the motion generator reference for sphere animation.

        Args:
            motion_gen: CuRobo motion generator instance
        """
        self._motion_gen_ref = motion_gen

    def mark_idle(self) -> None:
        """Signal that the planner is idle, clearing animations.

        This method advances the animation timelines and logs empty data to ensure that
        no leftover visualizations from the previous plan are shown. It's useful for
        creating a clean state between planning episodes.
        """
        # Advance plan timeline and emit empty anim so latest frame is blank
        rr.set_time("plan", sequence=self._current_frame)
        self._current_frame += 1
        empty = np.empty((0, 3), dtype=float)
        rr.log("world/anim/ee", rr.Points3D(positions=empty))
        rr.log("world/robot_animation", rr.Points3D(positions=empty))
        rr.log("world/attached_animation", rr.Points3D(positions=empty))

        # Also advance sphere animation timeline
        rr.set_time("sphere_animation", sequence=self._current_frame)
        rr.log("world/robot_animation", rr.Points3D(positions=empty))
        rr.log("world/attached_animation", rr.Points3D(positions=empty))
