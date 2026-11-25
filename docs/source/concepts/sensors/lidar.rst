.. _concepts_sensors_lidar:

OvRTX LiDAR
===========

``OVRTXLidar`` is an Isaac Lab :class:`~isaaclab.sensors.SensorBase` that measures a point cloud.
Core defines the common ``LidarCfg`` and ``LidarData`` types. The concrete ``OVRTXLidarCfg``
selects the sensor implementation and describes the firing pattern for the OvRTX renderer.

Each cloned environment gets an ``OmniLidar`` prim by default. At capture time the sensor composes
its offset with its parent's world pose. OvRTX renders the prims for that sensor as tiles of
one render product and returns a point cloud with one row of returns per environment.

Configure a scan
----------------

Add an ``OVRTXLidarCfg`` field to an interactive scene configuration. This example places a
sensor three meters above each environment origin and fires two rays per scan.

.. code-block:: python

   from isaaclab_ov.renderers import OVRTXLidarCfg

   from isaaclab.scene import InteractiveSceneCfg
   from isaaclab.utils import configclass


   @configclass
   class SceneCfg(InteractiveSceneCfg):
       lidar = OVRTXLidarCfg(
           prim_path="{ENV_REGEX_NS}/Lidar",
           offset=OVRTXLidarCfg.OffsetCfg(pos=(0.0, 0.0, 3.0)),
           scan_type="solidState",
           azimuth_deg=[0.0, 1.0],
           elevation_deg=[0.0, 0.0],
           fire_time_ns=[0, 0],
           channel_id=[0, 1],
           max_returns=1,
       )

Each index in the four firing-pattern lists describes one ray. Supply equally long,
nonempty lists of azimuths, elevations, firing times within the tick, and channel IDs.
A solid-state scan fires the pattern once. The default rotary scan repeats it across the
azimuth window in steps of ``rot_pattern_firing_res_deg``. Use ``cfg.num_rays()`` to get the
number of output slots per environment, including multiple returns and any tiled subsampling.

The default renderer configuration sets ``motion_bvh="auto"``. If you supply an explicit
:class:`~isaaclab_ov.renderers.OVRTXRendererCfg`, set ``motion_bvh`` to ``"auto"`` or ``"enable"``.

Read the point cloud
--------------------

In a standalone simulation loop, advance physics and the scene's sensor clocks before reading
the measurement. Reading ``data`` captures the scan when it is due. The default
``update_period=0.0`` samples every simulation step.

.. code-block:: python

   sim.step()
   scene.update(sim.get_physics_dt())

   data = scene["lidar"].data
   points = data.points.torch        # (num_envs, num_slots, 3)
   intensity = data.intensity.torch  # (num_envs, num_slots)
   valid = data.valid.torch          # (num_envs, num_slots)

   # Select hits from environment zero, keeping points and intensities aligned.
   xyz = points[0, valid[0]]
   signal = intensity[0, valid[0]]

Points are Cartesian XYZ coordinates in meters, including when OvRTX supplies spherical
returns (``elements_coords_type="SPHERICAL"``). The default frame is the sensor frame. Set
``output_frame_of_reference="WORLD"`` for world coordinates. Tiled capture requires
``include_invalid_points=True`` (the default). A missed return keeps its slot,
has non-finite coordinates and zero intensity, and has ``valid=False``. The buffers live on
the simulation device. Use ``.warp`` for a Warp view, or clone the Torch view to retain a scan
across subsequent updates.

The scene key selects the sensor and the first tensor index selects the environment. To use
front and rear sensors on one robot, declare two fields with distinct prim paths and read
``scene["front_lidar"].data`` and ``scene["rear_lidar"].data`` independently.
Calling ``scene["front_lidar"].reset(env_ids=[0])`` clears that sensor's returns for
environment zero and preserves the other environments' returns.

Run the Cartpole lidar example
------------------------------

The experimental ``IsaacContrib-Cartpole-Lidar`` and ``IsaacContrib-Cartpole-Lidar-Direct`` tasks
place a side-mounted OvRTX lidar aimed at the cart and pole. The policy receives two consecutive
range images of shape ``(49, 161)``. Each pixel holds the sensor-frame hit distance divided by
12 meters, with zero for a miss or an unsampled ray. The critic receives the four joint-state values.

Run the manager-based task for 32 steps on a CUDA device:

.. code-block:: bash

   uv run --extra ovrtx isaaclab random_agent --task IsaacContrib-Cartpole-Lidar --num_envs 4 --max_steps 32

Use ``--task IsaacContrib-Cartpole-Lidar-Direct`` to run the direct workflow.
