.. _concepts_sensors_foveated_camera:

.. currentmodule:: isaaclab_fovi

Foveated Camera
===============

A :class:`~sensors.FoveatedCamera` samples densely at its fixation and sparsely in the periphery.
The `fovi <https://nblauch.github.io/fovi>`__ library places the viewing rays. Moving the gaze rotates them
relative to the physical camera's mount.

.. figure:: ../../_static/overview/sensors/foveated_camera_overview.jpg
   :alt: The foveated ray grid and its published images at two fixations.

   Each pair shows the retinal rays on the viewing sphere and the image at that fixation.

Sampling
--------

The cortical magnification at angular distance :math:`r` from fixation is :math:`M(r) = 1/(r+a)`.
On the default warped Cartesian grid, a lattice point at normalized radius :math:`\rho` samples the eccentricity

.. math::

   r(\rho) = a\left[\left(1 + h/a\right)^\rho - 1\right],

where :math:`h` is half the retinal field of view. Smaller :attr:`~sensors.FoveatedCameraCfg.cmf_a_deg`
concentrates more samples near fixation. A large value approaches uniform radial spacing on the same grid.

The default grid publishes an upright ``(num_envs, N, N, channels)`` image with
``N = foveated_res``. Its circular footprint zeros the unused corners. An isotropic retina places samples
on rings and publishes ``(num_envs, num_samples, 1, channels)``; sample sets are experimental on RTX.
The camera options describe alternate footprints and sampling shells.

.. raw:: html

   <div class="fovi-viewer" data-fovi-sampling-viewer></div>

Camera and gaze
---------------

Run the demo with the ``fovi`` extra.

.. code-block:: bash

   uv run --extra fovi isaaclab demo foveated-camera

Add :class:`~sensors.FoveatedCameraCfg` to a scene like any other camera. Newton Warp and OptiX render the
retinal rays directly. Isaac RTX and OVRTX use lens lookup textures.

The retinal field of view defaults to the angle across the short side of the physical image. The camera
uses ideal pinhole optics or supported OpenCV distortion from ``spawn``. Supply
:attr:`~sensors.FoveatedCameraCfg.camera_model` for explicit calibration or an existing camera prim.
Use :attr:`~sensors.FoveatedCameraCfg.retinal_window_fraction` to shrink this window and
:attr:`~sensors.FoveatedCameraCfg.image_rotation_deg` to compensate for mount roll.

Call ``camera.set_gaze_angles(angles)`` with shape ``(num_envs, 2)`` in tilt, pan order and in degrees.
Positive tilt looks up and positive pan looks left. :meth:`~sensors.FoveatedCamera.get_world_poses`
reports the mount pose. Enable :attr:`~sensors.FoveatedCameraCfg.mask_to_sensor_fov` to zero rays that fall
outside the physical image as the gaze moves.

Pooling
-------

Enable :attr:`~sensors.FoveatedCameraCfg.pool_enabled` to render a denser set of rays and pool their values
into the published retina. Grid styles pool windows, while sample sets pool nearest neighbours.
The sensor mask excludes rays outside the physical image from the reduction. Segmentation selects the
modal label in its original dtype.

.. raw:: html

   <div class="fovi-viewer" data-fovi-pooling-viewer data-fov="90" data-cmf-a="2" data-res="16"></div>

Active gaze Cartpole
--------------------

``IsaacContrib-Cartpole-Camera-Fovi`` uses a 32 × 32 retina and two bounded gaze actions. A pole-tip teacher
supervises the gaze-action means during training. The gaze dimensions use fixed exploration noise.
Select ``presets=depth`` for depth images.

.. code-block:: bash

   uv run --extra fovi --extra rsl-rl isaaclab train --rl_library rsl_rl --task IsaacContrib-Cartpole-Camera-Fovi

For other manager-based tasks, use :class:`~mdp.GazeActionCfg` to move gaze, :func:`~mdp.gaze_state` to
observe it, and :class:`~mdp.gaze_teacher_labels` for optional supervision.
