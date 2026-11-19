.. _concepts_sensors_foveated_camera:

.. currentmodule:: isaaclab_fovi

Foveated Camera
===============

A :class:`~sensors.FoveatedCamera` samples densely at its fixation and sparsely in the periphery.
The `fovi <https://nblauch.github.io/fovi>`__ library maps each cell of an upright square grid to a viewing ray.
Moving the gaze rotates these rays relative to the physical camera's mount.

.. figure:: ../../_static/overview/sensors/foveated_camera_overview.jpg
   :alt: The foveated ray grid and the published images at two fixations.

   Each pair shows the retinal rays on the viewing sphere and the image published at that fixation.

Sampling
--------

At an angular distance :math:`r` from fixation, the cortical magnification is :math:`M(r)=1/(r+a)`.
For a normalized grid radius :math:`\rho` and retinal half-field angle :math:`h`, the ray's angular distance is

.. math::

   r(\rho) = a\left[\left(1 + h/a\right)^\rho - 1\right].

Smaller :attr:`~sensors.FoveatedCameraCfg.cmf_a_deg` concentrates more samples near fixation. A large value gives
nearly uniform radial angular spacing on the same grid. For a grid of side
:attr:`~sensors.FoveatedCameraCfg.foveated_res`, the camera publishes ``(num_envs, N, N, channels)``.
Cells outside the circular retinal footprint are zero.

Camera setup and gaze
---------------------

Install the ``fovi`` extra to run the camera demo.

.. code-block:: bash

   uv run --extra fovi isaaclab demo foveated-camera

Add a :class:`~sensors.FoveatedCameraCfg` to a scene like any camera configuration. Newton Warp and OptiX render
the sample rays directly. Isaac RTX and OVRTX use lens lookup textures.

The retinal field of view defaults to the angle spanned by the short side of the physical camera's image.
The camera reads ideal pinhole optics or supported OpenCV distortion from ``spawn``. Supply
:attr:`~sensors.FoveatedCameraCfg.camera_model` for explicit calibration, including when using an existing
camera prim. The :attr:`~sensors.FoveatedCameraCfg.retinal_window_fraction` setting shrinks the retinal window.

Call ``camera.set_gaze_angles(angles)`` with an array shaped ``(num_envs, 2)`` in tilt, pan order and in degrees.
Positive tilt looks up and positive pan looks left. Enable :attr:`~sensors.FoveatedCameraCfg.mask_to_sensor_fov`
to zero rays that land outside the physical image at the current gaze. This mask moves relative to the retinal grid
as the gaze changes.

Cartpole example
----------------

``IsaacContrib-Cartpole-Camera-Fovi`` uses a 32 × 32 retina and two bounded gaze actions. The actor observes the camera
image and the gaze state. Use ``presets=depth`` to select depth instead of RGB.

.. code-block:: bash

   uv run --extra fovi --extra rsl-rl isaaclab train --rl_library rsl_rl --task IsaacContrib-Cartpole-Camera-Fovi

For other manager-based tasks, use :class:`~mdp.GazeActionCfg` to add the tilt and pan actions and
:func:`~mdp.gaze_state` to observe the angles and angular velocity in units of the retinal field radius.
