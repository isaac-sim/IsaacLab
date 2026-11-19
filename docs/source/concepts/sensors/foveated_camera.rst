.. _concepts_sensors_foveated_camera:

.. currentmodule:: isaaclab_fovi

Foveated Camera
===============

A :class:`~sensors.FoveatedCamera` samples the scene densely where it looks and sparsely in the periphery, as a
retina does, instead of on a uniform pixel grid. It is a :class:`~isaaclab.sensors.Camera` whose rays are laid out
by the `fovi <https://pypi.org/project/fovi/>`__ library, and its gaze moves inside the field of view of the
physical camera it looks through.

The camera comes with the ``fovi`` extra:

.. code-block:: bash

   uv sync --extra fovi

The demo moves the gaze from object to object while the viewer shows the retina:

.. code-block:: bash

   uv run --extra fovi isaaclab demo foveated-camera

Configure a foveated camera
---------------------------

Add a :class:`~sensors.FoveatedCameraCfg` to a scene like any camera configuration. The renderer selects how the
retina is rendered:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Renderer configuration
     - Rendering
   * - :class:`~isaaclab_newton.renderers.NewtonWarpRendererCfg`
     - One ray per retinal sample.
   * - :class:`~isaaclab_physx.renderers.IsaacRtxRendererCfg`,
       :class:`~isaaclab_ov.renderers.OVRTXRendererCfg`
     - A lens lookup texture on the camera prim maps the image to the retina's rays.

The camera also accepts the Newton OptiX renderer (``optix`` extra), which receives the same rays as Newton Warp.

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab_fovi.sensors import FoveatedCameraCfg
   from isaaclab_newton.renderers import NewtonWarpRendererCfg

   eye = FoveatedCameraCfg(
       prim_path="{ENV_REGEX_NS}/Eye",
       data_types=["rgb"],
       renderer_cfg=NewtonWarpRendererCfg(),
       spawn=sim_utils.PinholeCameraCfg(),
       foveated_res=64,
   )

The camera computes its image size from the retina. With the default grid style it publishes an upright image of
shape ``(num_envs, foveated_res, foveated_res, channels)`` in :attr:`~isaaclab.sensors.Camera.data`, like any
camera, and any output the renderer supports.

The physical camera
~~~~~~~~~~~~~~~~~~~

The retina places its samples by visual angle inside the image of a physical camera, and its rays are cast at those
angles, so no pinhole image is resampled. By default that camera is the one ``spawn`` creates:

* :class:`~isaaclab.sim.PinholeCameraCfg` optics give an ideal pinhole.
* An OpenCV model in ``spawn.distortion`` gives that calibrated camera. An
  :class:`~isaaclab.sim.OpenCvFisheyeDistortionCfg` with zero coefficients is an ideal equidistant fisheye.

For any other calibrated camera, or a camera prim without spawn configuration, set
:attr:`~sensors.FoveatedCameraCfg.camera_model` to the fields of a ``fovi.sensing.projection.CameraModel``:

.. code-block:: python

   camera_model = {
       "model": "pinhole",  # or "fisheye"
       "image_size": (480, 640),  # (height, width)
       "intrinsics": (550.0, 550.0, 319.5, 239.5),  # fx, fy, cx, cy [px]
   }

The same fields configure fovi's software sampler on images from a real camera.

The projection is set in the configuration. :meth:`~sensors.FoveatedCamera.set_intrinsic_matrices` leaves a
foveated camera as configured and warns, as cameras with an OpenCV lens model do.

Foveation strength
~~~~~~~~~~~~~~~~~~

:attr:`~sensors.FoveatedCameraCfg.cmf_a_deg` is the cortical magnification constant in degrees. Smaller values
concentrate more samples at the fixation; the default, 0.591, is an estimate from human data. A value much larger
than the field of view samples nearly uniformly, which gives a matched non-foveated baseline with the same
resolution, field of view and gaze:

.. code-block:: bash

   uv run --extra fovi isaaclab demo foveated-camera env.scene.eye.cmf_a_deg=1000

Move the gaze
-------------

The gaze turns the retina relative to the physical camera's optical axis:

.. code-block:: python

   # (tilt, pan) [deg] from the optical axis, shape (num_envs, 2): positive tilt looks up, positive pan looks left.
   camera.set_gaze_angles(angles)

The gaze stays relative to the camera's mount when the mount moves, and
:meth:`~sensors.FoveatedCamera.get_world_poses` reports the mount without the gaze. Set
:attr:`~sensors.FoveatedCameraCfg.mask_to_sensor_fov` to zero the samples that fall outside the physical camera's
image at the current gaze.

Pooling
-------

With :attr:`~sensors.FoveatedCameraCfg.pool_enabled`, the camera renders a denser photoreceptor grid and pools it to
the retina, ``pool_res_mult`` times denser, with ``pool_mode`` (``avg``, ``gaussian``, ``max`` or ``sum``).
Segmentation outputs pool by modal label in their own dtype (``pool_int_mode="modal"``), since averaging would mix
labels into ones no object has. The sensor mask applies to the photoreceptors before pooling, so a pooled sample at
the image edge keeps what the camera sees of its neighbourhood.

Further sampling options are passed to fovi's ``SamplingCoords``, whose
`documentation <https://nblauch.github.io/fovi/docs/>`__ describes them: ``fov_type`` and ``radius_norm`` set the
region the retina covers, and sample-set styles such as ``foveated_style="isotropic"`` publish
``(num_envs, num_samples, 1, channels)``. Sample-set styles are experimental on the RTX renderers, where crowded
samples can share a lens-texture texel.

Active gaze
-----------

:mod:`isaaclab_fovi.mdp` holds the manager terms for a policy that moves the gaze:

* :class:`~mdp.GazeActionCfg`: two actions that move the gaze by up to ``step_size`` field radii per step, within
  ``gaze_limit`` of the optical axis. Make it the last action term. With the default ``mount="fixed"`` the teacher
  aims from the camera's orientation when the action is created; ``mount="tracked"`` lets the camera ride a moving
  body, such as a turning head (Newton and OVRTX).
* :func:`~mdp.gaze_state`: the gaze and its velocity.
* :class:`~mdp.gaze_teacher_labels`: a teacher's gaze action toward a point fixed to an asset, and whether that point
  is a valid target. It is a training label, never a policy input.

The ``IsaacContrib-Cartpole-Camera-Fovi`` task in :mod:`isaaclab_tasks.contrib.active_gaze` puts them together. It is
the stock Cartpole camera task with the camera foveated (a 32-sample retina), two gaze actions, and a teacher that looks
at the pole's tip. Its PPO adds a loss that pulls the gaze actions toward the teacher's, and its action head keeps
the gaze actions' noise fixed:

.. code-block:: bash

   uv run --extra fovi isaaclab train --rl_library rsl_rl --task IsaacContrib-Cartpole-Camera-Fovi

The task is registered when the ``fovi`` extra is installed.
