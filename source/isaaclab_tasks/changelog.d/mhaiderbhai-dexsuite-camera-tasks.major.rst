Changed
^^^^^^^

* **Breaking:** Split the KukaAllegro dexsuite camera (vision) configuration out of the state task
  into dedicated ``-Camera`` environments, matching the cartpole layout. The base state tasks
  (``Isaac-Reorient-KukaAllegro``, ``Isaac-Lift-KukaAllegro``) no longer accept the
  ``presets=single_camera`` / ``presets=duo_camera`` selectors; use the new camera tasks instead:

  * ``Isaac-Lift-KukaAllegro`` + ``presets=single_camera`` → ``Isaac-Lift-KukaAllegro-Camera``.
  * ``Isaac-Reorient-KukaAllegro`` + ``presets=single_camera`` → ``Isaac-Reorient-KukaAllegro-Camera``.

  The ``single_camera`` / ``duo_camera`` (and the camera data-type / renderer-backend) selectors now
  live on the ``-Camera`` tasks. The camera configs were moved to a dedicated
  ``dexsuite_kuka_allegro_camera_env_cfg`` module whose env configs inherit from the state base.

Added
^^^^^

* Added the ``Isaac-Reorient-KukaAllegro-Camera`` and ``Isaac-Lift-KukaAllegro-Camera`` vision
  environments.
