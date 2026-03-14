isaaclab\_contrib.rl
====================

.. automodule:: isaaclab_contrib.rl

Submodules
----------

isaaclab\_contrib.rl.rlinf
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab_contrib.rl.rlinf

.. note::

    The extension module requires the external ``rlinf`` package and cannot be
    introspected at documentation build time. The API is described textually below.

Extension Module
^^^^^^^^^^^^^^^^

The extension module (``isaaclab_contrib.rl.rlinf.extension``) is loaded by RLinf's
worker framework via the ``RLINF_EXT_MODULE`` environment variable. It is not imported
directly by user code.

**Setup:**

.. code-block:: bash

    export RLINF_EXT_MODULE="isaaclab_contrib.rl.rlinf.extension"
    export RLINF_CONFIG_FILE="/path/to/config.yaml"

**Public entry point:**

- ``register()`` -- Called by RLinf's worker to perform all setup. It:

  1. Registers GR00T observation and action converters.
  2. Patches GR00T's ``get_model`` for custom embodiment tags.
  3. Registers IsaacLab tasks into RLinf's ``REGISTER_ISAACLAB_ENVS`` registry.

**Expected YAML configuration** (under ``env.train.isaaclab``):

.. code-block:: yaml

    env:
      train:
        isaaclab: &isaaclab_config
          task_description: "Assemble trocar with dual-arm robot"
          main_images: "front_camera"
          extra_view_images: ["left_wrist_camera", "right_wrist_camera"]
          states:
            - key: "robot_joint_state"
              slice: [15, 29]
          gr00t_mapping:
            video:
              main_images: "video.room_view"
          action_mapping:
            prefix_pad: 15
      eval:
        isaaclab: *isaaclab_config  # Reuse via YAML anchor

Task IDs are read automatically from ``env.train.init_params.id`` and
``env.eval.init_params.id`` in the YAML config.
