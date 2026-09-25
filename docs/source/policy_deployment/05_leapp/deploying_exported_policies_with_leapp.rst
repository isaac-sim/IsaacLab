Deploy Exported Policies with LEAPP
===================================

.. currentmodule:: isaaclab

Isaac Lab provides :class:`~envs.LeappDeploymentEnv` for running exported policies back in
simulation without the training infrastructure. This is the Isaac Lab deployment path for
LEAPP-exported policies and is useful for validating that the packaged policy still behaves
correctly when driven through the deployment stack instead of the training stack.

Run the deployment command with the task name and the exported LEAPP ``.yaml`` file. Use the
same backend extra and backend selector that you used for training and export, and pass a
``--viz`` option when you want a viewport:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         # Newton backend (kitless)
         uv run --extra leapp isaaclab leapp deploy \
             --task <TASK_NAME> \
             --pipeline <PATH_TO_EXPORTED_LEAPP_YAML> \
             --viz newton_gl physics=newton_mjwarp

         # OV PhysX backend
         uv run --extra ovphysx --extra leapp isaaclab leapp deploy \
             --task <TASK_NAME> \
             --pipeline <PATH_TO_EXPORTED_LEAPP_YAML> \
             --viz kit physics=ovphysx

         # Isaac Sim PhysX backend
         OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y uv run --extra isaacsim --extra leapp isaaclab leapp deploy \
             --task <TASK_NAME> \
             --pipeline <PATH_TO_EXPORTED_LEAPP_YAML> \
             --viz kit physics=isaacsim_physx

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         :: Newton backend (kitless)
         uv run --extra leapp isaaclab leapp deploy ^
             --task <TASK_NAME> ^
             --pipeline <PATH_TO_EXPORTED_LEAPP_YAML> ^
             --viz newton_gl physics=newton_mjwarp

         :: OV PhysX backend
         uv run --extra ovphysx --extra leapp isaaclab leapp deploy ^
             --task <TASK_NAME> ^
             --pipeline <PATH_TO_EXPORTED_LEAPP_YAML> ^
             --viz kit physics=ovphysx

         :: Isaac Sim PhysX backend
         set OMNI_KIT_ACCEPT_EULA=Y
         set ACCEPT_EULA=Y
         uv run --extra isaacsim --extra leapp isaaclab leapp deploy ^
             --task <TASK_NAME> ^
             --pipeline <PATH_TO_EXPORTED_LEAPP_YAML> ^
             --viz kit physics=isaacsim_physx

What the Deployment Environment Does
------------------------------------

The deployment environment loads the task scene and then bypasses the training-time
observation, action, reward, termination, and curriculum managers. It reads the scene
objects and commands identified by the ``isaaclab_connection`` metadata in the LEAPP YAML,
uses them as inputs to the exported policy pipeline, and runs that pipeline through
LEAPP's ``InferenceManager``. The resulting outputs are written back to the mapped scene
entities.

Match the Training and Export Configuration
-------------------------------------------

The deployment command rebuilds the task configuration from ``--task``. It cannot infer the
training configuration from the checkpoint or LEAPP YAML, so you must provide the same
configuration selections used for training and export. The task name, LEAPP YAML, and
checkpoint must describe the same policy. The command applies the task's ``play_mode()``
configuration before Hydra overrides, matching the export path and disabling common
training-only behavior such as observation corruption.

Carry these settings through **training**, **export**, and **deployment**:

* **Task and Hydra overrides:** use the same task name and every configuration override that
  changes the scene, robot, sensors, command terms, observation/action term configuration,
  reset behavior, or simulation timing (``sim.dt`` and ``decimation``).
* **Physics and renderer presets:** pass the same backend selectors when you selected a
  non-default backend. For example, a policy trained and exported with
  ``physics=newton_mjwarp`` must also be deployed with ``physics=newton_mjwarp``; the same
  applies to the untyped ``presets=<PRESET_NAME>`` form.
* **Seed, when applicable:** use the same ``--seed`` when reproducible command sampling,
  reset events, or other randomized task behavior matters to your validation.

For example, append the preset after the deployment options:

.. code-block:: bash

   uv run --extra leapp isaaclab leapp deploy \
       --task Isaac-Humanoid \
       --pipeline <PATH_TO_EXPORTED_LEAPP_YAML> \
       --viz newton_gl \
       presets=newton_mjwarp

Visualization Options
^^^^^^^^^^^^^^^^^^^^^

Visualization does not change the policy or its LEAPP inputs. Omit ``--viz`` for headless
execution, use ``--viz kit`` for the Omniverse Kit viewport, or select ``newton_gl``,
``newton_rtx`` (experimental), ``rerun``, or ``viser`` when those visualizers are installed.
Use ``--viz none`` to explicitly disable all visualizers, and ``--max_visible_envs <COUNT>``
to limit displayed environments. Choose a visualizer compatible with the selected physics and
renderer backend. For a bounded headless canary, pass ``--max_steps <COUNT>``.

Controller and Hardware Handoff
-------------------------------

The Isaac Lab deployment command validates every controller-owned write declared in the pipeline.
For simulation, it explicitly supplies the built-in ``gravity_compensation`` capability from the
articulation model at the same cadence as the task's action manager. Unknown capabilities,
unresolved joints, duplicate owners, and overlap with a policy output stop deployment before the
policy runs.

On hardware, the robot runtime must explicitly acknowledge every capability listed under
``pipeline.configs.isaaclab.controller_owned_writes``. For example, the Franka stack policy emits
arm position and gripper targets while the FR3 joint-impedance controller owns gravity
compensation. The hardware adapter must therefore verify that gravity compensation is enabled in
that controller and must **not** synthesize or forward an additional model effort output, which
would create two gravity owners.

The ``action_apply`` cadence records where the simulator refreshes its substitute write. It does
not reduce a hardware controller's gravity compensation to the policy rate; the FR3 controller
continues to provide gravity compensation at its normal servo rate.

Use the generated tensor descriptions as the source of truth when wiring an Isaac ROS Deploy
graph. Match each input's semantic kind, dtype, shape, and element-name order; do not duplicate
normalization or layout conversion already captured inside the exported graph. Keep command
limits, freshness/watchdog checks, fault handling, and emergency-stop behavior outside the policy
and validate them before enabling robot motion. See the
`Isaac ROS Deploy documentation <https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_deploy/index.html>`__
for runtime graph integration.

.. note::

   For Direct workflow policies, see the
   :doc:`Direct workflow export guide <exporting_direct_workflow_policies_with_leapp>`.
   Direct workflow policies are not supported by ``LeappDeploymentEnv``.
