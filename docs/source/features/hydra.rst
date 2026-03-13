Hydra Configuration System
==========================

.. currentmodule:: isaaclab

Isaac Lab supports the `Hydra <https://hydra.cc/docs/intro/>`_ configuration system to modify the task's
configuration using command line arguments, which can be useful to automate experiments and perform hyperparameter tuning.

Any parameter of the environment can be modified by adding one or multiple elements of the form ``env.a.b.param1=value``
to the command line input, where ``a.b.param1`` reflects the parameter's hierarchy, for example ``env.actions.joint_effort.scale=10.0``.
Similarly, the agent's parameters can be modified by using the ``agent`` prefix, for example ``agent.seed=2024``.

The way these command line arguments are set follow the exact structure of the configuration files. Since the different
RL frameworks use different conventions, there might be differences in the way the parameters are set. For example,
with *rl_games* the seed will be set with ``agent.params.seed``, while with *rsl_rl*, *skrl* and *sb3* it will be set with
``agent.seed``.

As a result, training with hydra arguments can be run with the following syntax:

.. tab-set::
    :sync-group: rl-train

    .. tab-item:: rsl_rl
        :sync: rsl_rl

        .. code-block:: shell

            python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-v0 --headless env.actions.joint_effort.scale=10.0 agent.seed=2024

    .. tab-item:: rl_games
        :sync: rl_games

        .. code-block:: shell

            python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-v0 --headless env.actions.joint_effort.scale=10.0 agent.params.seed=2024

    .. tab-item:: skrl
        :sync: skrl

        .. code-block:: shell

            python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Cartpole-v0 --headless env.actions.joint_effort.scale=10.0 agent.seed=2024

    .. tab-item:: sb3
        :sync: sb3

        .. code-block:: shell

            python scripts/reinforcement_learning/sb3/train.py --task=Isaac-Cartpole-v0 --headless env.actions.joint_effort.scale=10.0 agent.seed=2024

The above command will run the training script with the task ``Isaac-Cartpole-v0`` in headless mode, and set the
``env.actions.joint_effort.scale`` parameter to 10.0 and the ``agent.seed`` parameter to 2024.

.. note::

    To keep backwards compatibility, and to provide a more user-friendly experience, we have kept the old cli arguments
    of the form ``--param``, for example ``--num_envs``, ``--seed``, ``--max_iterations``. These arguments have precedence
    over the hydra arguments, and will overwrite the values set by the hydra arguments.


Modifying advanced parameters
-----------------------------

Callables
^^^^^^^^^

It is possible to modify functions and classes in the configuration files by using the syntax ``module:attribute_name``.
For example, in the Cartpole environment:

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py
    :language: python
    :start-at: class ObservationsCfg
    :end-at: policy: PolicyCfg = PolicyCfg()
    :emphasize-lines: 9

we could modify ``joint_pos_rel`` to compute absolute positions instead of relative positions with
``env.observations.policy.joint_pos_rel.func=isaaclab.envs.mdp:joint_pos``.

Setting parameters to None
^^^^^^^^^^^^^^^^^^^^^^^^^^

To set parameters to None, use the ``null`` keyword, which is a special keyword in Hydra that is automatically converted to None.
In the above example, we could also disable the ``joint_pos_rel`` observation by setting it to None with
``env.observations.policy.joint_pos_rel=null``.

Dictionaries
^^^^^^^^^^^^
Elements in dictionaries are handled as a parameters in the hierarchy. For example, in the Cartpole environment:

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py
    :language: python
    :lines: 90-114
    :emphasize-lines: 11

the ``position_range`` parameter can be modified with ``env.events.reset_cart_position.params.position_range="[-2.0, 2.0]"``.
This example shows two noteworthy points:

- The parameter we set has a space, so it must be enclosed in quotes.
- The parameter is a list while it is a tuple in the config. This is due to the fact that Hydra does not support tuples.


Modifying inter-dependent parameters
------------------------------------

Particular care should be taken when modifying the parameters using command line arguments. Some of the configurations
perform intermediate computations based on other parameters. These computations will not be updated when the parameters
are modified.

For example, for the configuration of the Cartpole camera depth environment:

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_camera_env_cfg.py
    :language: python
    :start-at: class CartpoleDepthCameraEnvCfg
    :end-at: tiled_camera.width
    :emphasize-lines: 10, 15

If the user were to modify the width of the camera, i.e. ``env.tiled_camera.width=128``, then the parameter
``env.observation_space=[80,128,1]`` must be updated and given as input as well.

Similarly, the ``__post_init__`` method is not updated with the command line inputs. In the ``LocomotionVelocityRoughEnvCfg``, for example,
the post init update is as follows:

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
    :language: python
    :start-at: class LocomotionVelocityRoughEnvCfg
    :emphasize-lines: 23, 29, 31

Here, when modifying ``env.decimation`` or ``env.sim.dt``, the user needs to give the updated ``env.sim.render_interval``,
``env.scene.height_scanner.update_period``, and ``env.scene.contact_forces.update_period`` as input as well.


Custom Configuration Validation
--------------------------------

Configclass objects can define a ``validate_config()`` method to perform domain-specific
validation after all fields have been resolved. This hook is called automatically after preset
resolution and MISSING-field checks succeed, allowing you to catch invalid parameter
combinations early with clear error messages.

**Defining a validation hook:**

.. code-block:: python

   from isaaclab.utils import configclass

   @configclass
   class MyEnvCfg:
       physics_backend: str = "physx"
       use_multi_asset: bool = False

       def validate_config(self):
           if self.physics_backend == "newton" and self.use_multi_asset:
               raise ValueError(
                   "Newton physics does not support multi-asset spawning."
                   " Use a single-geometry object preset instead."
               )

**When it runs:**

1. All ``MISSING`` fields are checked first — if any remain, ``TypeError`` is raised.
2. Only then is ``validate_config()`` called on the **top-level** config object.
3. The hook should raise ``ValueError`` with a clear message and migration guidance.

**Common validation patterns:**

- Physics backend compatibility (e.g., Newton does not support multi-asset spawning)
- Renderer and camera data type compatibility (e.g., Newton Warp only supports ``rgb`` and ``depth``)
- Feature extractor compatibility with camera configuration


Preset System
-------------

The preset system lets you swap out entire config sections with a single command line argument.
Instead of overriding individual fields, you select a named preset that **completely replaces** the
config section (no field merging).

Presets are defined directly on config classes using a ``presets`` attribute. The system recursively
discovers all presets from nested configs automatically.


Override Order
^^^^^^^^^^^^^^

Overrides are applied in sequence:

1. **Auto-default**: Configs with a ``"default"`` preset auto-apply without CLI args
2. **Global presets**: ``presets=inference,newton`` applies to ALL matching configs
3. **Path presets**: ``env.actions.arm_action=relative_joint_position`` replaces specific section
4. **Scalar overrides**: ``env.sim.dt=0.001`` modifies individual fields


Defining Presets
^^^^^^^^^^^^^^^^

There are four styles for defining presets:

**Style 1: Inheritance** - Default values from base class, presets for alternatives:

.. code-block:: python

    @configclass
    class FrankaArmActionCfg(mdp.JointPositionActionCfg):
        """Franka arm action config with presets for different action types."""

        presets = {
            "joint_position_to_limit": mdp.JointPositionToLimitsActionCfg(
                asset_name="robot", joint_names=["panda_joint.*"]
            ),
            "relative_joint_position": mdp.RelativeJointPositionActionCfg(
                asset_name="robot", joint_names=["panda_joint.*"], scale=0.2
            ),
        }

**Style 2: Inner class** - Self-contained with nested preset definitions:

.. code-block:: python

    @configclass
    class SimCfg:
        """Simulation config with physics backend presets."""

        backend: str = "physx"
        dt: float = 0.005
        substeps: int = 2

        @configclass
        class Newton:
            backend: str = "newton"
            dt: float = 0.002
            substeps: int = 4
            solver_iterations: int = 8

        presets = {"newton": Newton()}

**Style 3: Preset-only with auto-default** - Pure composition, no default fields:

.. code-block:: python

    @configclass
    class ObservationsCfg:
        """Observation specifications with presets."""

        presets = {
            "default": DefaultObservationsCfg(),
            "noise_less": NoiselessObservationsCfg(),
        }

With Style 3, the ``"default"`` preset is automatically applied when no preset is selected.

**Style 4: PresetCfg class** - Declarative, class-based preset definitions:

.. code-block:: python

    from isaaclab_tasks.utils import PresetCfg

    @configclass
    class PhysicsCfg(PresetCfg):
        """Physics backend presets using class-based pattern."""

        default: PhysxCfg = PhysxCfg()
        newton: NewtonCfg = NewtonCfg()

    @configclass
    class SimCfg:
        physics: PhysicsCfg = PhysicsCfg()

With Style 4, each field on the ``PresetCfg`` subclass is a named preset. The ``default`` field
holds the config instance used when no CLI override is given. ``collect_presets`` automatically
discovers ``PresetCfg`` subclasses and converts their fields into a presets dict, so no
``presets`` attribute is needed. CLI usage is the same as other styles:

.. code-block:: bash

    # Use Newton physics backend
    python train.py --task=Isaac-Reach-Franka-v0 \
        env.sim.physics=newton

The ``default`` field can be set to ``None`` to make an optional feature that is disabled unless
explicitly selected on the command line:

.. code-block:: python

    @configclass
    class CameraPresetCfg(PresetCfg):
        default = None
        small: CameraCfg = CameraCfg(width=64, height=64)
        large: CameraCfg = CameraCfg(width=256, height=256)

    @configclass
    class SceneCfg:
        camera: CameraPresetCfg = CameraPresetCfg()

When no CLI argument is given, ``camera`` resolves to ``None`` (no camera):

.. code-block:: bash

    # camera is None — no camera overhead
    python train.py --task=Isaac-Reach-Franka-v0

    # activate camera with the "large" preset
    python train.py --task=Isaac-Reach-Franka-v0 \
        env.scene.camera=large


Using Presets
^^^^^^^^^^^^^

**Path presets** - Select a specific preset for one config path:

.. code-block:: bash

    # Use relative joint position action
    python train.py --task=Isaac-Reach-Franka-v0 \
        env.actions.arm_action=relative_joint_position

    # Use noiseless observations
    python train.py --task=Isaac-Reach-Franka-v0 \
        env.observations=noise_less

**Path preset + scalar override** - Select preset then modify a field:

.. code-block:: bash

    python train.py --task=Isaac-Reach-Franka-v0 \
        env.actions.arm_action=relative_joint_position \
        env.actions.arm_action.scale=0.5

**Global presets** - Apply the same preset name everywhere it exists:

.. code-block:: bash

    # Apply "inference" preset to all configs that define it
    # (e.g., observations, policy, etc.)
    python train.py --task=Isaac-Reach-Franka-v0 \
        presets=inference

**Multiple global presets** - Apply several non-conflicting presets:

.. code-block:: bash

    # Newton physics backend + inference mode
    python train.py --task=Isaac-Reach-Franka-v0 \
        presets=newton,inference

**Combined** - Global presets + path presets + scalar overrides:

.. code-block:: bash

    python train.py --task=Isaac-Reach-Franka-v0 \
        presets=inference \
        env.actions.arm_action=relative_joint_position \
        env.actions.arm_action.scale=0.5 \
        env.sim.dt=0.002


Global Preset Conflict Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If multiple global presets define the same path, an error is raised:

.. code-block:: bash

    # ERROR: both "fast" and "noise_less" define env.observations
    python train.py --task=Isaac-Reach-Franka-v0 \
        presets=fast,noise_less

    # ValueError: Conflicting global presets: 'fast' and 'noise_less'
    #             both define preset for 'env.observations'


Real-World Example
^^^^^^^^^^^^^^^^^^

The Franka Reach environment demonstrates presets in practice:

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/franka/joint_pos_env_cfg.py
    :language: python
    :start-at: @configclass
    :end-before: class FrankaReachEnvCfg_PLAY

This allows users to switch action types:

.. code-block:: bash

    # Default: JointPositionActionCfg (from inheritance)
    python train.py --task=Isaac-Reach-Franka-v0

    # Switch to relative joint position
    python train.py --task=Isaac-Reach-Franka-v0 \
        env.actions.arm_action=relative_joint_position

    # Switch to joint position with limits
    python train.py --task=Isaac-Reach-Franka-v0 \
        env.actions.arm_action=joint_position_to_limit


Summary
^^^^^^^

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Override Type
     - Syntax
     - Effect
   * - Scalar
     - ``env.sim.dt=0.001``
     - Modify single field
   * - Path preset
     - ``env.actions.arm_action=relative``
     - Replace entire section
   * - Global preset
     - ``presets=inference``
     - Apply everywhere matching
   * - Combined
     - ``presets=newton env.sim.dt=0.001``
     - Global + scalar overrides
