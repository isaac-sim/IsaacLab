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

The preset system lets you swap out entire config sections -- or individual scalar
values -- with a single command line argument. Instead of overriding individual
fields, you select a named preset that **completely replaces** the config section
(no field merging).

Presets are declared by subclassing :class:`~isaaclab_tasks.utils.hydra.PresetCfg`
or by using the :func:`~isaaclab_tasks.utils.hydra.preset` convenience factory. The
system recursively discovers all presets from nested configs automatically,
including presets inside dict-valued fields (e.g. ``actuators``).


Override Order
^^^^^^^^^^^^^^

Overrides are applied in sequence:

1. **Auto-default**: Configs with a ``"default"`` field auto-apply without CLI args
2. **Global presets**: ``presets=newton,inference`` applies to ALL matching configs
3. **Path presets**: ``env.backend=newton`` replaces a specific section
4. **Scalar overrides**: ``env.sim.dt=0.001`` modifies individual fields


Defining Presets with PresetCfg
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a :class:`~isaaclab_tasks.utils.hydra.PresetCfg` subclass where each field
is a named alternative. The ``default`` field is the config used when no CLI
override is given:

.. code-block:: python

    from isaaclab_tasks.utils import PresetCfg

    @configclass
    class PhysicsCfg(PresetCfg):
        default: PhysxCfg = PhysxCfg()
        newton: NewtonCfg = NewtonCfg()

    @configclass
    class MyEnvCfg:
        physics: PhysicsCfg = PhysicsCfg()

.. code-block:: bash

    # Use Newton physics backend
    python train.py --task=Isaac-Reach-Franka-v0 env.physics=newton

The ``default`` field can be set to ``None`` to make an optional feature that is
disabled unless explicitly selected:

.. code-block:: python

    @configclass
    class CameraPresetCfg(PresetCfg):
        default = None
        small: CameraCfg = CameraCfg(width=64, height=64)
        large: CameraCfg = CameraCfg(width=256, height=256)

    @configclass
    class SceneCfg:
        camera: CameraPresetCfg = CameraPresetCfg()

.. code-block:: bash

    # camera is None -- no camera overhead
    python train.py --task=Isaac-Reach-Franka-v0

    # activate camera with the "large" preset
    python train.py --task=Isaac-Reach-Franka-v0 env.scene.camera=large


Inline Presets with preset()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For simple values (scalars, lists) that don't warrant a full subclass, use the
:func:`~isaaclab_tasks.utils.hydra.preset` factory. It dynamically creates a
``PresetCfg`` instance from keyword arguments:

.. code-block:: python

    from isaaclab_tasks.utils.hydra import preset

    # Scalar preset -- one line, no boilerplate class
    self.scene.robot.actuators["legs"].armature = preset(default=0.0, newton=0.01, physx=0.0)

This is equivalent to defining a ``PresetCfg`` subclass with three ``float``
fields, but without the ceremony. The ``default`` keyword is required.

``preset()`` works for any value type -- scalars, lists, or even config
instances:

.. code-block:: python

    # Resolution preset on a camera config field
    width = preset(default=64, res128=128, res256=256)

    # List preset for camera data types
    @configclass
    class DataTypeCfg(PresetCfg):
        default: list = ["rgb"]
        depth: list = ["depth"]
        albedo: list = ["albedo"]

Use ``preset()`` when the definition fits on a single line.  Use a
``PresetCfg`` subclass when the options are verbose enough to benefit from
type annotations and multiline formatting.

The preset system discovers ``preset()`` values anywhere in the config tree,
including inside dict-valued fields such as ``actuators``:

.. code-block:: bash

    # Select newton preset globally -- sets armature to 0.01
    python train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 presets=newton


Using Presets
^^^^^^^^^^^^^

**Path presets** -- select a specific preset for one config path:

.. code-block:: bash

    python train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 \
        env.events=newton

**Global presets** -- apply the same preset name everywhere it exists:

.. code-block:: bash

    # Apply "newton" preset to all configs that define it
    python train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 \
        presets=newton

**Multiple global presets** -- apply several non-conflicting presets:

.. code-block:: bash

    python train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 \
        presets=newton,inference

**Combined** -- global presets + scalar overrides:

.. code-block:: bash

    python train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 \
        presets=newton \
        env.sim.dt=0.002


Global Preset Conflict Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If two global presets both match the same config path, an error is raised
so the ambiguity is caught early:

.. code-block:: text

    ValueError: Conflicting global presets: 'foo' and 'bar'
                both define preset for 'env.events'


Real-World Example
^^^^^^^^^^^^^^^^^^

The ANYmal-C locomotion environment shows both ``PresetCfg`` and ``preset()``
working together:

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/rough_env_cfg.py
    :language: python
    :lines: 21-42

A single ``presets=newton`` on the command line resolves every ``PresetCfg``
and ``preset()`` that defines a ``newton`` field: the physics engine is swapped
to Newton, ``AnymalCEventsCfg`` selects Newton-compatible events, and the
actuator armature is set to ``0.01``.

.. code-block:: bash

    # Default (PhysX events, armature=0.0)
    python train.py --task=Isaac-Velocity-Rough-Anymal-C-v0

    # Newton (Newton events, armature=0.01)
    python train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 presets=newton


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
     - ``env.events=newton``
     - Replace entire section
   * - Global preset
     - ``presets=newton``
     - Apply everywhere matching
   * - Combined
     - ``presets=newton env.sim.dt=0.001``
     - Global + scalar overrides
