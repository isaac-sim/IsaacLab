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

        .. tab-set::

           .. tab-item:: uv (Recommended)

              .. code-block:: shell

                  uv run isaaclab train --rl_library rsl_rl --task=Isaac-Cartpole env.actions.joint_effort.scale=10.0 agent.seed=2024

           .. tab-item:: isaaclab.sh / isaaclab.bat

              .. code-block:: shell

                  ./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Cartpole env.actions.joint_effort.scale=10.0 agent.seed=2024

    .. tab-item:: rl_games
        :sync: rl_games

        .. tab-set::

           .. tab-item:: uv (Recommended)

              .. code-block:: shell

                  uv run --extra rl-games isaaclab train --rl_library rl_games --task=Isaac-Cartpole env.actions.joint_effort.scale=10.0 agent.params.seed=2024

           .. tab-item:: isaaclab.sh / isaaclab.bat

              .. code-block:: shell

                  ./isaaclab.sh train --rl_library rl_games --task=Isaac-Cartpole env.actions.joint_effort.scale=10.0 agent.params.seed=2024

    .. tab-item:: skrl
        :sync: skrl

        .. tab-set::

           .. tab-item:: uv (Recommended)

              .. code-block:: shell

                  uv run --extra skrl isaaclab train --rl_library skrl --task=Isaac-Cartpole env.actions.joint_effort.scale=10.0 agent.seed=2024

           .. tab-item:: isaaclab.sh / isaaclab.bat

              .. code-block:: shell

                  ./isaaclab.sh train --rl_library skrl --task=Isaac-Cartpole env.actions.joint_effort.scale=10.0 agent.seed=2024

    .. tab-item:: sb3
        :sync: sb3

        .. tab-set::

           .. tab-item:: uv (Recommended)

              .. code-block:: shell

                  uv run --extra sb3 isaaclab train --rl_library sb3 --task=Isaac-Cartpole env.actions.joint_effort.scale=10.0 agent.seed=2024

           .. tab-item:: isaaclab.sh / isaaclab.bat

              .. code-block:: shell

                  ./isaaclab.sh train --rl_library sb3 --task=Isaac-Cartpole env.actions.joint_effort.scale=10.0 agent.seed=2024

The above command will run training with the task ``Isaac-Cartpole`` without selecting a visualizer,
and set the ``env.actions.joint_effort.scale`` parameter to 10.0 and the ``agent.seed`` parameter to 2024.

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

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_manager_env_cfg.py
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
Elements in dictionaries are handled as parameters in the hierarchy. For example, in the Cartpole environment:

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_manager_env_cfg.py
    :language: python
    :start-at: reset_cart_position = EventTerm(
    :end-before: reset_pole_position = EventTerm(

the ``position_range`` parameter can be modified with ``env.events.reset_cart_position.params.position_range="[-2.0, 2.0]"``.
This example shows two noteworthy points:

- The value contains a space, so it must be enclosed in quotes for the shell.
- The parameter is a list while it is a tuple in the config. This is due to the fact that Hydra does not support tuples.


Modifying inter-dependent parameters
------------------------------------

Particular care should be taken when modifying the parameters using command line arguments. Some of the configurations
perform intermediate computations based on other parameters. These computations will not be updated when the parameters
are modified.

For example, for the configuration of the Cartpole camera environment:

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_camera_env_cfg.py
    :language: python
    :start-at: class CartpoleTiledCameraCfg
    :end-at: observation_space = [3, 96, 96]

The configuration declares the single-frame channel count and a default spatial size.
At environment initialization, ``CartpoleCameraEnv`` rebuilds ``observation_space`` from
the resolved camera: the default ``frame_stack=2`` expands channels, and height/width are
taken from ``tiled_camera``. So ``env.tiled_camera.width=128 env.tiled_camera.height=128``
alone yields an effective stacked shape of ``[6,128,128]`` without also overriding
``env.observation_space``. The channel entry in ``observation_space`` must still match the
camera data type (for example ``[1, ...]`` with ``presets=depth``); presets already set this.

Class-body assignments are evaluated once at import time and do **not** track later
Hydra overrides unless runtime code explicitly rebuilds the dependent value, as this
camera environment does.

Similarly, the ``__post_init__`` method is not updated with the command line inputs. In the ``LocomotionVelocityRoughEnvCfg``, for example,
the post init update is as follows:

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py
    :language: python
    :start-at: class LocomotionVelocityRoughEnvCfg

Here, when modifying ``env.decimation`` or ``env.sim.dt``, the user needs to give the updated ``env.sim.render_interval``,
``env.scene.height_scanner.update_period``, and ``env.scene.contact_forces.update_period`` as input as well.


Custom Configuration Validation
--------------------------------

Configclass objects can define a ``validate_config()`` method to perform domain-specific
validation after all fields have been resolved. This hook is called automatically after preset
resolution and MISSING-field checks succeed, allowing you to catch invalid parameter
combinations early with clear error messages.

For example, the Franka reach configuration validates that its Newton IK action
preset is paired with a Newton physics configuration:

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/core/reach/config/franka/franka_reach_env_cfg.py
    :language: python
    :start-at: def validate_config(self) -> None:
    :end-at: raise ValueError("The 'newton_ik' action preset requires a Newton physics preset.")

**When it runs:**

1. All ``MISSING`` fields are checked first — if any remain, ``TypeError`` is raised.
2. Only then is ``validate_config()`` called on the **top-level** config object.
3. The hook should raise ``ValueError`` with a clear message and migration guidance.

**Common validation patterns:**

- Compatibility between independently selectable controllers, physics configurations, and renderers
- Renderer, camera data type, and feature extractor compatibility
- Numeric relationships or limits that cannot be expressed by field types alone


Preset System
-------------

For a user-focused introduction to choosing physics, rendering, and task
variants, start with :doc:`/source/concepts/backends_and_presets`. This section
covers the complete preset definition and resolution behavior.

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

The effective precedence, from lowest to highest, is:

1. **Defaults**: Each unresolved ``PresetCfg`` falls back to its ``default`` field.
2. **Typed and domain selections**: ``physics=newton_mjwarp`` selects physics while
   ``presets=rgb`` broadcasts a task-specific name to every matching config.
3. **Path selections**: ``env.sim.physics=newton_kamino`` targets one specific
   ``PresetCfg`` and takes precedence at that path.
4. **Play-mode changes**: When requested by a play command, the environment's
   ``play_mode()`` changes are applied to the resolved config.
5. **Scalar overrides**: ``env.sim.dt=0.001`` has the final say for an individual field.

If multiple broadcast names select different alternatives at the same active path,
resolution fails instead of silently choosing one.


Defining Presets with PresetCfg
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a :class:`~isaaclab_tasks.utils.hydra.PresetCfg` subclass where each field
is a named alternative. The ``default`` field is the config used when no CLI
override is given:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_tasks.utils import PresetCfg

    @configclass
    class PhysicsPresetsCfg(PresetCfg):
        isaacsim_physx: PhysxCfg = PhysxCfg()
        default: PhysxCfg = isaacsim_physx
        newton_mjwarp: NewtonCfg = NewtonCfg()

    @configclass
    class MyEnvCfg:
        sim: SimulationCfg = SimulationCfg(physics=PhysicsPresetsCfg())

Physics is owned by :class:`~isaaclab.sim.SimulationCfg`, so the preset's config
path is ``env.sim.physics``. For backend selection, prefer the typed selector:

.. code-block:: bash

    uv run isaaclab train --rl_library rsl_rl \
        --task Isaac-Cartpole physics=newton_mjwarp

Use the path form ``env.sim.physics=newton_mjwarp`` only when you intentionally
want to replace that one preset node without selecting other matching task presets.

The ``default`` field can be set to ``None`` to make an optional feature that is
disabled unless explicitly selected:

.. code-block:: python

    @configclass
    class CameraSettingsCfg:
        width: int = 64
        height: int = 64

    @configclass
    class CameraPresetCfg(PresetCfg):
        default = None
        small: CameraSettingsCfg = CameraSettingsCfg()
        large: CameraSettingsCfg = CameraSettingsCfg(width=256, height=256)

    @configclass
    class SceneCfg:
        camera: CameraPresetCfg = CameraPresetCfg()

Here, ``env.scene.camera`` resolves to ``None`` by default. A registered task using
this config can activate the large camera with the path selector
``env.scene.camera=large``.


.. _hydra-backend-solver-presets:

Backend and Solver Presets
^^^^^^^^^^^^^^^^^^^^^^^^^^

Physics backend selection uses the same preset system. A task can define a
``PresetCfg`` whose entries replace the complete physics config:

The Cartpole task's definition is a maintained example:

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_manager_env_cfg.py
    :language: python
    :start-at: class CartpolePhysicsCfg(PresetCfg):
    :end-before: ##

The ``newton_mjwarp`` and ``newton_kamino`` entries both select the Newton physics backend because
both entries are :class:`~isaaclab_newton.physics.NewtonCfg` objects. The difference
is the solver configuration: ``newton_mjwarp`` uses
:class:`~isaaclab_newton.physics.MJWarpSolverCfg`, while ``newton_kamino`` uses
:class:`~isaaclab_newton.physics.KaminoPADMMSolverCfg`.

Kamino is therefore a solver preset, not a separate Isaac Lab backend. The same
Newton assets, sensors, renderers, and visualizers are used after the preset is
resolved. It is a Proximal Alternating Direction Method of Multipliers (P-ADMM)
based solver for constrained rigid multi-body dynamics, and its Isaac Lab support
is currently beta.

.. note::

    Kamino support is experimental and currently depends on the asset being
    structured in a way that Kamino can consume. Assets that work with the
    MuJoCo-Warp or PhysX presets may still require model-structure updates before
    they work with ``physics=newton_kamino``.

.. code-block:: bash

    # Preferred: select and validate a physics preset by type
    uv run isaaclab train --rl_library rsl_rl \
        --task Isaac-Cartpole physics=newton_kamino

    # Advanced: replace only the physics config at this path
    uv run isaaclab train --rl_library rsl_rl \
        --task Isaac-Cartpole env.sim.physics=newton_kamino

Backend support is task-specific and changes as tasks are validated. Use the task's
``--help`` output as the source of truth. Passing ``physics=newton_kamino`` to a
task that does not advertise it fails; it does not add a Kamino configuration to
that task.


Inline Presets with preset()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For simple values (scalars, lists) that don't warrant a full subclass, use the
:func:`~isaaclab_tasks.utils.hydra.preset` factory. It dynamically creates a
``PresetCfg`` instance from keyword arguments:

.. code-block:: python

    from isaaclab_tasks.utils.hydra import preset

    # Scalar preset -- no boilerplate subclass
    self.scene.robot.actuators["legs"].armature = preset(
        default=0.0, isaacsim_physx=0.0, newton_mjwarp=0.01, physx=0.0
    )

This is equivalent to defining a ``PresetCfg`` subclass with the same ``float``
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

    # Select MJWarp physics and all matching dependent alternatives
    uv run isaaclab train --rl_library rsl_rl \
        --task IsaacContrib-Velocity-Rough-AnymalC physics=newton_mjwarp

The typed ``physics=`` selector uses broadcast resolution for the selected name,
so matching task-specific alternatives such as this armature value are updated too.
It additionally verifies that the name resolved a physics configuration; the free-form
``presets=`` selector does not provide that type check.


Typed Preset Selectors
^^^^^^^^^^^^^^^^^^^^^^

The preset CLI layer recognizes three ``key=value`` tokens (no leading dashes)
that can be appended to any training or play script command:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Token
     - Effect
   * - ``physics=NAME``
     - Typed selector for :class:`~isaaclab.physics.PhysicsCfg` variants
   * - ``renderer=NAME``
     - Typed selector for :class:`~isaaclab.renderers.renderer_cfg.RendererCfg` variants
   * - ``presets=NAME[,NAME,...]``
     - Broadcast: applied to every matching :class:`~isaaclab_tasks.utils.hydra.PresetCfg` in the config tree

The typed selectors use the same broadcast resolution as ``presets=``. This means
``physics=newton_mjwarp`` can also update dependent task presets with the same name,
such as actuator or event settings. They are not interchangeable, however: a typed
selector must resolve at least one config of its declared type or it raises an error.
Use ``physics=`` and ``renderer=`` for backend choices, and reserve ``presets=`` for
task-specific modes.

**Common physics preset names** (only when advertised by the task):

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Backend
   * - ``isaacsim_physx``
     - Concrete Isaac Sim PhysX configuration
   * - ``physx``
     - Automatic PhysX-family selection between configured alternatives
   * - ``newton_mjwarp``
     - Newton physics with the MuJoCo-Warp solver
   * - ``newton_kamino``
     - Newton physics with the Kamino solver (beta; limited tasks — see :ref:`hydra-backend-solver-presets`)
   * - ``ovphysx``
     - Concrete OvPhysX configuration for supported kit-less tasks

**Common renderer preset names** (when provided by
:class:`~isaaclab_tasks.utils.presets.MultiBackendRendererCfg`):

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Renderer
   * - ``isaacsim_rtx``
     - Concrete Isaac Sim RTX renderer
   * - ``rtx``
     - Automatic RTX-family selection between configured alternatives
   * - ``newton_renderer``
     - Newton Warp renderer
   * - ``ovrtx``
     - Concrete OVRTX renderer for supported kit-less tasks

The implicit ``default`` field is task-specific and is intentionally omitted from
``--help``. Do not infer a task's default backend from these conventional names;
inspect its help output or configuration. Automatic choices such as ``physics=physx``
and ``renderer=rtx`` are opt-in, not universal defaults.

Domain presets (observation modes, camera configurations, etc.) are task-specific.
Pass ``--task=<task-name> --help`` to a training command to see all presets available
for that task, grouped by selector type. Reinforcement-learning commands also list
the registered ``--agent`` values for the selected library. When a task declares
preset-to-agent compatibility, the compatible presets appear beneath each agent:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run isaaclab train --rl_library rsl_rl \
               --task Isaac-Cartpole-Camera --help

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          ./isaaclab.sh train --rl_library rsl_rl \
               --task Isaac-Cartpole-Camera --help

Preset and agent selection are otherwise independent. A task may use an alternate
agent for symmetry, recurrence, or another algorithm without changing its environment
preset.

.. note::

    Legacy aliases ``newton`` → ``newton_mjwarp`` and ``kamino`` → ``newton_kamino``
    are still accepted but emit a :class:`FutureWarning`. The renderer aliases
    ``isaacsim_rtx_renderer`` → ``isaacsim_rtx`` and ``ovrtx_renderer`` → ``ovrtx``
    behave the same way. Prefer the canonical names.


Using Presets
^^^^^^^^^^^^^

**Typed selectors** -- preferred form for physics and renderer backends:

.. code-block:: bash

    # Switch to Newton MuJoCo-Warp physics
    uv run isaaclab train --rl_library rsl_rl \
        --task IsaacContrib-Velocity-Rough-AnymalC physics=newton_mjwarp

    # Switch to Newton renderer for camera environments
    uv run isaaclab train --rl_library rsl_rl \
        --task Isaac-Cartpole-Camera-Direct renderer=newton_renderer

    # Combine typed selectors with a task-specific observation preset
    uv run isaaclab train --rl_library rsl_rl \
        --task Isaac-Cartpole-Camera-Direct \
        physics=newton_mjwarp renderer=newton_renderer presets=rgb

**Path presets** -- select a specific preset for one config path:

.. code-block:: bash

    # Replace only this task's physics preset node
    uv run isaaclab train --rl_library rsl_rl \
        --task Isaac-Cartpole env.sim.physics=newton_kamino

**Domain presets** -- broadcast a task-specific name everywhere it exists:

.. code-block:: bash

    # Keep the observation pipeline and camera data type in sync
    uv run isaaclab train --rl_library rsl_rl \
        --task Isaac-Cartpole-Camera presets=depth

**Multiple domain presets** -- apply several non-conflicting task choices:

.. code-block:: bash

    uv run isaaclab train --rl_library rsl_rl \
        --task Isaac-Lift-KukaAllegro-Camera presets=duo_camera,rgb128

**Combined** -- typed selectors, a domain preset, and a scalar override:

.. code-block:: bash

    uv run isaaclab train --rl_library rsl_rl \
        --task Isaac-Cartpole-Camera \
        physics=newton_mjwarp renderer=newton_renderer presets=rgb \
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

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/anymal_c/rough_env_cfg.py
    :language: python
    :start-at: class AnymalCRoughEnvCfg

The base velocity configuration also defines ``newton_mjwarp`` alternatives for
physics and a center-of-mass randomization event that MJWarp disables. An explicit
``physics=newton_mjwarp`` resolves the physics config and every active dependent
alternative with that name: the event is disabled and the ANYmal-C actuator
armature becomes ``0.01``. The typed selector also verifies that a physics preset
was actually selected.

.. code-block:: bash

    uv run isaaclab train --rl_library rsl_rl \
        --task IsaacContrib-Velocity-Rough-AnymalC physics=newton_mjwarp

Without a selector, each ``PresetCfg`` independently uses its own ``default``;
the name of a default alternative is not broadcast to other preset nodes.


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
     - ``env.sim.physics=newton_kamino``
     - Replace entire section
   * - Domain preset
     - ``presets=rgb``
     - Apply a task-specific name everywhere matching
   * - Typed physics selector
     - ``physics=newton_mjwarp``
     - Select a physics variant, update matching dependent presets, and require a typed match
   * - Typed renderer selector
     - ``renderer=newton_renderer``
     - Select a renderer variant and require a typed match
   * - Combined
     - ``physics=newton_mjwarp renderer=newton_renderer presets=rgb env.sim.dt=0.001``
     - Typed selectors + domain preset + scalar override
