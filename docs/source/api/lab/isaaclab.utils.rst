isaaclab.utils
==============

.. automodule:: isaaclab.utils

   .. Rubric:: Submodules

   .. autosummary::

      io
      array
      assets
      buffers
      datasets
      dict
      interpolation
      logger
      math
      mesh
      modifiers
      noise
      seed
      sensors
      string
      timer
      types
      version
      warp

   .. Rubric:: Functions

   .. autosummary::

      configclass

Configuration class
~~~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.configclass
   :members:
   :show-inheritance:

IO operations
~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.io
   :members:
   :imported-members:
   :show-inheritance:

Array operations
~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.array
   :members:
   :show-inheritance:

Asset operations
~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.assets
   :members:
   :show-inheritance:

Buffer operations
~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.buffers
   :members:
   :imported-members:
   :inherited-members:
   :show-inheritance:

.. _shared-delay:

Callable composition and delay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~isaaclab.utils.DelayCfg` wraps an observation, action term, or actuator configuration.
Each wrapper owns its complete callable computation and its state; nesting preserves evaluation order.
The owner constructs the enclosed term and submits the final commands. The wrapper does not submit commands itself.

.. code-block:: python

    from isaaclab.actuators import IdealPDActuatorCfg
    from isaaclab.envs import mdp
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.utils import DelayCfg

    observation = ObservationTermCfg(
        func=DelayCfg(term=mdp.joint_pos_rel, params={}, on="output", min_lag=1, max_lag=3)
    )
    action = DelayCfg(
        term=mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"]),
        on="input", min_lag=2, max_lag=2,
    )
    actuator = DelayCfg(
        term=IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=5.0),
        on="input", min_lag=2, max_lag=2, resample="reset",
    )

Assign ``action`` to an action configuration entry and ``actuator`` to an ``ArticulationCfg.actuators`` entry.
Function parameters belong on the innermost wrapper's ``params``; class-based terms use their own configuration.
Observation delay runs before observation modifiers, noise, clipping, and scaling.

``on="input"`` delays the first data argument before calling the enclosed term; other arguments stay current.
For an actuator, this delays desired position, velocity, and feed-forward effort together while using current
joint feedback. ``fields=("position",)`` delays just the position target. ``on="output"`` (the default) delays
the returned data: observations, generated action commands, or explicit actuator effort. An implicit actuator
has no exposed effort output and rejects output delay; input delay remains supported.

Action input delay counts policy steps. Action output delay counts physics steps and includes each step's live
state when generating commands, including relative position commands. Output delay is supported by joint position,
relative joint position, velocity, and effort actions. Older action terms that submit commands inside
``apply_actions()`` support input delay only. New action implementations return their command bundle from
``term()``; the manager submits it after the complete wrapper computation. ``term(actions)`` stages policy input.

Actuator delays count physics steps, including Newton's internal decimation and CUDA graph replay.
``computed_effort`` describes the current controller demand; ``applied_effort`` describes the delivered effort.
Requested commands retain separate storage, so a target written once remains the source on subsequent steps.
Observation delays count term evaluations, including extra observation queries. The observation manager resets the term after shape
inference; serialization only inspects its configuration.

The shared delay implementation uses one :class:`~isaaclab.utils.buffers.DelayBuffer` per signal bundle.
It stores a ring of frames without shifting history. ``resample="step"`` samples an inclusive uniform lag on
each evaluation; ``resample="reset"`` samples only at reset. A candidate older than the last delivered sample
is held to prevent out-of-order delivery. ``update_period`` controls delivery opportunities and
``per_env_phase`` samples their phase at reset. ``hold_prob`` skips a delivery opportunity. Holds may produce
samples older than ``max_lag``. ``per_env=False`` shares lag and hold draws across the batch; reset histories
and optional phases remain per environment. ``seed`` selects an independent, reproducible random stream.

The first evaluation after reset delivers fresh data. Missing history resolves to the oldest available sample
in the current episode. Partial resets affect only the selected environments and propagate through nested terms.
Returned data can be modified without corrupting retained history. Legacy ``DelayedPDActuatorCfg`` is a deprecated
constructor returning ``DelayCfg(term=IdealPDActuatorCfg(...), on="input", resample="reset", ...)``.
All configured actuator delays use this shared implementation, including native Newton execution.
The authored controller has no additional Newton delay.

``DelayedPDActuatorCfg`` and ``DelayedPDActuator`` remain callable until 3.2 but are no longer classes.
The former returns a ``DelayCfg``; the latter returns a ``Delay`` whose ``term`` owns the controller properties.
Replace old subclassing with an ``IdealPDActuator``/``IdealPDActuatorCfg`` subclass enclosed in ``DelayCfg``.
Edit controller fields through ``cfg.term`` and delay bounds through ``cfg.min_lag``/``cfg.max_lag``.
``RemotizedPDActuatorCfg`` now configures only the controller and its torque lookup table. Move its former
``min_delay``/``max_delay`` arguments to an outer ``DelayCfg`` with ``on="input"`` and ``resample="reset"``.

Actuator and delay-buffer invocation
++++++++++++++++++++++++++++++++++

Call actuator models as ``actuator(commands, joint_pos, joint_vel)`` and buffers as ``buffer(data)``.
Their ``compute(...)`` spelling is deprecated in Isaac Lab 3.0 and scheduled for removal in 3.2, after a full
release cycle. Existing custom ``compute`` overrides and ``super().compute()`` chains continue to work during
that transition. Migrate implementations to ``__call__`` and parent calls to ``super().__call__(...)``.
Legacy overrides warn at class definition; direct ``compute`` calls warn when invoked. Canonical calls to
migrated implementations emit no deprecation warning. Manager-level ``compute`` methods and the action
processing/application clocks are unchanged by this migration.

Custom mechanisms derive their configuration from ``WrapperCfg`` and implement a complete ``__call__`` and
``reset(env_ids)``. No input/output hook pair or universal side selector is required. Their input and output must
match the enclosing term's signal contract; custom code must also support the backend's graph capture when enabled.
Construction receives ``cfg``, the enclosed callable, ``num_envs``, ``device``, and the explicit keyword arguments
``input_supported``, ``output_supported``, and ``split_calls``. The last identifies terms whose calls with arguments
stage inputs and whose calls without arguments evaluate outputs on a different clock. These describe the owner's
call interface; they do not prescribe input/output hooks in a mechanism.

.. autoclass:: isaaclab.utils.WrapperCfg
   :members:
   :exclude-members: __init__

.. autoclass:: isaaclab.utils.Delay
   :members:
   :special-members: __call__

.. autoclass:: isaaclab.utils.DelayCfg
   :members:
   :exclude-members: __init__

Datasets operations
~~~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.datasets
   :members:
   :show-inheritance:
   :exclude-members: __init__, func

Dictionary operations
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.dict
   :members:
   :show-inheritance:

Interpolation operations
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.interpolation
   :members:
   :imported-members:
   :inherited-members:
   :show-inheritance:

Logger operations
~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.logger
   :members:
   :show-inheritance:

Math operations
~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.math
   :members:
   :inherited-members:
   :show-inheritance:

Mesh operations
~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.mesh
   :members:
   :imported-members:
   :show-inheritance:

Modifier operations
~~~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.modifiers
   :members:
   :imported-members:
   :special-members: __call__
   :inherited-members:
   :show-inheritance:
   :exclude-members: __init__, func

Noise operations
~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.noise
   :members:
   :imported-members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: __init__, func

Seed operations
~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.seed
   :members:
   :show-inheritance:

Sensor operations
~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.sensors
   :members:
   :show-inheritance:

String operations
~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.string
   :members:
   :show-inheritance:

Timer operations
~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.timer
   :members:
   :show-inheritance:

Type operations
~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.types
   :members:
   :show-inheritance:

Version operations
~~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.version
   :members:
   :show-inheritance:

Warp operations
~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.warp
   :members:
   :imported-members:
   :show-inheritance:

.. rubric:: Particle sampling functions

.. autosummary::

   sample_particles_in_mesh
   sample_particles_in_cavity

Warp Fabric kernels
^^^^^^^^^^^^^^^^^^^

Warp kernels for reading and writing Fabric ``Matrix4d`` attributes
(``omni:fabric:worldMatrix`` / ``omni:fabric:localMatrix``) via
:class:`wp.fabricarray` and :class:`wp.indexedfabricarray`. Used by
:class:`~isaaclab_physx.sim.views.FabricFrameView` to keep child world and
local matrices consistent without round-tripping through USD.

.. automodule:: isaaclab.utils.warp.fabric
   :members:
   :show-inheritance:
