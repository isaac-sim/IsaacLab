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
      delay
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

Delay configuration
~~~~~~~~~~~~~~~~~~~

:class:`~isaaclab.utils.DelayCfg` wraps an observation callable with latency and sample holds. It uses the same
:class:`~isaaclab.utils.buffers.DelayBuffer` primitive as the legacy :class:`~isaaclab.actuators.DelayedPDActuator`.
Native Newton actuators currently use Newton's own delay implementation. Each stream owns its
history and advances an index without shifting stored frames. The caller owns lag sampling and delivery cadence.

For example, add latency to a proprioceptive observation:

.. code-block:: python

    from isaaclab.envs import mdp
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.utils import DelayCfg

    joint_observation = ObservationTermCfg(
        func=DelayCfg(term=mdp.joint_pos_rel, params={}, min_lag=1, max_lag=3)
    )

Parameters of the wrapped callable belong in ``DelayCfg.params``; leave the outer term's ``params`` empty.
Delay runs before observation modifiers, noise, clipping, and scaling. The manager constructs the callable
and resets both its state and its delay history through the normal term lifecycle.

Lags and periods count wrapped term calls. Observation delay advances whenever the term is computed,
including extra observation queries. Actuator-target latency counts physics steps. Sharing buffer storage and
indexing does not couple the two clocks or their sampling policies.

At call :math:`t`, a lag :math:`\ell_t` is sampled uniformly from the inclusive configured bounds. The
candidate sample has timestamp :math:`s_t = \max(0, t - \ell_t)`, relative to that environment's last reset.
A refresh delivers this candidate only if its timestamp is at least as recent as the last delivered sample.
Otherwise, the previous output is held. Thus variable latency never delivers samples out of order.

``update_period`` controls refresh opportunities; ``per_env_phase`` randomizes their phase at reset.
``hold_prob`` independently skips a refresh for each environment. For example,
``DelayCfg(term=mdp.joint_pos_rel, update_period=3, per_env_phase=False)`` delivers inputs at calls 0, 3, 6, and so on,
holding each delivered frame between those calls. ``hold_prob=1`` holds the first frame until reset.
Held frames may become older than ``max_lag``: that bound limits sampled latency, not the duration of a hold.
``per_env=False`` shares the sampled lag across environments; their holds and reset schedules remain independent.

The first input after reset is delivered immediately. During warmup, unavailable history resolves to that first
input. Partial resets invalidate only the selected environments, so history from a previous episode cannot leak
into a new one. Invalid lag updates through ``DelayBuffer.set_time_lag()`` raise before changing stored lags.

.. autoclass:: isaaclab.utils.DelayCfg
   :members:

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
