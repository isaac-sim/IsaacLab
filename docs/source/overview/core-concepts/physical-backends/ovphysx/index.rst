OvPhysX Backend
===============

.. warning::

    OvPhysX is **highly experimental** and is not recommended for general use yet.
    The public surface is changing rapidly while the backend is under active
    development. Expect feature coverage and test commands to change between
    Isaac Lab 3.0 beta releases.

.. warning::

    Do not combine OvPhysX with the Kit visualizer. Commands such as
    ``presets=ovphysx --visualizer kit`` are unsupported because OvPhysX
    loads USD-dependent PhysX plugins from its own package, while Kit already
    owns a separate USD/plugin stack in the same process. Use
    ``--visualizer newton``, ``--visualizer rerun``, ``--visualizer viser``,
    or omit ``--visualizer`` for headless execution.

OvPhysX is a kit-less variant of the PhysX backend. It drives PhysX directly
(without the Omniverse Kit runtime) and reads scene-level solver parameters
from the USD ``PhysicsScene`` prim rather than from a Python config. The Python
config :class:`~isaaclab_ovphysx.physics.OvPhysxCfg` only exposes the handful of
GPU buffer sizes that are not represented on the USD schema.

OvPhysX is selected through :class:`~isaaclab_ovphysx.physics.OvPhysxCfg`:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_ovphysx.physics import OvPhysxCfg

    sim_cfg = SimulationCfg(physics=OvPhysxCfg())

Why use OvPhysX?
----------------

* **Kit-less execution.** OvPhysX avoids Omniverse Kit, which makes it a useful
  experimental path for headless deployments and for backends that don't need
  the Kit runtime stack.
* **USD-as-source-of-truth.** Solver parameters are taken from the
  ``PhysicsScene`` USD prim, so authoring tools that already manage USD scenes
  do not need a parallel Python config.

What works today
----------------

The asset and sensor surface tracks PhysX, but only a subset is implemented and
validated at the time of writing. The following pieces are available on
``release/3.0.0-beta2``:

* RigidObject — merged via
  `PR #5426 <https://github.com/isaac-sim/IsaacLab/pull/5426>`_.
* Articulation — merged via
  `PR #5459 <https://github.com/isaac-sim/IsaacLab/pull/5459>`_.
* RigidObjectCollection — merged via
  `PR #5570 <https://github.com/isaac-sim/IsaacLab/pull/5570>`_.
* Contact Sensor — merged via
  `PR #5422 <https://github.com/isaac-sim/IsaacLab/pull/5422>`_.
* SceneDataProvider — merged via
  `PR #5589 <https://github.com/isaac-sim/IsaacLab/pull/5589>`_.
* FrameView — merged via
  `PR #5678 <https://github.com/isaac-sim/IsaacLab/pull/5678>`_.

Additional OvPhysX work remains in flight. IMU, Frame Transformer, Joint Wrench,
PVA, Ray Caster, and rendering support are not documented as supported here
until their implementations land on ``release/3.0.0-beta2`` and pass the backend smoke
tests.

Installation
------------

The Isaac Lab source install includes the ``isaaclab_ovphysx`` package, but it
does not install the heavier ``ovphysx`` runtime wheel by default. After a
standard source install, install the optional OvPhysX runtime dependency from
the repository root:

.. code-block:: bash

    ./isaaclab.sh -i 'ov[ovphysx]'

You can also install all OV runtime wheels with:

.. code-block:: bash

    ./isaaclab.sh -i 'ov[all]'

The ``ov[ovphysx]`` selector installs ``source/isaaclab_ovphysx`` with its
``[ovphysx]`` extra. If the wheel is missing, OvPhysX-specific tests skip with
``ovphysx wheel not installed`` and user code fails at import time.

Testing the Installation
------------------------

First check that the Python package and runtime wheel import correctly:

.. code-block:: bash

    ./isaaclab.sh -p -c "import ovphysx.types; from isaaclab_ovphysx.physics import OvPhysxCfg; print('OvPhysX runtime OK')"

Then run a small backend smoke test:

.. code-block:: bash

    ./isaaclab.sh -p -m pytest source/isaaclab_ovphysx/test/assets/test_rigid_object.py::test_initialization -k cpu

To try a task that declares an OvPhysX physics preset, use the same preset CLI
syntax as the other backends:

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct-v0 --num_envs 128 --headless presets=ovphysx

This command starts a headless zero-action rollout; stop it with ``Ctrl+C``
after the environment has started and stepped successfully.

Status and follow-up
--------------------

OvPhysX is still experimental, so the feature list above is intentionally
conservative. Broader feature coverage and documentation parity are tracked in
`issue #5634 <https://github.com/isaac-sim/IsaacLab/issues/5634>`_.

For architectural context, see :doc:`../../multi_backend_architecture`.
