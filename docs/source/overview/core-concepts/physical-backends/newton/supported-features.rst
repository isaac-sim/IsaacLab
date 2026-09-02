Supported Features
==================

The Newton backend is in beta. Breaking changes and incomplete documentation are
still expected, and official support or debugging assistance will only be
available once the integration reaches an official release.


Discovering Newton-Supported Tasks
----------------------------------

Most multi-backend tasks support Newton when their physics ``PresetCfg``
declares a ``newton_mjwarp`` (or ``newton_kamino``) entry. To list tasks with
a selectable Newton preset:

.. code-block:: bash

    grep -rln "newton_mjwarp" source/isaaclab_tasks/

Tasks built specifically for Newton can instead assign
:class:`~isaaclab_newton.physics.NewtonCfg` directly. The coupled-MPM
``IsaacContrib-Franka-Pour`` artifact-backed task and
``IsaacContrib-UR10-Particle-Push`` use this fixed-Newton pattern and should be launched
without a ``physics=`` selector. Passing
``physics=newton_mjwarp`` to a task without that preset will raise an error at
launch. The :doc:`mjwarp-solver` page covers how to add a selectable Newton
preset to your own task.

Implicit MPM Demos
------------------

The repository includes standalone granular, rigid-coupled snowball, and
mesh-cavity filling examples. See :doc:`using-mpm` for the runnable commands,
minimal authoring path, and tuning guidance.


Supported APIs
--------------

The following capabilities are covered by the Newton backend on ``develop`` at
the time of writing. The list is non-exhaustive and continues to grow.

isaaclab
^^^^^^^^

* Articulation API (multi-link and single-body articulations)
* Rigid Object and Rigid Object Collection APIs
* Sensors: Contact Sensor, IMU, Frame Transformer, Joint Wrench, PVA
* Direct and Manager-based single-agent workflows
* Backend-neutral deformable object API
* Omniverse Kit visualizer (when Isaac Sim is installed)
* Cable Object API for standalone, open, linear, unwelded curves using VBD
* Newton-Warp visualizer (kit-less)
* Tiled rendering via the Newton-Warp renderer

isaaclab_newton
^^^^^^^^^^^^^^^^

* Standalone VBD deformable solver

* Implicit Material Point Method (MPM) solver and declarative particle assets
* Fixed and capture-compatible capacity-bounded sparse MPM grids
* Standard visual-material binding for MPM particle visualization

isaaclab_contrib
^^^^^^^^^^^^^^^^

* Newton deformable object integration
* MJWarp and VBD proxy and ADMM coupling
* Proxy-based coupling for rigid and particle solvers, including MJWarp + MPM

The following sensors are backend-agnostic (implemented in ``isaaclab`` core)
and work transparently with Newton:

* Ray Caster
* Camera — see :doc:`../../sensors/camera`

isaaclab_assets
^^^^^^^^^^^^^^^

* Quadrupeds: AnymalB, AnymalC, AnymalD, Unitree A1, Unitree Go1, Unitree
  Go2, Spot
* Humanoids: Unitree H1, Unitree G1, Cassie
* Arms and hands: Franka, UR10, Allegro Hand, Shadow Hand
* Toy examples: Cartpole, Ant, Humanoid

isaaclab_tasks
^^^^^^^^^^^^^^

Direct workflows:

* Cartpole (state, RGB, depth)
* Ant, Humanoid
* Allegro Hand Repose Cube, Shadow Hand, Shadow Hand Over
* Locomotion (shared base env)

Manager-based workflows:

* Classic: Cartpole, Ant, Humanoid
* Locomotion velocity, flat terrain: A1, AnymalB, AnymalC, AnymalD, Cassie,
  Unitree G1, Go1, Go2, Unitree H1, Spot
* Locomotion velocity, rough terrain: AnymalC, Cassie, Go1, Go2
* Manipulation: reach (Franka, UR10), cabinet, lift and reorient (Franka, KukaAllegro)
* Manipulation lift with deformable objects: Franka soft-body lift, Franka cloth
  lift (via MJWarp and VBD proxy coupling)
* Coupled MPM manipulation: Franka pour and UR10 particle push (MJWarp + MPM)


Solver Coverage
---------------

* **MuJoCo-Warp solver**: the primary, validated path for every supported task.
* **Kamino solver**: beta. Currently validated on ``Isaac-Cartpole-Direct``,
  ``Isaac-Ant-Direct``, ``Isaac-Cartpole``, and ``Isaac-Ant``. See
  :doc:`kamino-solver`.
* **VBD solver**: experimental, exposed through :mod:`isaaclab_newton.physics`
  for standalone cloth, soft-body, and cable simulation. Rigid and deformable
  scenes can use proxy or ADMM coupling from :mod:`isaaclab_contrib.coupling` so
  MJWarp advances rigid bodies and VBD advances deformable particles. Cable objects work with standalone
  VBD and with :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` when a named VBD
  entry owns the cable. See :doc:`using-vbd-solver` and
  :doc:`newton-manager-abstraction`.
* **Implicit MPM solver**: experimental, supporting standalone particle
  materials and proxy-coupled rigid-MPM scenes. Capacity-bounded sparse grids
  and fixed grids support CUDA graph capture.


Known Gaps
----------

* Soft bodies are available through the experimental VBD path, while particle
  materials are available through implicit MPM; other non-rigid PhysX features
  are not yet covered.
* Behaviour on stiff contact stacks can diverge from PhysX; expect to retune
  contact and substep parameters when porting tasks across backends.
* Multi-agent and self-play workflows are not yet wired up for Newton.
