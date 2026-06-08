Supported Features
==================

The Newton backend is in beta. Breaking changes and incomplete documentation are
still expected, and official support or debugging assistance will only be
available once the integration reaches an official release.


Discovering Newton-Supported Tasks
----------------------------------

A task supports the Newton backend when its physics ``PresetCfg`` declares a
``newton_mjwarp`` (or ``newton_kamino``) entry. To list every task that
currently supports Newton:

.. code-block:: bash

    grep -rln "newton_mjwarp" source/isaaclab_tasks/

Passing ``physics=newton_mjwarp`` to a task without that preset will raise an
error at launch. The :doc:`mjwarp-solver` page covers how to add a Newton
preset to your own task.


Supported APIs
--------------

The following capabilities are covered by the Newton backend on ``release/3.0.0-beta2`` at
the time of writing. The list is non-exhaustive and continues to grow.

isaaclab
^^^^^^^^

* Articulation API (multi-link and single-body articulations)
* Rigid Object and Rigid Object Collection APIs
* Sensors: Contact Sensor, IMU, Frame Transformer, Joint Wrench, PVA
* Direct and Manager-based single-agent workflows
* Backend-neutral deformable object API
* Omniverse Kit visualizer (when Isaac Sim is installed)
* Newton-Warp visualizer (kit-less)
* Tiled rendering via the Newton-Warp renderer

isaaclab_contrib
^^^^^^^^^^^^^^^^

* Experimental Newton deformable objects
* VBD deformable solver (see :doc:`using-vbd-solver`)
* Coupled MJWarp + VBD and Featherstone + VBD solver managers

The following sensors are backend-agnostic (implemented in ``isaaclab`` core)
and work transparently with Newton:

* Ray Caster
* Camera — see :doc:`../../sensors/camera`

isaaclab_assets
^^^^^^^^^^^^^^^

* Quadrupeds: Anymal-B, Anymal-C, Anymal-D, Unitree A1, Unitree Go1, Unitree
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
* Locomotion velocity, flat terrain: A1, Anymal-B, Anymal-C, Anymal-D, Cassie,
  Unitree G1, Go1, Go2, Unitree H1, Spot
* Locomotion velocity, rough terrain: Anymal-C, Cassie, Go1, Go2
* Manipulation: reach (Franka, UR10), cabinet, dexsuite
* Manipulation lift with deformable objects: Franka soft-body lift, Franka cloth
  lift (via coupled MJWarp + VBD)


Solver Coverage
---------------

* **MuJoCo-Warp solver**: the primary, validated path for every supported task.
* **Kamino solver**: beta. Currently validated on ``Isaac-Cartpole-Direct-v0``,
  ``Isaac-Ant-Direct-v0``, ``Isaac-Cartpole-v0``, and ``Isaac-Ant-v0``. See
  :doc:`kamino-solver`.
* **VBD solver**: experimental, exposed through :mod:`isaaclab_contrib.deformable`
  for cloth and soft-body simulation. Most often used inside the coupled
  MJWarp + VBD or Featherstone + VBD managers so one solver advances rigid
  bodies and VBD advances deformable particles. See :doc:`using-vbd-solver`
  and :doc:`newton-manager-abstraction`.


Known Gaps
----------

* Soft bodies and particles are available through the experimental VBD path in
  :mod:`isaaclab_contrib.deformable`; other non-rigid PhysX features are not
  yet covered.
* Behaviour on stiff contact stacks can diverge from PhysX; expect to retune
  contact and substep parameters when porting tasks across backends.
* Multi-agent and self-play workflows are not yet wired up for Newton.
