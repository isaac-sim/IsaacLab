.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

Newton Native Data and Selection API
====================================

Mental model
------------

The Newton backend exposes its engine-owned `Model
<https://newton-physics.github.io/newton/stable/api/_generated/newton.Model.html>`_, `State
<https://newton-physics.github.io/newton/stable/api/_generated/newton.State.html>`_, `Control
<https://newton-physics.github.io/newton/stable/api/_generated/newton.Control.html>`_, and optional
``Contacts`` objects through :class:`isaaclab_newton.physics.NewtonManager`. Their Warp arrays are
the live engine data, rather than values pulled into a separate per-asset view buffer. The model
owns structural and static arrays and labels; a state owns evolving simulation arrays; and a
control owns actuation inputs. Contacts are optional and depend on the active solver and collision
path.

Lifecycle prerequisite
----------------------

Access Newton data after the simulation context has initialized the physics backend and built its
model. Treat the objects as invalid after a model rebuild and reacquire them after Isaac Lab has
reinitialized the simulation. This API is intended for code that can own the necessary lifecycle
and synchronization responsibilities.

Reuse Isaac Lab-owned access
----------------------------

Use the manager accessors to obtain the current Newton objects:

.. code-block:: python

   from isaaclab_newton.physics import NewtonManager

   model = NewtonManager.get_model()
   state = NewtonManager.get_state_0()
   control = NewtonManager.get_control()
   contacts = NewtonManager.get_contacts()

   body_poses = state.body_q
   joint_forces = control.joint_f

:meth:`isaaclab_newton.physics.NewtonManager.get_model` can construct a visualization shadow
model when PhysX is active. The write semantics in this guide apply only when Newton is the active,
authoritative physics backend.

Isaac Lab's Newton-backed assets expose the same generic
``newton.selection.ArticulationView`` selection helper. For example, reuse an articulation's
root selection instead of constructing the matching selection again:

.. code-block:: python

   robot = scene["robot"]
   selection = robot.root_view
   joint_positions = selection.get_dof_positions(state)

Newton uses this generic, label-based selection concept for Isaac Lab articulations, rigid objects,
rigid-object collections, and cables. It is a selection helper over model indices, not per-asset
typed storage.

Create raw access
-----------------

Code that owns a matching model can construct its own selection from a model and a label pattern:

.. code-block:: python

   from newton.selection import ArticulationView

   selection = ArticulationView(
       model,
       pattern="/World/envs/env_*/Robot",
   )

Read/write semantics
--------------------

Selections provide typed convenience methods as well as generic string-keyed
``get_attribute()`` and ``set_attribute()`` methods. The generic methods expose
engine properties that do not have dedicated selection methods. Clone a
selected value before modifying it when you want the write to remain explicit:

.. code-block:: python

   import warp as wp
   from newton import ModelFlags
   from isaaclab_newton.physics import NewtonManager

   rolling_friction = wp.clone(
       selection.get_attribute("shape_material_mu_rolling", model)
   )
   # Modify rolling_friction with a Warp kernel before writing it back.
   selection.set_attribute(
       "shape_material_mu_rolling",
       model,
       rolling_friction,
   )
   NewtonManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

The string names a Newton model attribute rather than an Isaac Lab field. This
example uses rolling friction because it has no dedicated selection method.
Notify the manager with the flag appropriate to the property changed; other
model writes can require a different flag. State writes that change generalized
coordinates instead require forward-kinematics synchronization through
``NewtonManager.invalidate_fk()`` and ``NewtonManager.forward()``.

Ownership, synchronization, and invalidation
---------------------------------------------

Direct writes bypass Isaac Lab caches and shape/order validation. Some Newton solvers swap current
and next state buffers, so reacquire
:meth:`isaaclab_newton.physics.NewtonManager.get_state_0` when code needs the current authoritative
state on a later step. A selection can survive state-buffer swaps because it describes model
indices, but it must be recreated after a model rebuild. Solver-specific generalized- and
maximal-coordinate conventions remain authoritative.

Authoritative references
------------------------

* `Model reference <https://newton-physics.github.io/newton/stable/api/_generated/newton.Model.html>`_
* `State reference <https://newton-physics.github.io/newton/stable/api/_generated/newton.State.html>`_
* `Control reference <https://newton-physics.github.io/newton/stable/api/_generated/newton.Control.html>`_
* `ArticulationView selection reference <https://newton-physics.github.io/newton/stable/api/_generated/newton.selection.ArticulationView.html>`_
* `Newton articulation guide <https://newton-physics.github.io/newton/stable/concepts/articulations.html>`_
