# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

Newton Native Data and Selection API
====================================

Mental model
------------

The Newton backend exposes its engine-owned ``Model``, ``State``, ``Control``, and optional
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

Access live engine data
-----------------------

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

Reuse an Isaac Lab selection
----------------------------

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

Create a generic selection
--------------------------

Code that owns a matching model can construct its own selection from a model and a label pattern:

.. code-block:: python

   from newton.selection import ArticulationView

   selection = ArticulationView(
       model,
       pattern="/World/envs/env_*/Robot",
   )

Discover data and labels
------------------------

Discover the available labels, counts, state fields, and selection operations at runtime instead
of assuming a fixed set of arrays or methods:

.. code-block:: python

   print(model.articulation_label)
   print(model.body_label)
   print(model.joint_label)

   model_counts = {
       name: getattr(model, name)
       for name in dir(model)
       if name.endswith("_count") and isinstance(getattr(model, name), int)
   }
   selection_shape = {
       "instances": selection.count,
       "joint_dofs_per_instance": selection.joint_dof_count,
       "links_per_instance": selection.link_count,
   }
   state_fields = sorted(name for name in dir(state) if not name.startswith("_"))
   selection_methods = sorted(
       name
       for name in dir(selection)
       if name.startswith(("get_", "set_"))
   )
   print(model_counts)
   print(selection_shape)
   print(state_fields)
   print(selection_methods)

Read and write through a selection
----------------------------------

Read from a ``Model`` or ``State`` source, then write values to a ``Model`` or ``State`` target.
For the installed Newton version, ``ArticulationView.get_dof_positions()``
accepts a source and ``ArticulationView.set_dof_positions()`` accepts a target,
values, and an optional mask. Clone before modifying values so the intermediate changes are
explicit:

.. code-block:: python

   import warp as wp
   from isaaclab_newton.physics import NewtonManager

   joint_positions = wp.clone(selection.get_dof_positions(state))
   selection.set_dof_positions(state, joint_positions)
   NewtonManager.invalidate_fk()
   NewtonManager.forward()

Callers can modify the cloned Warp array before writing it through the selection. Because this
example writes manager-owned state, notify the manager with ``invalidate_fk()`` and then use its
solver-aware ``forward()`` path to propagate generalized-coordinate edits to body transforms. Raw
``newton.eval_fk`` remains appropriate for compatible standalone models and solvers outside the
manager-owned lifecycle; the exact synchronization requirement depends on the edited arrays and
active solver.

For model-property changes, notify the manager with the flag appropriate to the property changed.
For example, changing body inertial properties uses:

.. code-block:: python

   from newton import ModelFlags
   from isaaclab_newton.physics import NewtonManager

   NewtonManager.add_model_change(ModelFlags.BODY_INERTIAL_PROPERTIES)

This example does not imply that every model write uses the same flag.

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

`Newton API reference <https://newton-physics.github.io/newton/stable/api/newton.html>`_

`Newton articulation and selection guide <https://newton-physics.github.io/newton/stable/concepts/articulations.html>`_
