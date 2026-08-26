.. _concepts_sensors_contact:
.. _overview_sensors_contact:

.. currentmodule:: isaaclab

Contact Sensor
==============

A :class:`~sensors.ContactSensor` aggregates contacts on one or more rigid bodies into batched force
measurements. The sensor scope is defined by :attr:`~sensors.ContactSensorCfg.prim_path`: each matched
body becomes one sensor body in every environment.

.. figure:: ../../_static/overview/sensors/contact_diagram.jpg
   :align: center
   :figwidth: 100%
   :alt: A contact sensor reporting total and filtered contact forces

Net and filtered forces
-----------------------

:attr:`~sensors.ContactSensorData.net_forces_w` is the sum of normal contact forces acting on each
sensor body, expressed in the world frame. It includes contact with every body in the scene.

Set :attr:`~sensors.ContactSensorCfg.filter_prim_paths_expr` when the force from specific collision
partners is also needed. The resulting :attr:`~sensors.ContactSensorData.force_matrix_w` retains one
entry per filter expression while ``net_forces_w`` remains the unfiltered total.

Body-level filtering supports a many-to-one relationship: ``prim_path`` must resolve to one sensor
body per environment when filters are configured. Define one sensor per source body when separate
filtered forces are required, for example one sensor for each foot. Newton additionally supports
shape-level sensing and filtering through ``sensor_shape_prim_expr`` and
``filter_shape_prim_expr``.

Configure the sensor
--------------------

Add the configuration to an :class:`~isaaclab.scene.InteractiveSceneCfg`:

.. code-block:: python

   from isaaclab.scene import InteractiveSceneCfg
   from isaaclab.sensors import ContactSensorCfg


   class MySceneCfg(InteractiveSceneCfg):
       left_foot_contact = ContactSensorCfg(
           prim_path="{ENV_REGEX_NS}/Robot/LF_FOOT",
           update_period=0.0,
           history_length=6,
           filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
           track_air_time=True,
       )

``update_period=0.0`` samples every physics step. ``history_length`` stores earlier force samples.
Enable optional buffers only when they are needed. Their backend support differs:

* Isaac Sim PhysX supports pose, filtered contact-point, and filtered friction-force tracking.
* OvPhysX supports pose tracking for a single sensor body per environment, but not contact-point or
  friction-force tracking.
* Newton supports filtered contact-point tracking, but not pose or friction-force tracking.

Filtered contact points and friction forces require ``filter_prim_paths_expr``. Contact-rich Isaac
Sim PhysX scenes may require a larger ``max_contact_data_count_per_prim``.

Read the data
-------------

For ``E`` environments, ``S`` sensor bodies, ``F`` filter expressions, and history length ``H``, the
principal Torch views have these contracts:

.. list-table::
   :header-rows: 1
   :widths: 32 26 42

   * - Buffer
     - Shape
     - Meaning
   * - ``net_forces_w.torch``
     - ``(E, S, 3)``
     - Net normal contact force [N] in world frame
   * - ``net_forces_w_history.torch``
     - ``(E, H, S, 3)``
     - Net-force history [N]
   * - ``force_matrix_w.torch``
     - ``(E, S, F, 3)``
     - Normal force [N] from each filtered partner
   * - ``contact_pos_w.torch``
     - ``(E, S, F, 3)``
     - Average filtered contact position [m] in world frame; unavailable on OvPhysX
   * - ``friction_forces_w.torch``
     - ``(E, S, F, 3)``
     - Filtered friction force [N] in world frame; Isaac Sim PhysX only
   * - ``current_air_time.torch`` / ``current_contact_time.torch``
     - ``(E, S)``
     - Current mode duration [s]

Supported optional buffers are ``None`` unless their matching tracking option or filter is enabled.
Requesting an unsupported tracking option, or reading an unsupported data property, raises
``NotImplementedError``.

.. code-block:: python

   contact = scene["left_foot_contact"]
   net_force = contact.data.net_forces_w.torch
   object_force = contact.data.force_matrix_w.torch

Use ``debug_vis=True`` while validating body expressions and filter partners. The visualization shows
sensor contacts but does not change the reported data.

.. figure:: ../../_static/overview/sensors/contact_visualization.jpg
   :align: center
   :figwidth: 100%
   :alt: Contact sensor debug visualization

A complete runnable example is available in
``scripts/demos/sensors/contact_sensor.py``:

.. code-block:: bash

   uv run --extra isaacsim python scripts/demos/sensors/contact_sensor.py
