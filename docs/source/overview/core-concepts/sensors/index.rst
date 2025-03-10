.. _overview_sensors:

Sensors
=========

In this section, we will overview the various sensor APIs provided by Isaac Lab.

Every sensor in Isaac Lab inherits from the ``SensorBase`` abstract class that provides the core functionality inherent to all sensors, which is to provide access to "measurements" of the scene. These measurements can take many forms such as ray-casting results, camera rendered images, or even simply ground truth data queried directly from the simulation (such as poses). Whatever the data may be, we can think of the sensor as having a buffer that is periodically updated with measurements by querying the scene. This ``update_period`` is defined in "simulated" seconds, meaning that even if the flow of time in the simulation is dilated relative to the real world, the sensor will update at the appropriate rate. The ``SensorBase`` is also designed with vectorizability in mind, holding the buffers for all copies of the sensor across cloned environments.

Updating the buffers is done by overriding the ``_update_buffers_impl`` abstract method of the ``SensorBase`` class. On every time-step of the simulation, ``dt``, all sensors are queried for an update. During this query, the total time since the last update is incremented by ``dt`` for every buffer managed by that particular sensor. If the total time is greater than or equal to the ``update_period`` for a buffer, then that buffer is flagged to be updated on the next query.

The following pages describe the available sensors in more detail:

.. toctree::
    :maxdepth: 1

    camera
    contact_sensor
    frame_transformer
    imu
    ray_caster
