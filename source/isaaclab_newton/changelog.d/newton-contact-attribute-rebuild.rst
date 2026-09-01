Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.physics.NewtonManager` dropping requested extended contact
  attributes when the Newton model is finalized more than once. The pending attribute set was
  cleared after the first forward, so a later rebuild allocated a :class:`newton.Contacts` buffer
  without ``force`` and :meth:`newton.sensors.SensorContact.update` raised on the first step for
  any environment using a contact sensor with ``use_mujoco_contacts=False``.
* Fixed :meth:`~isaaclab_newton.physics.NewtonManager.add_contact_sensor` not recording its
  ``force`` request with the manager, which left the existing re-request path dormant.
