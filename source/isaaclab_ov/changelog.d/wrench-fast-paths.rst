Changed
^^^^^^^

* Changed the external-wrench writers to submit through
  :meth:`~isaaclab.utils.wrench_composer.WrenchComposer.get_forces_and_torques`. A wrench that is already
  global-frame at the center of mass is now packed without rotating it into the body frame and back,
  and an all-local-frame wrench no longer reads the body transforms before packing.
