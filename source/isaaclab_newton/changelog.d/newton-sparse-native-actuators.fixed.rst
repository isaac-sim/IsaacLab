* Fixed Newton-native actuators driving and resetting the wrong environments for an articulation that exists in only
  some environments: such an articulation now raises at initialization, pointing to ``use_newton_actuators=False``.
