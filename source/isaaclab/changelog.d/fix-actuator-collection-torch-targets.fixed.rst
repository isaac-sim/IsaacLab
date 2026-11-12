* Fixed ``ActuatorCollection`` target-command setters passing torch tensors straight to ``wp.launch``, which
  raised ``AttributeError: 'Tensor' object has no attribute 'ptr'`` during Warp graph capture. Torch values and
  index selectors are now wrapped as Warp arrays before launch.
