* Fixed callables nested under classes, such as ``collections:Counter.update`` or a config's staticmethods
  and classmethods, failing to resolve after serialization or a Hydra override.
