from .utils.lazy_importer import LazyImporter

# Lazily import heavy dependencies so importing ``isaaclab`` does not
# immediately import them. Currently only ``isaacsim`` is required.
isaacsim = LazyImporter("isaacsim")

__all__ = ["isaacsim"]
