from .base_rigid_object import BaseRigidObject
from isaaclab.utils.backend_utils import FactoryBase

class RigidObject(FactoryBase):
    """Factory for creating articulation instances."""

    def __new__(cls, *args, **kwargs) -> BaseRigidObject:
        """Create a new instance of an articulation based on the backend."""
        # The `FactoryBase` __new__ method will handle the logic and return
        # an instance of the correct backend-specific articulation class,
        # which is guaranteed to be a subclass of `BaseArticulation` by convention.
        return super().__new__(cls, *args, **kwargs)