from .base_rigid_object_data import BaseRigidObjectData
from isaaclab.utils.backend_utils import FactoryBase

class RigidObjectData(FactoryBase):
    """Factory for creating rigid object data instances."""

    def __new__(cls, *args, **kwargs) -> BaseRigidObjectData:
        """Create a new instance of a rigid object data based on the backend."""
        return super().__new__(cls, *args, **kwargs)