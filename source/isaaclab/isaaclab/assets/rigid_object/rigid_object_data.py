from .base_rigid_object_data import BaseRigidObjectData
from isaaclab.utils.backend_utils import FactoryBase, Registerable

class RigidObjectData(FactoryBase):
    """Factory for creating rigid object data instances."""

    def __new__(cls, *args, **kwargs) -> BaseRigidObjectData:
        """Create a new instance of a rigid object data based on the backend."""
        return super().__new__(cls, *args, **kwargs)

class RegisterableRigidObjectData(Registerable):
    """A mixin to register a rigid object data with the RigidObjectData factory."""
    __factory_class__ = RigidObjectData