from __future__ import annotations
from .base_rigid_object_data import BaseRigidObjectData
from isaaclab.utils.backend_utils import FactoryBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData as NewtonRigidObjectData

class RigidObjectData(FactoryBase):
    """Factory for creating rigid object data instances."""

    def __new__(cls, *args, **kwargs) -> BaseRigidObjectData | NewtonRigidObjectData:
        """Create a new instance of a rigid object data based on the backend."""
        return super().__new__(cls, *args, **kwargs)