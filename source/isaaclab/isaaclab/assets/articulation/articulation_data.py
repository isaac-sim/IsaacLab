from .base_articulation_data import BaseArticulationData
from isaaclab.utils.backend_utils import FactoryBase, Registerable

class ArticulationData(FactoryBase):
    """Factory for creating articulation data instances."""

    def __new__(cls, *args, **kwargs) -> BaseArticulationData:
        """Create a new instance of an articulation data based on the backend."""
        return super().__new__(cls, *args, **kwargs)

class RegisterableArticulationData(Registerable):
    """A mixin to register an articulation data with the ArticulationData factory."""
    __factory_class__ = ArticulationData