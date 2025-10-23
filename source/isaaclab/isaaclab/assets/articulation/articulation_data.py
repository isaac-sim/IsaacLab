from __future__ import annotations
from .base_articulation_data import BaseArticulationData
from isaaclab.utils.backend_utils import FactoryBase
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.newton.assets.articulation.articulation_data import ArticulationData as NewtonArticulationData

class ArticulationData(FactoryBase):
    """Factory for creating articulation data instances."""

    def __new__(cls, *args, **kwargs) -> BaseArticulationData | NewtonArticulationData:
        """Create a new instance of an articulation data based on the backend."""
        return super().__new__(cls, *args, **kwargs)