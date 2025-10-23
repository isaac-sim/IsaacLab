from __future__ import annotations
from .base_articulation import BaseArticulation
from isaaclab.utils.backend_utils import FactoryBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.newton.assets.articulation import Articulation as NewtonArticulation

class Articulation(FactoryBase):
    """Factory for creating articulation instances."""

    def __new__(cls, *args, **kwargs) -> BaseArticulation | NewtonArticulation:
        """Create a new instance of an articulation based on the backend."""
        # The `FactoryBase` __new__ method will handle the logic and return
        # an instance of the correct backend-specific articulation class,
        # which is guaranteed to be a subclass of `BaseArticulation` by convention.
        return super().__new__(cls, *args, **kwargs)
