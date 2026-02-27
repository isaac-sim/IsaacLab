# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import logging

logger = logging.getLogger(__name__)


class FactoryBase:
    """A generic factory class that dynamically loads backends."""

    def __init_subclass__(cls, **kwargs):
        """Initializes a new factory subclass."""
        super().__init_subclass__(**kwargs)
        cls._registry = {}
        # Determine the module subpath for dynamic loading.
        # e.g., if factory is in 'isaaclab.assets.articulation.articulation',
        # the subpath becomes 'assets.articulation'.
        module_parts = cls.__module__.split(".")
        if module_parts[0] != "isaaclab":
            raise ImportError(f"Factory class {cls.__name__} must be defined within the 'isaaclab' package.")
        # The subpath is what comes between 'isaaclab' and the final module name.
        cls._module_subpath = ".".join(module_parts[1:-1])

    @classmethod
    def register(cls, name: str, sub_class) -> None:
        """Register a new implementation class."""
        if name in cls._registry and cls._registry[name] is not sub_class:
            raise ValueError(f"Backend {name!r} already registered with a different class for factory {cls.__name__}.")
        cls._registry[name] = sub_class
        logger.info(f"Registered backend {name!r} for factory {cls.__name__}.")

    @classmethod
    def _get_backend(cls, *args, **kwargs) -> str:
        """Return the backend name for this factory. Override in subclasses to dispatch by config."""
        return "physx"  # Backwards compatibility with old code.

    def __new__(cls, *args, **kwargs):
        """Create a new instance of an implementation based on the backend."""
        backend = cls._get_backend(*args, **kwargs)

        if cls == FactoryBase:
            raise TypeError("FactoryBase cannot be instantiated directly. Please subclass it.")

        # If backend is not in registry, try to import it and register the class.
        # This is done to only import the module once.
        if backend not in cls._registry:
            # Construct the module name from the backend and the determined subpath.
            module_name = f"isaaclab_{backend}.{cls._module_subpath}"
            try:
                module = importlib.import_module(module_name)
                module_class = getattr(module, cls.__name__)
                # Manually register the class
                cls.register(backend, module_class)

            except ImportError as e:
                raise ValueError(
                    f"Could not import module for backend {backend!r} for factory {cls.__name__}. "
                    f"Attempted to import from '{module_name}'.\n"
                    f"Original error: {e}"
                ) from e

        # Now check registry again. The import should have registered the class.
        try:
            impl = cls._registry[backend]
        except KeyError:
            available = list(cls.get_registry_keys())
            raise ValueError(
                f"Unknown backend {backend!r} for {cls.__name__}. "
                f"A module was found at '{module_name}', but it did not contain a class with the name {cls.__name__}.\n"
                f"Currently available backends: {available}."
            ) from None
        # Return an instance of the chosen class.
        return impl(*args, **kwargs)

    @classmethod
    def get_registry_keys(cls) -> list[str]:
        """Returns a list of registered backend names."""
        return list(cls._registry.keys())
