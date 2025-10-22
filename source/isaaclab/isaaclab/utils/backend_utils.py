import importlib

class FactoryBase:
    """A generic factory class that dynamically loads backends."""

    def __init_subclass__(cls, **kwargs):
        """Initializes a new factory subclass."""
        super().__init_subclass__(**kwargs)
        cls._registry = {}
        # Determine the module subpath for dynamic loading.
        # e.g., if factory is in 'isaaclab.assets.articulation.articulation',
        # the subpath becomes 'assets.articulation'.
        module_parts = cls.__module__.split('.')
        if module_parts[0] != 'isaaclab':
            raise ImportError(f"Factory class {cls.__name__} must be defined within the 'isaaclab' package.")
        # The subpath is what comes between 'isaaclab' and the final module name.
        cls._module_subpath = ".".join(module_parts[1:-1])

    @classmethod
    def register(cls, name: str, sub_class) -> None:
        """Register a new implementation class."""
        if name in cls._registry and cls._registry[name] is not sub_class:
            raise ValueError(f"Backend {name!r} already registered with a different class for factory {cls.__name__}.")
        cls._registry[name] = sub_class

    def __new__(cls, *args, **kwargs):
        """Create a new instance of an implementation based on the backend."""

        backend = "newton"

        if cls == FactoryBase:
            raise TypeError("FactoryBase cannot be instantiated directly. Please subclass it.")

        # If backend is not in registry, try to import it.
        if backend not in cls._registry:
            # Construct the module name from the backend and the determined subpath.
            module_name = f"isaaclab.{backend}.{cls._module_subpath}"
            try:
                importlib.import_module(module_name)
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
            # Suggest the specialized mixin name by convention (e.g., "RegisterableArticulation").
            registerable_mixin_name = f"Registerable{cls.__name__}"
            raise ValueError(
                f"Unknown backend {backend!r} for {cls.__name__}. "
                f"A module was found at '{module_name}', but it did not register an implementation.\n"
                f"Ensure the implementation class in that module inherits from the correct registerable mixin "
                f"(e.g., `{registerable_mixin_name}`) and has a `__backend_name__` attribute set to {backend!r}.\n"
                f"Currently available backends: {available}."
            ) from None
        # Return an instance of the chosen class.
        return impl(*args, **kwargs)

    @classmethod
    def get_registry_keys(cls) -> list[str]:
        """Returns a list of registered backend names."""
        return list(cls._registry.keys())

class Registerable:
    """A mixin class to make an implementation class registerable to a factory.

    The factory must be specified as a class attribute `__factory_class__`.
    The backend name must be specified as a class attribute `__backend_name__`.
    """
    def __init_subclass__(cls, **kwargs):
        """Register the class in the factory provided."""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "__factory_class__") and hasattr(cls, "__backend_name__"):
            factory_class = cls.__factory_class__
            backend_name = cls.__backend_name__
            print(f"Registering backend '{backend_name}' for factory '{factory_class.__name__}'.")
            factory_class.register(backend_name, cls)
