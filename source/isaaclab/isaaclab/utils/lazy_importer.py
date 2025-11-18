import importlib
from typing import Any


class LazyImporter:
    """Lightweight proxy that imports a module on first use.

    This is intended for heavy, optional dependencies (for example ``isaacsim``)
    so importing :mod:`isaaclab` does not eagerly import them.

    Example
    -------
    .. code-block:: python

        isaacsim = LazyImporter("isaacsim")
        app = isaacsim.SimulationApp(...)
    """

    def __init__(self, module_name: str, module: Any | None = None):
        # ``module`` is accepted for backwards compatibility with the old API
        # but ignored by the new implementation.
        self._module_name = module_name
        self._module: Any | None = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return self._module

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - thin wrapper
        return getattr(self._load(), item)

    def __repr__(self) -> str:
        return f"<LazyImporter {self._module_name!r}>"
