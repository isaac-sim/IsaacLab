# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed service locator for lifecycle-managed singletons."""

from __future__ import annotations

from typing import TypeVar

_T = TypeVar("_T")


def _try_close(service: object) -> None:
    """Call close() on *service* if it exists and is callable."""
    close = getattr(service, "close", None)
    if callable(close):
        close()


class ServiceLocator:
    """A typed service registry keyed by class, interface, or abstract base class.

    Services are registered and retrieved using subscript syntax::

        locator[FabricStageCache] = FabricStageCache(stage)
        cache = locator[FabricStageCache]

    Deleting a service calls ``close()`` on it if available::

        del locator[FabricStageCache]

    All registered services are closed and cleared via :meth:`close_all`.
    """

    def __init__(self) -> None:
        self._services: dict[type, object] = {}

    def __getitem__(self, cls: type[_T]) -> _T | None:
        """Retrieve a service by its key class, or ``None`` if not registered."""
        return self._services.get(cls)  # type: ignore[return-value]

    def __setitem__(self, cls: type[_T], instance: _T) -> None:
        """Register a service under the given key.

        The key can be the concrete class of *instance*, a parent class,
        or an abstract base class / protocol — allowing retrieval by
        interface rather than implementation.

        Does *not* close a previously registered service — the caller is
        responsible for closing the old instance before replacing it.
        Use ``del locator[cls]`` to close and remove, or :meth:`pop` to
        remove without closing.
        """
        self._services[cls] = instance

    def __delitem__(self, cls: type) -> None:
        """Close and remove a service.

        Calls ``close()`` on the instance if it has one, then removes it.

        Raises:
            KeyError: If no service is registered under *cls*.
        """
        instance = self._services.pop(cls)
        _try_close(instance)

    def __contains__(self, cls: type) -> bool:
        """Check if a service is registered under *cls*."""
        return cls in self._services

    def pop(self, cls: type[_T]) -> _T | None:
        """Remove and return a service without closing it.

        Returns:
            The previously registered instance, or ``None`` if not registered.
        """
        return self._services.pop(cls, None)  # type: ignore[return-value]

    def close_all(self, caught_exceptions: list[Exception]) -> None:
        """Close all registered services and clear the registry.

        Calls ``close()`` on each service that has one.  Exceptions are
        always collected into *caught_exceptions* — closing continues for
        all remaining services regardless of failures.

        Args:
            caught_exceptions: A list to which any exceptions raised by
                service ``close()`` calls are appended.
        """
        services = list(self._services.values())
        self._services.clear()
        for service in services:
            try:
                _try_close(service)
            except Exception as e:
                caught_exceptions.append(e)
