# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-agnostic data collector for manager-term live plots."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaaclab.managers import ManagerBase

logger = logging.getLogger(__name__)

# Maximum number of per-term components to log as individual scalars.
# Terms with more components are silently skipped in scalar backends.
_MAX_SCALAR_COMPONENTS = 32


class ManagerLivePlots:
    """Collects per-step scalar data from a :class:`~isaaclab.managers.ManagerBase` for live plotting.

    This class is backend-agnostic: it calls
    :meth:`~isaaclab.managers.ManagerBase.get_active_iterable_terms` each step and returns the
    results as plain Python lists of floats.  Visualizer backends (Newton, Rerun, Viser, Kit) use
    the collected data through their own rendering APIs.

    Example usage by a backend::

        source = ManagerLivePlots("reward_manager", env.reward_manager)
        for term_name, values in source.collect(env_idx=0).items():
            for i, v in enumerate(values):
                viewer.log_scalar(f"reward_manager/{term_name}[{i}]", v)
    """

    def __init__(
        self,
        manager_name: str,
        manager: ManagerBase,
        term_names: list[str] | None = None,
    ):
        """Initialize the data collector.

        Args:
            manager_name: Human-readable label used as the plot namespace prefix.
            manager: The manager whose terms to collect.
            term_names: Optional allowlist of term names to include.  ``None`` collects all terms.
        """
        self.manager_name = manager_name
        self._manager = manager
        self._allowed: set[str] | None = set(term_names) if term_names is not None else None

    def collect(self, env_idx: int) -> dict[str, list[float]]:
        """Pull current term values from the manager for one environment.

        Only terms whose values are scalar or low-dimensional vectors (up to
        :data:`_MAX_SCALAR_COMPONENTS` elements) are returned.  Image-shaped data (3-D+) is
        omitted and should be handled separately with :meth:`collect_images`.

        Args:
            env_idx: Environment index to sample.

        Returns:
            Mapping from term name to a list of float values sampled this step.
        """
        result: dict[str, list[float]] = {}
        try:
            for term_name, values in self._manager.get_active_iterable_terms(env_idx=env_idx):
                if self._allowed is not None and term_name not in self._allowed:
                    continue
                arr = np.asarray(values, dtype=np.float32)
                if arr.ndim > 2:
                    continue
                flat = arr.flatten()
                if flat.size > _MAX_SCALAR_COMPONENTS:
                    continue
                result[term_name] = [float(v) for v in flat]
        except Exception:
            logger.debug(
                "[ManagerLivePlots] Failed to collect terms from %s.", self.manager_name, exc_info=True
            )
        return result

    def collect_images(self, env_idx: int) -> dict[str, np.ndarray]:
        """Pull image-shaped term values (3-D+) from the manager.

        Args:
            env_idx: Environment index to sample.

        Returns:
            Mapping from term name to a NumPy array with shape ``(H, W[, C])``.
        """
        result: dict[str, np.ndarray] = {}
        try:
            for term_name, values in self._manager.get_active_iterable_terms(env_idx=env_idx):
                if self._allowed is not None and term_name not in self._allowed:
                    continue
                arr = np.asarray(values)
                if arr.ndim >= 3:
                    result[term_name] = arr
        except Exception:
            logger.debug(
                "[ManagerLivePlots] Failed to collect images from %s.", self.manager_name, exc_info=True
            )
        return result
