# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declaration of weights a component loads at runtime."""

import glob
import os
from dataclasses import MISSING, dataclass

from isaaclab.utils.assets import retrieve_file_path


@dataclass
class Checkpoint:
    """Weights a component loads at runtime, declared on the component's own config.

    Exactly one of :attr:`run_glob` and :attr:`url` is set:

    * ``run_glob`` -- this training run writes the file, such as a vision feature extractor
      trained alongside the policy.
    * ``url`` -- the weights already exist, such as a frozen encoder or a low-level policy.

    :meth:`resolve` returns the file to load, so a component never encodes where its weights come
    from. Tooling that copies weights elsewhere records the copy in :attr:`local_path`, which
    :meth:`resolve` prefers; :mod:`isaaclab_rl.utils.pretrained_checkpoint` does this for the
    checkpoints published beside a policy.
    """

    name: str = MISSING
    """Identity of these weights among the declarations a task's components make."""

    run_glob: str | None = None
    """Glob, relative to the training run directory, matching the file this run writes."""

    url: str | None = None
    """Published location of pre-existing weights."""

    local_path: str | None = None
    """A copy already placed locally. Takes precedence in :meth:`resolve`."""

    def __post_init__(self) -> None:
        if (self.run_glob is None) == (self.url is None):
            raise ValueError(
                f"The {self.name!r} checkpoint must declare exactly one of run_glob and url,"
                f" got run_glob={self.run_glob!r} and url={self.url!r}."
            )

    @property
    def is_run_artifact(self) -> bool:
        """Whether this run produces the file, as opposed to fetching a published one."""
        return self.run_glob is not None

    def find_in(self, run_dir: str) -> str | None:
        """Return the newest file in a training run matching :attr:`run_glob`, or ``None``.

        Args:
            run_dir: A training run's log directory.
        """
        if not self.is_run_artifact:
            return None
        matches = glob.glob(os.path.join(run_dir, self.run_glob))
        return max(matches, key=os.path.getmtime) if matches else None

    def resolve(self, log_dir: str | None = None, cache_dir: str | None = None) -> str:
        """Return the local file a component should load.

        A fetched copy (:attr:`local_path`) wins. Otherwise a run artifact is the newest file this
        run wrote into :paramref:`log_dir`, and pre-existing weights are downloaded into
        :paramref:`cache_dir`.

        Args:
            log_dir: The run directory the component writes to and reads from.
            cache_dir: Download directory for :attr:`url` weights. ``None`` uses the system
                temporary directory.

        Raises:
            FileNotFoundError: If no matching file is found in :paramref:`log_dir`.
            ValueError: If a run artifact is resolved without a :paramref:`log_dir`.
        """
        if self.local_path is not None:
            return self.local_path
        if not self.is_run_artifact:
            return retrieve_file_path(self.url, cache_dir)
        if log_dir is None:
            raise ValueError(f"Resolving the {self.name!r} checkpoint requires the directory it was written to.")
        path = self.find_in(log_dir)
        if path is None:
            raise FileNotFoundError(
                f"No {self.name!r} checkpoint was found in '{log_dir}'. Train the task to produce one."
            )
        return path
