# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declaration of weights a component loads at runtime."""

import glob
import os
from dataclasses import MISSING

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.configclass import configclass
from isaaclab.utils.io import latest_file


@configclass
class Checkpoint:
    """Weights a component loads at runtime, declared on the component's own config.

    Exactly one of :attr:`run_glob` and :attr:`url` is set:

    * ``run_glob`` -- this training run writes the file (a vision feature extractor trained with
      the policy). It is collected and published beside the policy checkpoint as
      ``<policy stem>_<name><extension>`` and fetched with it.
    * ``url`` -- the weights already exist (a frozen encoder, a low-level policy). They are only
      fetched, never published by the checkpoint tooling.

    The checkpoint tooling discovers declarations by walking the resolved environment config, so a
    task declares nothing; the component that writes or consumes the file owns its name.
    """

    name: str = MISSING
    """Identity of the file, used to publish it beside the policy checkpoint."""

    run_glob: str | None = None
    """Glob, relative to the training run directory, matching the file this run writes."""

    url: str | None = None
    """Published location of pre-existing weights."""

    @property
    def is_run_artifact(self) -> bool:
        """Whether this run produces the file, as opposed to fetching a published one."""
        return self.run_glob is not None

    @property
    def extension(self) -> str:
        """File extension, taken from whichever of :attr:`run_glob` or :attr:`url` is set."""
        return os.path.splitext(self.run_glob or self.url)[1]

    def find_in(self, run_dir: str) -> str | None:
        """Return the newest file in a training run matching :attr:`run_glob`, or ``None``.

        Args:
            run_dir: A training run's log directory.
        """
        if not self.is_run_artifact:
            return None
        return latest_file(run_dir, self.run_glob)

    def resolve(self, log_dir: str | None = None, cache_dir: str | None = None) -> str:
        """Return the local file a component should load.

        A run artifact is the published copy in :paramref:`log_dir` if one is present, else the
        newest file this run wrote there. Pre-existing weights are downloaded into
        :paramref:`cache_dir`. Neither convention is known to the caller.

        Args:
            log_dir: Directory the component reads from: the pretrained-checkpoint cache under
                ``--checkpoint pretrained``, the run's own log directory otherwise.
            cache_dir: Download directory for :attr:`url` weights. ``None`` uses the system
                temporary directory.

        Raises:
            FileNotFoundError: If no matching file is found in :paramref:`log_dir`.
            ValueError: If a run artifact is resolved without a :paramref:`log_dir`.
        """
        if not self.is_run_artifact:
            return retrieve_file_path(self.url, cache_dir)
        if log_dir is None:
            raise ValueError(f"Resolving the {self.name!r} checkpoint requires the directory it was written to.")
        # a published copy carries the suffix; a run writes the native name. They never share a
        # directory, so `or` selects the convention present rather than arbitrating between them.
        published = glob.glob(os.path.join(log_dir, f"*_{self.name}{self.extension}"))
        path = max(published, key=os.path.getmtime) if published else self.find_in(log_dir)
        if path is None:
            raise FileNotFoundError(
                f"No {self.name!r} checkpoint was found in '{log_dir}'. Train the task to produce one."
            )
        return path
