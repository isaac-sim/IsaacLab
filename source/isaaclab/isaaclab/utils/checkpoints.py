# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declaration of weights a component loads at runtime."""

import glob
import os
from dataclasses import MISSING, dataclass


@dataclass
class Checkpoint:
    """Weights this training run writes, declared on the component's own config.

    The component names the file it writes through :attr:`run_glob`, and :meth:`resolve` returns
    the file to load, so it never encodes where the weights come from. Tooling that copies the
    weights elsewhere records the copy in :attr:`local_path`, which :meth:`resolve` prefers;
    :mod:`isaaclab_rl.utils.pretrained_checkpoint` does this for the checkpoints published beside
    a policy.
    """

    name: str = MISSING
    """Identity of these weights among the declarations a task's components make."""

    run_glob: str = MISSING
    """Glob, relative to the training run directory, matching the file this run writes."""

    local_path: str | None = None
    """A copy already placed locally. Takes precedence in :meth:`resolve`."""

    def find_in(self, run_dir: str) -> str | None:
        """Return the newest file in a training run matching :attr:`run_glob`, or ``None``.

        Args:
            run_dir: A training run's log directory.
        """
        matches = glob.glob(os.path.join(run_dir, self.run_glob))
        return max(matches, key=os.path.getmtime) if matches else None

    def resolve(self, log_dir: str) -> str:
        """Return the local file a component should load.

        A copy recorded in :attr:`local_path` wins; otherwise it is the newest file this run
        wrote into :paramref:`log_dir`.

        Args:
            log_dir: The run directory the component writes to and reads from.

        Raises:
            FileNotFoundError: If no matching file is found in :paramref:`log_dir`.
        """
        if self.local_path is not None:
            return self.local_path
        path = self.find_in(log_dir)
        if path is None:
            raise FileNotFoundError(
                f"No {self.name!r} checkpoint was found in '{log_dir}'. Train the task to produce one."
            )
        return path
