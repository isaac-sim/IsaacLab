# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Own the detached OVStage required by OVPhysX."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence

import ovstage


class OvstageBackend:
    """Own the detached stage required by OVPhysX."""

    def __init__(self):
        self.resources = contextlib.ExitStack()
        self.reset()

    def populate(self, usda: str) -> None:
        """Import the serialized physics scene before OVPhysX attaches."""
        if self.ordinal:
            return
        ovstage.population.open_usd_from_string(self.stage, usda, ordinal=1, domains=ovstage.PopulationDomain.ALL)
        self.ordinal = 1
        self.commit()

    @contextlib.contextmanager
    def query(self, prim_paths: Sequence[str]) -> Iterator[ovstage.Query]:
        """Keep a physics attribute query and its path list alive together."""
        paths = self.paths.create_path_list_from_strings(prim_paths)
        try:
            with self.stage.query_from_path_list(paths) as query:
                yield query
        finally:
            self.paths.destroy_path_list(paths)

    def commit(self) -> int:
        """Seal physics edits at the current ordinal and advance to the next."""
        ordinal = self.ordinal
        self.stage.advance_write_floor(ordinal=ordinal).wait()
        self.ordinal += 1
        return ordinal

    def close(self) -> None:
        """Release the stage after its physics consumer detaches."""
        self.resources.close()
        self.stage = self.paths = None

    def reset(self) -> None:
        """Recreate the stage for a fresh physics scene."""
        self.close()
        self.ordinal = 0
        self.resources = contextlib.ExitStack()
        config = ovstage.StageConfig(
            runtime_default_hierarchy_computation_model=ovstage.HierarchyComputationModel.GPU_INCREMENTAL
        )
        self.stage = self.resources.enter_context(ovstage.Stage("isaaclab.physics", config=config))
        self.paths = self.resources.enter_context(ovstage.PathDictionary(self.stage))
