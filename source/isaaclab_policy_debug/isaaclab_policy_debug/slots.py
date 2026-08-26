# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


class SlotAllocator:
    """Allocate the lowest free fixed-capacity simulation world slot."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._owners: dict[int, object] = {}

    def allocate(self, owner: object) -> int:
        for slot in range(self.capacity):
            if slot not in self._owners:
                self._owners[slot] = owner
                return slot
        raise RuntimeError(f"Policy debug capacity is full ({self.capacity} active checkpoints)")

    def release(self, slot: int) -> None:
        self._owners.pop(slot, None)

    @property
    def active_count(self) -> int:
        return len(self._owners)
