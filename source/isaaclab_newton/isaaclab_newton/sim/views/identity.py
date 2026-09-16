# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Concrete body identities from Newton's existing articulation selection layout."""

from newton import Model
from newton.selection import ArticulationView


def body_paths(view: ArticulationView, model: Model, env_index: int) -> list[str]:
    """Return selected body labels in public view order for one environment."""
    layout = view.frequency_layouts[Model.AttributeFrequency.BODY]
    selected = range(layout.slice.start, layout.slice.stop) if layout.slice is not None else layout.indices.numpy()
    # The layout stores body offsets within each match, not absolute model indices.
    return [
        model.body_label[
            layout.offset + env_index * layout.stride_between_worlds + match * layout.stride_within_worlds + int(body)
        ]
        for match in range(view.count_per_world)
        for body in selected
    ]
