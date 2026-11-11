# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA graph capture of an MDP's runtime functions, with non-capturable launches run eagerly between the graphs."""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable, Sequence

import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

from .mdp import MDP, Backend, MDPState, Transition
from .plan import Call


class CapturedMDP:
    """The runtime functions of an MDP, bound to fixed buffers and replayed as CUDA graphs.

    Consecutive capturable launches of a function are captured as one graph, and eager terms and other
    non-capturable hooks run eagerly between the graphs. Capturing runs nothing. Before each call, write the actions
    and the mask of the worlds to reset or observe into the bound ``actions`` and ``mask`` buffers. On CPU, the
    launches run eagerly.
    """

    def __init__(self, mdp: MDP, state: MDPState, transition: Transition, actions: ProxyArray, mask: ProxyArray):
        self.mdp = mdp
        self.num_graphs = 0  # CUDA graphs replayed by one call of each runtime function, summed over the functions
        self._pool = torch.cuda.graph_pool_handle() if self._on_cuda and mdp.backend is Backend.TORCH else None
        launches = mdp.launches
        self._apply_actions = self._capture(launches.apply_actions, Call(state, actions=actions))
        self._evaluate = self._capture(
            launches.evaluate, Call(state, transition=transition, observations=transition.observations)
        )
        self._reset = self._capture(launches.reset, Call(state, mask=mask))
        self._observe = self._capture(launches.observe, Call(state, mask=mask, observations=transition.observations))

    def apply_actions(self) -> None:
        """Store the bound actions and write them to their target fields."""
        for run in self._apply_actions:
            run()

    def evaluate(self) -> None:
        """Compute terminations, rewards, and successor observations into the bound transition."""
        for run in self._evaluate:
            run()

    def reset(self) -> None:
        """Reset the worlds selected by the bound mask."""
        for run in self._reset:
            run()

    def observe(self) -> None:
        """Compute the observations of the worlds selected by the bound mask into the bound transition."""
        for run in self._observe:
            run()

    @property
    def _on_cuda(self) -> bool:
        return wp.get_device(self.mdp.device).is_cuda

    def _capture(self, launches: Sequence[Callable[[Call], None]], call: Call) -> list[Callable[[], None]]:
        """Capture each run of consecutive capturable launches as a graph, and bind the others to the call."""
        runs = []
        for capturable, group in itertools.groupby(launches, lambda launch: launch.capturable):
            group = list(group)
            if not (capturable and self._on_cuda):
                runs += [functools.partial(launch, call) for launch in group]
            elif self.mdp.backend is Backend.WARP:
                with wp.ScopedCapture(self.mdp.device) as capture:
                    for launch in group:
                        launch(call)
                runs.append(functools.partial(wp.capture_launch, capture.graph))
            else:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self._pool):
                    for launch in group:
                        launch(call)
                runs.append(graph.replay)
            self.num_graphs += capturable and self._on_cuda
        return runs
