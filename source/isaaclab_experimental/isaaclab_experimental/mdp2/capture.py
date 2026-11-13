# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA graph capture of an MDP's runtime functions, with non-capturable launches run eagerly between the graphs."""

from __future__ import annotations

import contextlib
import functools
import itertools
from collections.abc import Callable, Iterator, Sequence

import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

from .compiled import MDP, MDPState, Transition
from .plan import Call


@contextlib.contextmanager
def torch_in_warp_capture(device: str, pool: torch.cuda.MemPool) -> Iterator[None]:
    """Run Torch work on Warp's current stream of a CUDA device, so that a Warp graph capture records it.

    Torch allocates from ``pool`` meanwhile: the graph keeps using the addresses it recorded, so their memory must
    stay out of Torch's shared cache for as long as the graph lives. Torch's own random generators repeat their
    recorded draws on replay, so Torch work in a graph draws from MDP2's streams instead.
    """
    with torch.cuda.stream(wp.stream_to_torch(wp.get_stream(device))), torch.cuda.use_mem_pool(pool):
        yield


class CapturedMDP:
    """The runtime functions of an MDP, bound to fixed buffers and replayed as CUDA graphs.

    Consecutive capturable launches of a function are captured as one graph, and eager terms and other
    non-capturable hooks run eagerly between the graphs. Capturing runs nothing. Before each call, write the actions
    and the mask of the worlds to reset or observe into the bound ``actions`` and ``mask`` buffers. On CPU, the
    launches run eagerly.
    """

    def __init__(
        self,
        mdp: MDP,
        state: MDPState,
        transition: Transition,
        actions: ProxyArray,
        mask: ProxyArray,
        commit: Callable[[], None] | None = None,
    ):
        self.mdp = mdp
        self.num_graphs = 0  # CUDA graphs replayed by one call of each runtime function, summed over the functions
        self._on_cuda = wp.get_device(mdp.device).is_cuda
        self._pool = torch.cuda.MemPool() if self._on_cuda else None

        launches = mdp.launches
        self._apply_actions = self._capture(launches.apply_actions, Call(state, actions=actions))
        self._evaluate = self._capture(
            launches.evaluate, Call(state, transition=transition, observations=transition.observations)
        )
        self._reset = self._capture(launches.reset, Call(state, mask=mask, commit=commit))
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

    def _capture(self, launches: Sequence[Callable[[Call], None]], call: Call) -> list[Callable[[], None]]:
        """Capture each run of consecutive capturable launches as a graph, and bind the others to the call."""
        runs = []
        for capturable, group in itertools.groupby(launches, lambda launch: launch.capturable):
            group = list(group)
            if not (capturable and self._on_cuda):
                runs += [functools.partial(launch, call) for launch in group]
                continue

            # Torch launches, of either backend, join the Warp capture.
            with torch_in_warp_capture(self.mdp.device, self._pool), wp.ScopedCapture(self.mdp.device) as capture:
                for launch in group:
                    launch(call)
            runs.append(functools.partial(wp.capture_launch, capture.graph))
            self.num_graphs += 1

        return runs
