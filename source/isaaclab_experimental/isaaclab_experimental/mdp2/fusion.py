# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fusion of per-world statements into generated Warp kernels."""

from __future__ import annotations

import hashlib
import linecache
import math
import re
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Any

import warp as wp

_ARGUMENT = re.compile(r"\ba\d+\b")


@dataclass(frozen=True)
class FusedKernel:
    """A generated kernel. Calling it with a runtime call launches the kernel with arguments read from it."""

    kernel: wp.Kernel
    dim: int
    arguments: tuple[Callable[[Any], Any], ...]
    device: str

    def __call__(self, call: Any) -> None:
        wp.launch(self.kernel, self.dim, [get(call) for get in self.arguments], device=self.device)


class Fusion:
    """Fuses the statements of one runtime function into as few kernels as their data allows.

    Statements are source lines that name kernel arguments. A kernel runs them in two kinds of threads:

    - the world thread, one per world, runs its statements in the order they were added;
    - an element block runs one statement for every world and element of a shape, one thread each. Threads are
      ordered by block, world, then element, so neighbors run the same statement and write adjacent elements.

    All threads of a kernel run at once, and element threads only write outputs that nothing reads. A new kernel
    therefore starts wherever an element thread would read an argument that the world thread writes, and at
    every step that launches its own kernels.
    """

    def __init__(self, name: str, num_worlds: int, device: str, mask: Callable[[Any], wp.array] | None = None):
        """With ``mask``, which reads the boolean array of worlds to run for from a call, other worlds do nothing."""
        self.name = name
        self.num_worlds = num_worlds
        self.device = device
        self._steps: list[Callable[[Any], None]] = []
        self._namespace: dict[str, Any] = {"wp": wp}
        self._functions: dict[wp.Function, str] = {}
        self._arguments: dict[Hashable, tuple[str, Any, Callable[[Any], Any]]] = {}
        self._mask = None if mask is None else self.argument(mask, wp.array(dtype=wp.bool), mask)
        self._blocks: list[tuple[tuple[int, ...], list[str]]] = []
        self._world: list[str] | None = None
        self._written: set[str] = set()  # Arguments the world thread writes in the current kernel
        self._read: set[str] = set()  # Arguments the element threads read in the current kernel

    def argument(self, key: Hashable, annotation: Any, get: Callable[[Any], Any]) -> str:
        """Source name of the kernel argument whose value is ``get(call)``. A key names one argument."""
        if key not in self._arguments:
            self._arguments[key] = (f"a{len(self._arguments)}", annotation, get)
        return self._arguments[key][0]

    def constant(self, annotation: Any, value: Any) -> str:
        """Source name of a kernel argument with a fixed value. Kernels are reused when only values change."""
        return self.argument(object(), annotation, lambda call: value)

    def function(self, function: wp.Function) -> str:
        """Source name of a ``wp.func``."""
        name = self._functions.setdefault(function, f"f{len(self._functions)}")
        self._namespace[name] = function
        return name

    def world(self, *lines: str, writes: Iterable[Hashable] = ()) -> None:
        """Add a statement to the world thread. ``writes`` are the keys of the arguments it writes."""
        written = {self._arguments[key][0] for key in writes}
        if written & self._read:
            self._cut()
        if self._world is None:
            self._world = []
            self._blocks.append(((), self._world))
        self._world += lines
        self._written |= written

    def elements(self, shape: tuple[int, ...], *lines: str) -> None:
        """Add an element block: a statement for ``world``, the indices ``i0, i1, ...``, and the flat ``element``."""
        read = set(_ARGUMENT.findall("\n".join(lines)))
        if read & self._written:
            self._cut()
        self._blocks.append((shape, list(lines)))
        self._read |= read

    def call(self, step: Callable[[Any], None]) -> None:
        """Add a step that launches its own kernels, after the statements added so far."""
        self._cut()
        self._steps.append(step)

    def steps(self) -> tuple[Callable[[Any], None], ...]:
        """The generated kernels and the added steps, in order."""
        self._cut()
        return tuple(self._steps)

    def _cut(self) -> None:
        """Generate the kernel of the current blocks, and start a new one."""
        if not self._blocks:
            return
        body, begin = [], 0
        for shape, lines in self._blocks:
            size = math.prod(shape)
            body += [
                f"{'elif' if begin else 'if'} thread < {begin + size} * worlds:",
                f"    local = thread - {begin} * worlds",
                f"    world = local // {size}",
                f"    element = local % {size}",
                *(f"    i{d} = element // {math.prod(shape[d + 1 :])} % {n}" for d, n in enumerate(shape)),
                *([f"    if not {self._mask}[world]:", "        return"] if self._mask else []),
                *(f"    {line}" for line in lines),
            ]
            begin += size
        used = set(_ARGUMENT.findall("\n".join(body)))
        arguments = [argument for argument in self._arguments.values() if argument[0] in used]
        signature = ", ".join(["worlds", *(name for name, _, _ in arguments)])
        source = "\n".join(
            [f"def {self.name}({signature}):", "    thread = wp.tid()", *(f"    {line}" for line in body), ""]
        )
        # Warp reads kernel source through linecache.
        filename = f"<mdp2 {hashlib.sha256(source.encode()).hexdigest()[:16]}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        exec(compile(source, filename, "exec"), self._namespace)
        function = self._namespace[self.name]
        function.__annotations__ = {"worlds": int, **{name: annotation for name, annotation, _ in arguments}}
        num_worlds = self.num_worlds
        self._steps.append(
            FusedKernel(
                kernel=wp.kernel(function, module="unique"),
                dim=begin * num_worlds,
                arguments=(lambda call: num_worlds, *(get for _, _, get in arguments)),
                device=self.device,
            )
        )
        self._blocks, self._world, self._written, self._read = [], None, set(), set()
