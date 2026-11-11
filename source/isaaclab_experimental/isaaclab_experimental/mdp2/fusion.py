# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Code generation that fuses per-world Warp statements into a few generated kernels."""

from __future__ import annotations

import hashlib
import linecache
import math
import re
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from typing import Any

import warp as wp

from .plan import Launch

_ARGUMENT = re.compile(r"\ba\d+\b")  # Source names of kernel arguments, see Fusion.argument


@dataclass(frozen=True)
class FusedKernel:
    """A generated kernel, and how to read its launch arguments from a runtime call."""

    kernel: wp.Kernel
    dim: int
    arguments: tuple[Callable[[Any], Any], ...]
    device: str
    capturable: bool = True

    def __call__(self, call: Any) -> None:
        wp.launch(self.kernel, self.dim, [get(call) for get in self.arguments], device=self.device)


@dataclass
class ThreadAccess:
    """Kernel arguments that the statements of one kind of thread read or write."""

    touched: set[str] = field(default_factory=set)
    written: set[str] = field(default_factory=set)


class Fusion:
    """Fuses the statements of one runtime function into as few Warp kernels as data dependencies allow.

    Statements are source lines that refer to kernel arguments. A kernel runs them in two kinds of threads:

    - one world thread per world, which runs its statements in the order they were added;
    - element blocks, which run one statement per world and output element, one thread each. Threads are ordered by
      block, then world, then element, so neighboring threads run the same statement and write adjacent memory.

    All threads of a kernel run concurrently, so a new kernel starts whenever a statement would read or write an
    argument that another thread writes. Statements declare the arguments they write, except their own output
    elements. Hooks that launch their own work run between kernels: a hook that declares what it reads and writes
    runs before the statements of the current kernel if they do not depend on each other, so that consecutive hooks
    do not split it; any other hook ends the current kernel.
    """

    def __init__(self, name: str, num_worlds: int, device: str, mask: Callable[[Any], wp.array] | None = None):
        """If ``mask`` is given, it reads the worlds to run from each call, and threads of other worlds return."""
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

        # Arguments that the world thread and the element threads touch and write in the current kernel.
        self._world_access = ThreadAccess()
        self._element_access = ThreadAccess()

    def argument(self, key: Hashable, annotation: Any, get: Callable[[Any], Any]) -> str:
        """Return the source name of a kernel argument read with ``get(call)``. Each key maps to one argument."""
        if key not in self._arguments:
            self._arguments[key] = (f"a{len(self._arguments)}", annotation, get)
        return self._arguments[key][0]

    def constant(self, annotation: Any, value: Any) -> str:
        """Return the source name of an argument with a fixed value, so changing the value reuses the kernel."""
        return self.argument(object(), annotation, lambda call: value)

    def function(self, function: wp.Function) -> str:
        """Return the source name of a ``wp.func`` for use in statements."""
        name = self._functions.setdefault(function, f"f{len(self._functions)}")
        self._namespace[name] = function
        return name

    def world(self, *lines: str, writes: Iterable[Hashable] = ()) -> None:
        """Append a statement to the world thread. ``writes`` lists the keys of the arguments it writes."""
        self._join(self._world_access, [self._element_access], lines, writes)
        if self._world is None:
            self._world = []
            self._blocks.append(((), self._world))
        self._world += lines

    def elements(self, shape: tuple[int, ...], *lines: str, writes: Iterable[Hashable] = ()) -> None:
        """Append an element block that runs a statement with ``world``, indices ``i0, i1, ...``, and ``element``."""
        self._join(self._element_access, [self._world_access, self._element_access], lines, writes)
        self._blocks.append((shape, list(lines)))

    def call(
        self,
        step: Callable[[Any], None],
        capturable: bool = True,
        reads: Iterable[Hashable] | None = None,
        writes: Iterable[Hashable] | None = None,
    ) -> None:
        """Append a hook that launches its own work. ``reads`` and ``writes`` are the keys of the arguments it uses."""
        if reads is None or writes is None or self._depends(reads, writes):
            self._cut()
        self._steps.append(Launch(step, capturable))

    def steps(self) -> tuple[Callable[[Any], None], ...]:
        """Return the generated kernels and hooks in launch order."""
        self._cut()
        return tuple(self._steps)

    def _join(
        self, access: ThreadAccess, others: list[ThreadAccess], lines: Iterable[str], writes: Iterable[Hashable]
    ) -> None:
        """Record what a statement reads and writes, starting a new kernel first if it would race with ``others``."""
        touched = set(_ARGUMENT.findall("\n".join(lines)))
        written = {self._arguments[key][0] for key in writes}
        if any(touched & other.written or written & other.touched for other in others):
            self._cut()

        access.touched |= touched
        access.written |= written

    def _depends(self, reads: Iterable[Hashable], writes: Iterable[Hashable]) -> bool:
        """Return whether a hook and the statements of the current kernel read or write each other's arguments."""
        read = {self._arguments[key][0] for key in reads if key in self._arguments}
        written = {self._arguments[key][0] for key in writes if key in self._arguments}
        accesses = (self._world_access, self._element_access)
        return any(read & access.written or written & access.touched for access in accesses)

    def _cut(self) -> None:
        """Generate a kernel from the pending blocks, and start a new one."""
        if not self._blocks:
            return

        # Each block runs in a contiguous range of threads, ordered by world, then element.
        body, begin = [], 0
        for shape, lines in self._blocks:
            size = math.prod(shape)
            body += [
                f"{'elif' if begin else 'if'} thread < {begin + size} * worlds:",
                f"    local = thread - {begin} * worlds",
                f"    world = local // {size}",
                f"    element = local % {size}",
                *(
                    f"    i{d} = element // {math.prod(shape[d + 1 :])} % {dim_size}"
                    for d, dim_size in enumerate(shape)
                ),
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

        self._blocks, self._world = [], None
        for access in (self._world_access, self._element_access):
            access.touched.clear()
            access.written.clear()
