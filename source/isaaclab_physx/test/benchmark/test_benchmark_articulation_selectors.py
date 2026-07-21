# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-safe checks for articulation benchmark selector generation."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_generator_functions() -> dict:
    """Load only pure input-generator functions without starting the simulator."""
    benchmark_path = Path(__file__).parents[2] / "benchmark" / "assets" / "benchmark_articulation.py"
    tree = ast.parse(benchmark_path.read_text(), filename=str(benchmark_path))
    function_names = {
        "make_tensor_env_ids",
        "make_tensor_body_ids",
        "gen_set_external_force_and_torque_torch_tensor",
        "_make_tensor_dtype_generator",
    }
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in function_names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), benchmark_path, "exec"), namespace)
    return namespace


def test_external_wrench_int64_mode_varies_body_ids_only() -> None:
    """Keep external environment IDs int32 while exercising WrenchComposer int64 body IDs."""
    functions = _load_generator_functions()
    generator = functions["_make_tensor_dtype_generator"](
        functions["gen_set_external_force_and_torque_torch_tensor"], torch.int64
    )

    inputs = generator(SimpleNamespace(num_instances=3, num_bodies=2, device="cpu"))

    assert inputs["env_ids"].dtype == torch.int32
    assert inputs["body_ids"].dtype == torch.int64
    assert inputs["env_ids"].tolist() == [0, 1, 2]
    assert inputs["body_ids"].tolist() == [0, 1]
