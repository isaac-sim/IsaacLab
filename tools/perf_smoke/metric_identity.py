# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Identify the selected runtime FPS producer without importing simulation code."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

PACKAGE = Path("source/isaaclab/isaaclab/benchmark")


class _WithoutDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        node = super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
        return node


def _normalized(node: ast.AST) -> str:
    return ast.dump(_WithoutDocstrings().visit(copy.deepcopy(node)), include_attributes=False)


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one source definition of {name}.")
    return matches[0]


def _implementation(module: ast.Module, names: set[str]) -> tuple[dict[str, str], set[str]]:
    """Include selected functions and their same-module helpers and bindings."""
    definitions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
    for node in module.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                definitions[alias.asname or alias.name.split(".")[0]] = node
    result, referenced = {}, set()
    pending = set(names)
    while pending:
        name = pending.pop()
        if name in result:
            continue
        if name not in definitions:
            raise ValueError(f"Referenced FPS helper {name} has no identifiable source definition.")
        definition = definitions[name]
        result[name] = _normalized(definition)
        loaded = {
            node.id for node in ast.walk(definition) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        referenced.update(loaded)
        pending.update((loaded & definitions.keys()) - result.keys())
    return result, referenced


def metric_definition(checkout_root: Path) -> dict:
    """Fingerprint the runtime producer call and its FPS aggregation implementation.

    This is source equivalence metadata, not a claim that arbitrary code implements
    a named mathematical formula. Unknown call shapes retain an explanatory reason.
    """
    result = {"total_fps": None, "provenance": "producer_source_ast"}
    try:
        runtime_path, builder_path = PACKAGE / "entrypoints/runtime.py", PACKAGE / "builders.py"
        runtime = ast.parse((checkout_root / runtime_path).read_text(encoding="utf-8"))
        builders = ast.parse((checkout_root / builder_path).read_text(encoding="utf-8"))
        run = _function(runtime, "run")
        imports = [
            node
            for node in ast.walk(runtime)
            if isinstance(node, ast.ImportFrom)
            and node.module == "isaaclab.benchmark"
            and any(alias.name == "builders" and alias.asname in (None, "builders") for alias in node.names)
        ]
        if not imports or any(
            isinstance(node, ast.Name) and node.id == "builders" and isinstance(node.ctx, ast.Store)
            for node in ast.walk(runtime)
        ):
            raise ValueError("The runtime builders binding does not identify the selected benchmark builders module.")
        calls = [
            node
            for node in ast.walk(run)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "builders"
            and node.func.attr == "build_runtime"
        ]
        if len(calls) != 1:
            raise ValueError("The runtime producer does not have one identifiable builders.build_runtime call.")
        call = calls[0]
        if call.args or any(keyword.arg is None for keyword in call.keywords):
            raise ValueError("The runtime producer supplies positional or expanded arguments.")
        function = _function(builders, "build_runtime")
        settings = [keyword.value for keyword in call.keywords if keyword.arg == "aggregate_throughput"]
        if not settings:
            settings = [
                default
                for argument, default in zip(function.args.kwonlyargs, function.args.kw_defaults)
                if argument.arg == "aggregate_throughput"
            ]
        if len(settings) != 1 or not isinstance(settings[0], ast.Constant) or type(settings[0].value) is not bool:
            raise ValueError("The producer's aggregation setting is not a literal or identifiable boolean default.")
        aggregate = settings[0].value
        canonical_call = copy.deepcopy(call)
        canonical_call.keywords = [item for item in canonical_call.keywords if item.arg != "aggregate_throughput"]
        canonical_call.keywords.append(ast.keyword(arg="aggregate_throughput", value=ast.Constant(aggregate)))
        implementation, referenced = _implementation(builders, {"build_runtime"})
        inputs = {str(runtime_path): _normalized(canonical_call), str(builder_path): implementation}
        metric_names = {
            alias.name
            for node in builders.body
            if isinstance(node, ast.ImportFrom) and node.module == "metrics" and node.level == 1
            for alias in node.names
            if (alias.asname or alias.name) in referenced
        }
        if metric_names:
            metrics_path = PACKAGE / "metrics.py"
            metrics = ast.parse((checkout_root / metrics_path).read_text(encoding="utf-8"))
            inputs[str(metrics_path)] = _implementation(metrics, metric_names)[0]
        digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        result.update(
            total_fps="source-fps-v1:" + digest,
            aggregate_throughput=aggregate,
            source_paths=sorted(inputs),
            reason="Identity fingerprints the runtime producer call and selected FPS aggregation source.",
        )
    except (OSError, UnicodeError, SyntaxError, ValueError, TypeError) as exc:
        result["reason"] = f"FPS producer identity is unknown: {exc}"
    return result
