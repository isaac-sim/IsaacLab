# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Identify the selected runtime FPS producer without importing simulation code."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

PACKAGE = Path("source/isaaclab/isaaclab/benchmark")


class _WithoutDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        node = super().generic_visit(node)
        # Python 3.12 adds this empty field to non-generic definitions. Its
        # absence on older interpreters describes the same producer source.
        if getattr(node, "type_params", None) == []:
            del node.type_params
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
    options = {"show_empty": True} if "show_empty" in inspect.signature(ast.dump).parameters else {}
    return ast.dump(_WithoutDocstrings().visit(copy.deepcopy(node)), include_attributes=False, **options)


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one source definition of {name}.")
    return matches[0]


def _definitions(module: ast.Module) -> dict[str, ast.AST]:
    definitions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
    for node in module.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                selected = copy.deepcopy(node)
                selected.names = [alias]
                definitions[alias.asname or alias.name.split(".")[0]] = selected
    return definitions


def _implementation(module: ast.Module, names: set[str]) -> tuple[dict[str, str], set[str]]:
    """Include selected functions and their same-module helpers and bindings."""
    definitions = _definitions(module)
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


def _reads(node: ast.AST) -> set[str]:
    names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)}
    # Comprehension iteration variables are local to their expression.
    local = {
        item.id
        for generator in ast.walk(node)
        if isinstance(generator, ast.comprehension)
        for item in ast.walk(generator.target)
        if isinstance(item, ast.Name)
    }
    return names - local


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        return _target_names(node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(item) for item in node.elts))
    return set()


def _writes(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Assign):
        return set().union(*(_target_names(target) for target in node.targets))
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return _target_names(node.target)
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
        return _target_names(node.value.func.value)
    return set()


def _fps_statements(body: list[ast.stmt], needed: set[str]) -> tuple[list[ast.stmt], set[str]]:
    """Select the current builder's assignments and branches feeding total FPS."""
    selected = []
    needed = set(needed)
    for node in reversed(body):
        if isinstance(node, ast.If):
            left, before_left = _fps_statements(node.body, needed)
            right, before_right = _fps_statements(node.orelse, needed)
            if left or right:
                selected.append(ast.If(test=node.test, body=left or [ast.Pass()], orelse=right))
                needed = before_left | before_right | _reads(node.test)
        elif _writes(node) & needed:
            selected.append(node)
            if (isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) for target in node.targets)) or (
                isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
            ):
                needed -= _writes(node)
            needed |= _reads(node)
        elif isinstance(node, (ast.For, ast.While, ast.Try, ast.With, ast.Match)):
            if any(_writes(item) & needed for item in ast.walk(node)):
                raise ValueError("The FPS builder uses an unrecognized control-flow assignment.")
    return list(reversed(selected)), needed


def _builder_fps(function: ast.FunctionDef) -> tuple[list[ast.stmt], set[str]]:
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    if len(returns) != 1 or not function.body or function.body[-1] is not returns[0]:
        raise ValueError("The FPS builder does not have one final result expression.")
    value = returns[0].value
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "Runtime":
        fields = [keyword.value for keyword in value.keywords if keyword.arg == "total_fps"]
        if len(fields) != 1:
            raise ValueError("The runtime builder does not identify one total_fps result field.")
        value = fields[0]
    if value is None:
        raise ValueError("The runtime builder returns no FPS result.")
    statements, needed = _fps_statements(function.body[:-1], _reads(value))
    return [*statements, ast.Return(value=value)], needed


def _runtime_inputs(run: ast.FunctionDef, call: ast.Call, expressions: list[ast.AST]) -> list[ast.AST]:
    """Follow FPS input assignments, stopping at the measured environment and CLI inputs."""
    assignments = [
        node for node in ast.walk(run) if _writes(node) and getattr(node, "lineno", call.lineno) < call.lineno
    ]
    parents = {id(child): parent for parent in ast.walk(run) for child in ast.iter_child_nodes(parent)}
    selected = {}
    pending = set().union(*(_reads(expression) for expression in expressions))
    visited = {"env", "args"}
    while pending - visited:
        name = (pending - visited).pop()
        visited.add(name)
        for node in assignments:
            if name in _writes(node):
                value = node
                child = node
                while (parent := parents.get(id(child))) is not run:
                    if isinstance(parent, ast.If):
                        pending.update(_reads(parent.test))
                        value = ast.If(
                            test=parent.test,
                            body=[value] if child in parent.body else [ast.Pass()],
                            orelse=[value] if child in parent.orelse else [],
                        )
                    elif isinstance(parent, (ast.For, ast.While, ast.Try, ast.FunctionDef, ast.AsyncFunctionDef)):
                        raise ValueError("The runtime FPS input uses an unrecognized control-flow assignment.")
                    if parent is None:
                        break
                    child = parent
                selected[id(node)] = (node.lineno, node.col_offset, value)
                pending.update(_reads(node))
    return [value for _, _, value in sorted(selected.values(), key=lambda item: item[:2])]


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
        statements, required = _builder_fps(function)
        arguments = {argument.arg for argument in (*function.args.args, *function.args.kwonlyargs)}
        defaults = {
            argument.arg: value
            for argument, value in zip(function.args.kwonlyargs, function.args.kw_defaults)
            if value is not None
        }
        supplied = {**defaults, **{keyword.arg: keyword.value for keyword in call.keywords}}
        supplied["aggregate_throughput"] = ast.Constant(aggregate)
        if not (required & arguments) <= supplied.keys():
            raise ValueError("The runtime producer does not supply every FPS input.")
        fps_inputs = {name: supplied[name] for name in sorted(required & arguments)}
        fps_body = ast.Module(body=statements, type_ignores=[])
        implementation, referenced = _implementation(builders, _reads(fps_body) & _definitions(builders).keys())
        referenced.update(_reads(fps_body))
        runtime_nodes = _runtime_inputs(run, call, list(fps_inputs.values()))
        runtime_body = ast.Module(body=runtime_nodes, type_ignores=[])
        runtime_references = _reads(runtime_body) | set().union(*(_reads(value) for value in fps_inputs.values()))
        runtime_helpers, _ = _implementation(runtime, runtime_references & _definitions(runtime).keys())
        inputs = {
            str(runtime_path): {
                "arguments": {name: _normalized(value) for name, value in fps_inputs.items()},
                "input_assignments": _normalized(runtime_body),
                "helpers": runtime_helpers,
            },
            str(builder_path): {"fps": _normalized(fps_body), "helpers": implementation},
        }
        timing_calls = {
            node.func.attr
            for expression in (*fps_inputs.values(), *runtime_nodes)
            for node in ast.walk(expression)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "stepping"
        }
        if timing_calls:
            if not any(
                isinstance(node, ast.ImportFrom)
                and node.module == "isaaclab.benchmark"
                and any(alias.name == "stepping" and alias.asname in (None, "stepping") for alias in node.names)
                for node in ast.walk(runtime)
            ) or any(
                isinstance(node, ast.Name) and node.id == "stepping" and isinstance(node.ctx, ast.Store)
                for node in ast.walk(runtime)
            ):
                raise ValueError("The FPS timing input does not identify the selected benchmark stepping module.")
            stepping_path = PACKAGE / "stepping.py"
            stepping = ast.parse((checkout_root / stepping_path).read_text(encoding="utf-8"))
            inputs[str(stepping_path)] = _implementation(stepping, timing_calls)[0]
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
            total_fps="source-fps-v2:" + digest,
            aggregate_throughput=aggregate,
            source_paths=sorted(inputs),
            reason="Identity fingerprints FPS aggregation, its runtime inputs and the selected timing producer.",
        )
    except (OSError, UnicodeError, SyntaxError, ValueError, TypeError) as exc:
        result["reason"] = f"FPS producer identity is unknown: {exc}"
    return result
