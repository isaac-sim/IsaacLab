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


def _normalized(node: ast.AST) -> str:
    node = copy.deepcopy(node)
    for item in ast.walk(node):
        # Ignore Python 3.12's empty type_params so fingerprints match older interpreters.
        if getattr(item, "type_params", None) == []:
            del item.type_params
        if isinstance(item, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                item.body
                and isinstance(item.body[0], ast.Expr)
                and isinstance(item.body[0].value, ast.Constant)
                and isinstance(item.body[0].value.value, str)
            ):
                item.body.pop(0)
    options = {"show_empty": True} if "show_empty" in inspect.signature(ast.dump).parameters else {}
    return ast.dump(node, include_attributes=False, **options)


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


def _runtime_fps_value(node: ast.Return) -> ast.AST:
    value = node.value
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "Runtime":
        fields = [keyword.value for keyword in value.keywords if keyword.arg == "total_fps"]
        if len(fields) != 1:
            raise ValueError("The runtime builder does not identify one total_fps result field.")
        return fields[0]
    raise ValueError("The early return does not identify a Runtime FPS result.")


def _normalize_early_returns(body: list[ast.stmt], result: str, continuation: list[ast.stmt]) -> list[ast.stmt]:
    """Keep subsequent calculations only on paths that do not return early."""
    normalized = []
    body = [*body, *continuation]
    for index, node in enumerate(body):
        if isinstance(node, ast.Return):
            value = _runtime_fps_value(node)
            if any(
                result in _reads(statement) - set().union(*(_writes(item) for item in ast.walk(statement)))
                for statement in body[index + 1 :]
            ):
                raise ValueError("An early FPS result is used again after its return.")
            if not (isinstance(value, ast.Name) and value.id == result):
                normalized.append(
                    ast.copy_location(ast.Assign(targets=[ast.Name(id=result, ctx=ast.Store())], value=value), node)
                )
            break
        if isinstance(node, ast.If) and any(isinstance(item, ast.Return) for item in ast.walk(node)):
            left = _normalize_early_returns(node.body, result, body[index + 1 :])
            right = _normalize_early_returns(node.orelse, result, body[index + 1 :])
            left, _ = _fps_statements(left, {result})
            right, _ = _fps_statements(right, {result})
            test = node.test
            if not left and right:
                test = (
                    test.operand
                    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
                    else ast.UnaryOp(op=ast.Not(), operand=test)
                )
                left, right = right, []
            if left or right:
                normalized.append(ast.copy_location(ast.If(test=test, body=left, orelse=right), node))
            break
        if any(isinstance(item, ast.Return) for item in ast.walk(node)):
            raise ValueError("The FPS builder has an early return in unsupported control flow.")
        normalized.append(node)
    return normalized


def _builder_fps(function: ast.FunctionDef) -> tuple[list[ast.stmt], set[str]]:
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    if not returns or not function.body or function.body[-1] not in returns:
        raise ValueError("The FPS builder does not have one final result expression.")
    final = function.body[-1]
    value = final.value
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "Runtime":
        value = _runtime_fps_value(final)
    if value is None:
        raise ValueError("The runtime builder returns no FPS result.")
    body = function.body[:-1]
    if len(returns) > 1:
        if not isinstance(value, ast.Name):
            raise ValueError("The early FPS returns do not share an identifiable final result local.")
        body = _normalize_early_returns(body, value.id, [])
    statements, needed = _fps_statements(body, _reads(value))
    return [*statements, ast.Return(value=value)], needed


def _substitute(node: ast.AST, bindings: dict[str, ast.AST]) -> ast.AST:
    """Replace loaded names while preserving local expression scopes."""
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in bindings:
        return copy.deepcopy(bindings[node.id])
    node = copy.copy(node)
    if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        raise ValueError("The producer helper introduces an unresolved expression scope.")
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            setattr(node, field, _substitute(value, bindings))
        elif isinstance(value, list):
            setattr(node, field, [_substitute(item, bindings) if isinstance(item, ast.AST) else item for item in value])
    return node


def _keyword_mapping(
    value: ast.AST, owner: ast.FunctionDef, before: ast.AST, runtime: ast.Module
) -> dict[str, ast.AST]:
    """Read explicit keyword mappings without executing benchmark code."""
    if isinstance(value, ast.Name):
        writes = [node for node in ast.walk(owner) if value.id in _writes(node) and node.lineno < before.lineno]
        if len(writes) != 1 or not isinstance(writes[0], (ast.Assign, ast.AnnAssign)):
            raise ValueError("The producer's expanded keyword mapping is not an unmodified assignment.")
        assignment = writes[0]
        target = assignment.targets[0] if isinstance(assignment, ast.Assign) else assignment.target
        if not isinstance(target, ast.Name) or target.id != value.id:
            raise ValueError("The producer's expanded keyword mapping is not a direct assignment.")
        parents = {id(child): parent for parent in ast.walk(owner) for child in ast.iter_child_nodes(parent)}
        parent = parents[id(assignment)]
        if parent is not owner and not (parent.lineno <= before.lineno <= parent.end_lineno):
            raise ValueError("The producer's expanded keyword mapping is conditional.")
        if any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == value.id
            and (assignment.lineno, assignment.col_offset)
            < (node.lineno, node.col_offset)
            <= (before.lineno, before.col_offset)
            and (node.lineno, node.col_offset) != (value.lineno, value.col_offset)
            for node in ast.walk(owner)
        ):
            raise ValueError("The expanded keyword mapping is also used outside the selected call.")
        result = _keyword_mapping(assignment.value, owner, assignment, runtime)
        if any(
            isinstance(node, (ast.Call, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom))
            for item in result.values()
            for node in ast.walk(item)
        ):
            raise ValueError("An expanded keyword input has an unresolved evaluation order.")
        if any(not isinstance(item, (ast.Name, ast.Constant)) for item in result.values()) and any(
            isinstance(node, ast.stmt) and assignment.end_lineno < node.lineno < before.lineno
            for node in ast.walk(owner)
        ):
            raise ValueError("An expanded keyword expression was captured before intervening statements.")
        names = set().union(*(_reads(item) for item in result.values()))
        if any(_writes(node) & names and assignment.lineno < node.lineno < before.lineno for node in ast.walk(owner)):
            raise ValueError("An expanded keyword input was reassigned after the mapping was built.")
        return result
    if isinstance(value, ast.Dict):
        result = {}
        for key, item in zip(value.keys, value.values):
            if key is None:
                result.update(_keyword_mapping(item, owner, before, runtime))
            elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                result[key.value] = item
            else:
                raise ValueError("The producer's expanded keywords do not have literal names.")
        return result
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "dict" and not value.args:
        if _bindings(owner, "dict") or _bindings(runtime, "dict"):
            raise ValueError("The expanded mapping does not use the built-in dict constructor.")
        return _call_keywords(value, owner, runtime)
    raise ValueError("The producer's expanded keyword mapping cannot be resolved from source.")


def _call_keywords(call: ast.Call, owner: ast.FunctionDef, runtime: ast.Module) -> dict[str, ast.AST]:
    result = {}
    for keyword in call.keywords:
        values = (
            {keyword.arg: keyword.value}
            if keyword.arg is not None
            else _keyword_mapping(keyword.value, owner, call, runtime)
        )
        if result.keys() & values.keys():
            raise ValueError("The runtime producer supplies duplicate keyword arguments.")
        result.update(values)
    return result


def _forwarding_calls(
    call: ast.Call, owner: ast.FunctionDef, runtime: ast.Module, seen: tuple[str, ...] = ()
) -> list[tuple[ast.FunctionDef, ast.Call]] | None:
    """Find forwarding helpers before attempting to resolve their inputs."""
    if isinstance(call.func, ast.Attribute) and call.func.attr == "build_runtime":
        if isinstance(call.func.value, ast.Name):
            name = call.func.value.id
            if name == "builders" and _benchmark_binding(owner, name, runtime):
                return []
            bindings = _bindings(owner, name) or _bindings(runtime, name)
            if bindings and all(
                (
                    isinstance(node, ast.ImportFrom)
                    and node.module is not None
                    and not node.module.startswith("isaaclab.benchmark")
                    and node.level == 0
                )
                or (
                    isinstance(node, ast.Import)
                    and all(
                        not alias.name.startswith("isaaclab.benchmark")
                        for alias in node.names
                        if (alias.asname or alias.name.split(".")[0]) == name
                    )
                )
                for node in bindings
            ):
                return None
        raise ValueError("A build_runtime call does not identify the selected benchmark builder.")
    if not isinstance(call.func, ast.Name) or call.func.id in seen:
        return None
    functions = {node.name: node for node in runtime.body if isinstance(node, ast.FunctionDef)}
    helper = functions.get(call.func.id)
    if helper is None:
        bindings = _bindings(owner, call.func.id) or _bindings(runtime, call.func.id)
        if any(
            isinstance(node, ast.ImportFrom)
            and any(
                alias.name == "build_runtime" and (alias.asname or alias.name) == call.func.id for alias in node.names
            )
            and (node.level or (node.module or "").startswith("isaaclab.benchmark"))
            for node in bindings
        ):
            raise ValueError("An imported runtime producer cannot be resolved at its call site.")
        return None
    body = [
        node
        for node in helper.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
        and not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        )
    ]
    forwarded = (
        body[0].value
        if len(body) == 1 and isinstance(body[0], ast.Return) and isinstance(body[0].value, ast.Call)
        else None
    )
    chain = None if forwarded is None else _forwarding_calls(forwarded, helper, runtime, (*seen, helper.name))
    if chain is None:
        if any(
            _forwarding_calls(node, helper, runtime, (*seen, helper.name)) is not None
            for node in _producer_nodes(helper, runtime, (*seen, helper.name))
        ):
            raise ValueError("The runtime producer helper does more than forward its inputs.")
        return None
    if any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and any((alias.asname or alias.name) in _reads(forwarded) for alias in node.names)
        and not (isinstance(node, ast.ImportFrom) and node.module == "isaaclab.benchmark")
        for node in helper.body
    ):
        raise ValueError("The producer helper's imports cannot be resolved at its call site.")
    if helper.decorator_list or _bindings(runtime, helper.name) != [helper] or _bindings(owner, helper.name):
        raise ValueError("The producer helper is decorated, reassigned or shadowed at its call site.")
    return [(helper, forwarded), *chain]


def _producer_call(call: ast.Call, owner: ast.FunctionDef, runtime: ast.Module) -> ast.Call | None:
    chain = _forwarding_calls(call, owner, runtime)
    if chain is None:
        return None
    location = call
    if any(isinstance(node, ast.Starred) for node in call.args):
        raise ValueError("The producer helper supplies unresolved positional arguments.")
    for helper, forwarded in chain:
        arguments = helper.args
        positional = [*arguments.posonlyargs, *arguments.args]
        parameters = []
        defaults = (
            dict(zip([arg.arg for arg in positional][-len(arguments.defaults) :], arguments.defaults))
            if arguments.defaults
            else {}
        )
        defaults.update(
            {arg.arg: value for arg, value in zip(arguments.kwonlyargs, arguments.kw_defaults) if value is not None}
        )
        for group, kind in (
            (arguments.posonlyargs, inspect.Parameter.POSITIONAL_ONLY),
            (arguments.args, inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ([arguments.vararg] if arguments.vararg else [], inspect.Parameter.VAR_POSITIONAL),
            (arguments.kwonlyargs, inspect.Parameter.KEYWORD_ONLY),
            ([arguments.kwarg] if arguments.kwarg else [], inspect.Parameter.VAR_KEYWORD),
        ):
            parameters.extend(
                inspect.Parameter(arg.arg, kind, default=defaults.get(arg.arg, inspect.Parameter.empty))
                for arg in group
            )
        keywords = _call_keywords(call, owner, runtime)
        inputs = [*call.args, *keywords.values()]
        bound = inspect.Signature(parameters).bind(*call.args, **keywords)
        if any(not isinstance(value, ast.Constant) for name, value in defaults.items() if name not in bound.arguments):
            raise ValueError("The producer helper uses a default that cannot be resolved at its call site.")
        parameters_names = {parameter.name for parameter in parameters}
        if any(_bindings(owner, name) for name in _reads(forwarded) - parameters_names - {"builders"}):
            raise ValueError("A producer helper's global input is shadowed at its call site.")
        bound.apply_defaults()
        bindings = dict(bound.arguments)
        if arguments.vararg:
            raise ValueError("The producer helper forwards unresolved positional arguments.")
        if arguments.kwarg:
            values = bindings[arguments.kwarg.arg]
            bindings[arguments.kwarg.arg] = ast.Dict(
                keys=[ast.Constant(name) for name in values], values=list(values.values())
            )
        if any(
            isinstance(node, (ast.Call, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom))
            for value in bindings.values()
            for node in ast.walk(value)
        ):
            raise ValueError("A producer helper argument has an unresolved evaluation order.")
        forwarded_inputs = [*forwarded.args]
        for keyword in forwarded.keywords:
            if keyword.arg is None and not isinstance(keyword.value, ast.Name):
                forwarded_inputs.extend(_keyword_mapping(keyword.value, helper, forwarded, runtime).values())
            else:
                forwarded_inputs.append(keyword.value)
        if any(not isinstance(value, (ast.Name, ast.Constant)) for value in forwarded_inputs):
            raise ValueError("The producer helper does more than pass through captured inputs.")
        call = ast.copy_location(_substitute(forwarded, bindings), location)
        if any(not isinstance(value, (ast.Name, ast.Constant)) for value in inputs):
            expanded = [*call.args, *_call_keywords(call, owner, runtime).values()]
            if [
                ast.dump(value, include_attributes=True) for value in inputs if not isinstance(value, ast.Constant)
            ] != [
                ast.dump(value, include_attributes=True) for value in expanded if not isinstance(value, ast.Constant)
            ]:
                raise ValueError("The producer helper changes the evaluation order or count of captured inputs.")
    if call.args:
        raise ValueError("The runtime producer supplies positional arguments.")
    return ast.copy_location(
        ast.Call(
            func=call.func,
            args=[],
            keywords=[
                ast.keyword(arg=name, value=value) for name, value in _call_keywords(call, owner, runtime).items()
            ],
        ),
        location,
    )


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


def _scope_nodes(scope: ast.AST):
    """Visit one lexical scope without borrowing bindings from nested functions."""
    for node in ast.iter_child_nodes(scope):
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            yield from _scope_nodes(node)


def _bindings(scope: ast.AST, name: str) -> list[ast.AST]:
    return [
        node
        for node in _scope_nodes(scope)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id == name
        or isinstance(node, ast.arg)
        and node.arg == name
        or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == name
        or isinstance(node, (ast.Import, ast.ImportFrom))
        and any((alias.asname or alias.name.split(".")[0]) == name for alias in node.names)
    ]


def _benchmark_binding(scope: ast.AST, name: str, runtime: ast.Module) -> bool:
    bindings = _bindings(scope, name) or _bindings(runtime, name)
    return bool(bindings) and all(
        isinstance(node, ast.ImportFrom)
        and node.module == "isaaclab.benchmark"
        and any(alias.name == name and alias.asname in (None, name) for alias in node.names)
        for node in bindings
    )


def _producer_nodes(scope: ast.AST, runtime: ast.Module, seen: tuple[str, ...] = ()) -> list[ast.Call]:
    """Keep nested producers visible without resolving their inputs in the caller's scope."""
    scoped = set(_scope_nodes(scope))
    parents = {child: parent for parent in ast.walk(scope) for child in ast.iter_child_nodes(parent)}
    calls = []
    for node in ast.walk(scope):
        if not isinstance(node, ast.Call):
            continue
        if node in scoped:
            calls.append(node)
            continue
        owner = parents[node]
        while not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            owner = parents[owner]
        if _forwarding_calls(node, owner, runtime, seen) is not None or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "build_runtime"
            and isinstance(node.func.value, ast.Name)
            and not _bindings(owner, node.func.value.id)
        ):
            raise ValueError("A runtime producer is defined in a nested scope.")
    return calls


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
        calls = [
            producer
            for node in _producer_nodes(run, runtime)
            if (producer := _producer_call(node, run, runtime)) is not None
        ]
        if len(calls) != 1:
            raise ValueError("The runtime producer does not have one identifiable builders.build_runtime call.")
        call = calls[0]
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
            if not _benchmark_binding(run, "stepping", runtime):
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
