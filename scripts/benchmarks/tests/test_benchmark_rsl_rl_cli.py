# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CLI-level tests for benchmark_rsl_rl.py.

These tests exercise only the argparse layer — they do not import the
whole script (which launches Isaac Sim at import time). A minimal reimport
of the argparse setup is shared via ``_build_parser``.
"""

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    """Mirror of the parser setup in benchmark_rsl_rl.py.

    Kept in lockstep with the script; when a new flag is added there,
    add it here too.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--backend", choices=["physx", "newton"], default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--schema_v1_output", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    return parser


def test_log_dir_flag_defaults_none():
    args = _build_parser().parse_args([])
    assert args.log_dir is None


def test_log_dir_flag_captured():
    args = _build_parser().parse_args(["--log_dir", "/tmp/bundle/training_data"])
    assert args.log_dir == "/tmp/bundle/training_data"


def _inject_preset(args_cli, hydra_args: list[str]) -> list[str]:
    """Mirror of the inject_preset logic in benchmark_rsl_rl.py.

    Invariant: when --backend X is set AND hydra_args does NOT already
    contain a ``presets=...`` entry, prepend ``presets=X``.
    """
    if args_cli.backend is None:
        return hydra_args
    existing = [a for a in hydra_args if a.startswith("presets=")]
    if existing:
        print(f"[WARNING] --backend={args_cli.backend} ignored; explicit {existing[0]} wins.")
        return hydra_args
    return [f"presets={args_cli.backend}"] + hydra_args


def test_backend_injects_preset_when_none_given():
    args = _build_parser().parse_args(["--backend", "newton"])
    out = _inject_preset(args, ["env.decimation=4"])
    assert out == ["presets=newton", "env.decimation=4"]


def test_backend_does_not_inject_when_preset_already_present(capsys):
    args = _build_parser().parse_args(["--backend", "newton"])
    out = _inject_preset(args, ["presets=custom", "env.decimation=4"])
    assert out == ["presets=custom", "env.decimation=4"]
    assert "ignored" in capsys.readouterr().out


def test_backend_unset_is_noop():
    args = _build_parser().parse_args([])
    out = _inject_preset(args, ["env.decimation=4"])
    assert out == ["env.decimation=4"]


def _inject_preset_with_validation(args_cli, hydra_args: list[str], has_physics_preset_fn) -> list[str]:
    """Mirror of the new gated injection in benchmark_rsl_rl.py.

    has_physics_preset_fn is the only injection point — the test passes
    a stub returning True / False; the production caller passes the real
    has_physics_preset(raw_cfg, name) closure.
    """
    import sys

    if args_cli.backend is None:
        return hydra_args
    existing = [a for a in hydra_args if a.startswith("presets=")]
    if existing:
        print(f"[WARNING] --backend={args_cli.backend} ignored; explicit {existing[0]} wins.")
        return hydra_args
    if not has_physics_preset_fn(args_cli.backend):
        sys.stderr.write(
            f"[ERROR] preset_unsupported: task {args_cli.task!r} has no "
            f"{args_cli.backend!r} preset. Inspect raw_cfg.sim.physics or "
            f"re-enumerate {{physx,newton}}_envs.yaml.\n"
        )
        sys.exit(2)
    return [f"presets={args_cli.backend}"] + hydra_args


def test_validation_blocks_unsupported_preset(capsys):
    args = _build_parser().parse_args(["--task", "Isaac-Foo-v0", "--backend", "physx"])
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        _inject_preset_with_validation(args, ["env.x=1"], has_physics_preset_fn=lambda name: False)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "preset_unsupported:" in captured.err
    assert "Isaac-Foo-v0" in captured.err


def test_validation_passes_when_supported():
    args = _build_parser().parse_args(["--task", "Isaac-Bar-v0", "--backend", "newton"])
    out = _inject_preset_with_validation(args, ["env.x=1"], has_physics_preset_fn=lambda name: True)
    assert out == ["presets=newton", "env.x=1"]


def test_validation_skipped_when_explicit_preset_present(capsys):
    """Explicit presets= in hydra_args bypasses validation (operator override)."""
    args = _build_parser().parse_args(["--task", "Isaac-Foo-v0", "--backend", "physx"])

    def _bomb(name: str) -> bool:
        raise AssertionError("validator must not run when explicit preset is present")

    out = _inject_preset_with_validation(args, ["presets=custom", "env.x=1"], has_physics_preset_fn=_bomb)
    assert out == ["presets=custom", "env.x=1"]
    assert "ignored" in capsys.readouterr().out


def _inject_preset_with_validation_v2(
    args_cli,
    hydra_args: list[str],
    has_physics_preset_fn,
    native_backend_matches_fn,
) -> list[str]:
    """Mirror of the new gated injection in benchmark_rsl_rl.py (post native-backend fix).

    Two stub injection points:
      - has_physics_preset_fn(name) -> bool (existing)
      - native_backend_matches_fn(name) -> bool (new)
    """
    import sys

    if args_cli.backend is None:
        return hydra_args
    existing = [a for a in hydra_args if a.startswith("presets=")]
    if existing:
        print(f"[WARNING] --backend={args_cli.backend} ignored; explicit {existing[0]} wins.")
        return hydra_args
    if has_physics_preset_fn(args_cli.backend):
        return [f"presets={args_cli.backend}"] + hydra_args
    if native_backend_matches_fn(args_cli.backend):
        print(
            f"[INFO] task {args_cli.task!r} has no '{args_cli.backend}' preset; "
            f"running on native {args_cli.backend} backend (no injection).",
            file=sys.stderr,
        )
        return hydra_args
    sys.stderr.write(
        f"[ERROR] preset_unsupported: task {args_cli.task!r} has no "
        f"{args_cli.backend!r} preset. Inspect raw_cfg.sim.physics or "
        f"re-enumerate {{physx,newton}}_envs.yaml.\n"
    )
    raise SystemExit(2)


def test_validation_skips_injection_when_native_matches(capsys):
    """No preset, but cfg type matches request → run with no injection + [INFO] log."""
    args = _build_parser().parse_args(["--task", "Isaac-Quadcopter-Direct-v0", "--backend", "physx"])
    out = _inject_preset_with_validation_v2(
        args,
        ["env.x=1"],
        has_physics_preset_fn=lambda name: False,
        native_backend_matches_fn=lambda name: True,
    )
    assert out == ["env.x=1"]
    captured = capsys.readouterr()
    assert "running on native physx" in captured.err
    assert "no injection" in captured.err


def test_validation_still_blocks_when_native_mismatches(capsys):
    """No preset AND cfg type doesn't match → existing exit-2 + preset_unsupported: stderr (regression)."""
    args = _build_parser().parse_args(["--task", "Isaac-NewtonOnly-v0", "--backend", "physx"])
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        _inject_preset_with_validation_v2(
            args,
            ["env.x=1"],
            has_physics_preset_fn=lambda name: False,
            native_backend_matches_fn=lambda name: False,
        )
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "preset_unsupported:" in captured.err
