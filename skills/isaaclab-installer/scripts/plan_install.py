#!/usr/bin/env python3
"""plan_install.py — turn a (combo, system facts, user prefs) tuple into a
fully-resolved ordered list of shell commands. Does NOT execute anything.

Outputs a JSON plan that execute_install.py consumes.

Usage:
    python3 scripts/plan_install.py \
        --combo pip-uv-source \
        --preflight preflight.json \
        --env-name env_isaaclab \
        --isaaclab-dir $HOME/IsaacLab \
        --isaacsim-path $HOME/isaacsim \
        --output plan.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    colorize, load_combos, now_iso, print_header, print_info, print_ok,
    print_warn,
)


def _python_path_for_env(env_manager, isaaclab_dir, env_name, home):
    """Return the absolute path to the python binary inside the new env."""
    if env_manager == "uv":
        # uv venv is created at $ISAACLAB_DIR/$ENV_NAME OR $HOME/$ENV_NAME (pip-only combos)
        # We assume the cwd of the create_env step is canonical:
        # source combos: cwd=ISAACLAB_DIR; pip-only combos: cwd=HOME.
        # We'll prefer ISAACLAB_DIR when it exists in the combo, else HOME.
        return f"{isaaclab_dir}/{env_name}/bin/python"
    if env_manager == "conda":
        # We can't know CONDA_PREFIX up front without running conda; use a sentinel
        # path resolved at execute time via `conda run -n env --no-capture-output which python`.
        return f"$(conda run -n {env_name} --no-capture-output which python)"
    if env_manager == "venv":
        return f"{isaaclab_dir}/{env_name}/bin/python"
    if env_manager == "none":
        return "python3"
    return "python3"


def _pip_for_env(env_manager):
    if env_manager == "uv":
        return "uv pip"
    return "{ENV_PYTHON} -m pip"


def build_plan(combo_id, facts, env_name, isaaclab_dir, isaacsim_path, env_python_override=None):
    combos_mod = load_combos()
    combo = combos_mod.get_combo(combo_id)
    if not combo:
        raise SystemExit(f"Unknown combo id: {combo_id}")

    arch = facts["os"]["machine"]
    placeholders = {
        "ISAACLAB_DIR": str(isaaclab_dir),
        "ENV_NAME": env_name,
        "HOME": facts.get("home") or str(Path.home()),
        "ISAACSIM_PATH": str(isaacsim_path or ""),
        "ISAACSIM_VERSION": combos_mod.DEFAULT_ISAACSIM_VERSION,
        "TORCH_PIN": combos_mod.DEFAULT_TORCH_PIN,
        "TORCH_INDEX": combos_mod.torch_index_for_arch(arch),
        "PIP": _pip_for_env(combo["env_manager"]),
        "ENV_PYTHON": env_python_override or _python_path_for_env(
            combo["env_manager"], isaaclab_dir, env_name, facts.get("home")),
    }
    # Two-pass placeholder resolution: PIP references ENV_PYTHON.
    def fmt(s, depth=0):
        if not isinstance(s, str):
            return s
        prev = None
        cur = s
        # iterate up to 3x to resolve nested {ENV_PYTHON} inside {PIP}
        for _ in range(3):
            if prev == cur:
                break
            prev = cur
            try:
                cur = cur.format(**placeholders)
            except KeyError as e:
                raise SystemExit(f"Unresolved placeholder {e} in: {s}")
        return cur

    steps_out = []
    for raw_step in combo["steps"]:
        step = {
            "id": raw_step["id"],
            "title": fmt(raw_step["title"]),
            "cmd": fmt(raw_step["cmd"]),
            "cwd": fmt(raw_step.get("cwd", "{HOME}")),
            "requires_sudo": raw_step.get("requires_sudo", False),
            "on_failure": raw_step.get("on_failure", "abort"),
        }
        if raw_step.get("skip_if"):
            step["skip_if"] = fmt(raw_step["skip_if"])
        if raw_step.get("env"):
            step["env"] = {k: fmt(v) for k, v in raw_step["env"].items()}
        if raw_step.get("needs_auth"):
            step["needs_auth"] = raw_step["needs_auth"]
        if raw_step.get("notes"):
            step["notes"] = fmt(raw_step["notes"])
        if raw_step.get("manual_step"):
            step["manual_step"] = True
        steps_out.append(step)

    verify = {
        "cmd": fmt(combo["verify"]["cmd"]),
        "cwd": fmt(combo["verify"]["cwd"]),
        "headless_ok": combo["verify"].get("headless_ok", True),
    }

    return {
        "schema_version": 1,
        "generated_at": now_iso(),
        "combo_id": combo["id"],
        "combo_title": combo["title"],
        "placeholders": placeholders,
        "steps": steps_out,
        "verify": verify,
        "notes": combo.get("notes", []),
        "isaaclab_dir": str(isaaclab_dir),
        "isaacsim_path": str(isaacsim_path) if isaacsim_path else None,
        "env_name": env_name,
        "env_manager": combo["env_manager"],
        "isaaclab_source": combo["isaaclab_source"],
        "isaacsim_source": combo["isaacsim_source"],
    }


def print_plan(plan, stream=sys.stdout):
    w = stream.write
    w("\n")
    w(colorize("============================================================\n", "blue"))
    w(colorize(f" Installation Plan — {plan['combo_id']}\n", "bold"))
    w(colorize("============================================================\n", "blue"))
    w(f"  Title:       {plan['combo_title']}\n")
    w(f"  Env name:    {plan['env_name']}  ({plan['env_manager']})\n")
    w(f"  IsaacLab:    {plan['isaaclab_dir']}\n")
    if plan["isaacsim_path"]:
        w(f"  IsaacSim:    {plan['isaacsim_path']}\n")
    w(colorize("------------------------------------------------------------\n", "blue"))
    for i, s in enumerate(plan["steps"], 1):
        sudo = "  [sudo]" if s["requires_sudo"] else ""
        manual = "  [manual]" if s.get("manual_step") else ""
        w(f"  {i:2d}. {s['title']}{sudo}{manual}\n")
        w(f"      cwd: {s['cwd']}\n")
        w(f"      $ {s['cmd']}\n")
        if s.get("notes"):
            w(f"      note: {s['notes']}\n")
        if s.get("skip_if"):
            w(f"      skip-if: {s['skip_if']}\n")
    w(colorize("------------------------------------------------------------\n", "blue"))
    w(f"  Verify:      {plan['verify']['cmd']}\n")
    if plan["notes"]:
        w(colorize("\nNotes for this combo:\n", "yellow"))
        for n in plan["notes"]:
            w(f"  - {n}\n")
    w("\n")


def main(argv=None):
    p = argparse.ArgumentParser(description="Resolve a combo into an executable plan.")
    p.add_argument("--combo", required=True)
    p.add_argument("--preflight", required=True)
    p.add_argument("--env-name", default="env_isaaclab")
    p.add_argument("--isaaclab-dir", required=True)
    p.add_argument("--isaacsim-path", default=None,
                   help="Required only for binary/source Isaac Sim combos.")
    p.add_argument("--env-python", default=None,
                   help="Override the path used for ENV_PYTHON (advanced).")
    p.add_argument("-o", "--output", help="Write plan JSON to this path.")
    args = p.parse_args(argv)

    facts = json.loads(Path(args.preflight).read_text())
    plan = build_plan(
        combo_id=args.combo,
        facts=facts,
        env_name=args.env_name,
        isaaclab_dir=Path(args.isaaclab_dir).expanduser(),
        isaacsim_path=Path(args.isaacsim_path).expanduser() if args.isaacsim_path else None,
        env_python_override=args.env_python,
    )
    print_plan(plan, sys.stderr)
    text = json.dumps(plan, indent=2)
    if args.output:
        Path(args.output).write_text(text)
        print(f"[plan] wrote {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
