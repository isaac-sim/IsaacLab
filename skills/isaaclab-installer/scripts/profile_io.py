#!/usr/bin/env python3
"""profile_io.py — read / write / reproduce install_profile.yaml.

Subcommands:
    show      — print the active profile in a readable form.
    locate    — print the path of the active profile.
    redact    — print the profile with paths/usernames redacted (safe to attach to issues).
    reproduce — build a plan from an existing profile so a teammate can rerun it.

Used by the SKILL.md flows and by the user when sharing profiles.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    DEFAULT_PROFILE_PATH, colorize, print_header, print_info, print_ok,
    print_warn, read_yaml, write_yaml,
)


def _redact_string(s, home, user):
    if not isinstance(s, str):
        return s
    if home:
        s = s.replace(home, "~")
    if user:
        s = s.replace(user, "<user>")
    return s


def _redact(obj, home, user):
    if isinstance(obj, dict):
        return {k: _redact(v, home, user) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v, home, user) for v in obj]
    if isinstance(obj, str):
        return _redact_string(obj, home, user)
    return obj


def cmd_show(args):
    profile = read_yaml(args.profile)
    print_header(f"Install profile: {args.profile}")
    if not profile:
        print_warn("Profile is empty.")
        return 1
    print(json.dumps(profile, indent=2, default=str))
    return 0


def cmd_locate(args):
    path = Path(args.profile)
    if path.is_file():
        print(str(path))
        return 0
    print(f"# No profile at {path}", file=sys.stderr)
    return 1


def cmd_redact(args):
    profile = read_yaml(args.profile)
    home = str(Path.home())
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    redacted = _redact(profile, home, user)
    if args.output:
        write_yaml(args.output, redacted)
        print_ok(f"Redacted profile written to {args.output}")
        print_info("Safe to attach to a GitHub issue.")
    else:
        # Always print to stdout as JSON for unambiguous parsing
        print(json.dumps(redacted, indent=2, default=str))
    return 0


def cmd_reproduce(args):
    """Build a plan + (optionally) execute it from a teammate's profile.

    Maps the saved profile back into the inputs needed by plan_install.py.
    """
    profile = read_yaml(args.profile)
    if not profile:
        print(f"Could not read profile: {args.profile}", file=sys.stderr)
        return 1
    combo = profile.get("combo_id")
    if not combo:
        print("Profile is missing combo_id.", file=sys.stderr)
        return 1
    paths = profile.get("paths") or {}
    print_header("Reproducing install from profile")
    print_info(f"Combo:        {combo}")
    print_info(f"Env manager:  {(profile.get('install_method') or {}).get('env_manager')}")
    print_info(f"IsaacLab dir: {paths.get('isaaclab_dir')}")
    if paths.get("isaacsim_path"):
        print_info(f"IsaacSim:     {paths.get('isaacsim_path')}")
    print()
    print_info("To regenerate the install plan, run:")
    print()
    env_name = (profile.get('install_method') or {}).get('env_name', 'env_isaaclab')
    isaaclab_dir = paths.get('isaaclab_dir') or "$HOME/IsaacLab"
    parts = [
        "python3 scripts/plan_install.py",
        f"  --combo {combo}",
        "  --preflight preflight.json",
        f"  --env-name {env_name}",
        f"  --isaaclab-dir {isaaclab_dir}",
    ]
    if paths.get("isaacsim_path"):
        parts.append(f"  --isaacsim-path {paths['isaacsim_path']}")
    parts.append("  --output plan.json")
    print("  " + " \\\n  ".join(parts))
    print()
    print_info("Then execute it:")
    print("  python3 scripts/execute_install.py --plan plan.json")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Inspect or replay Isaac Lab install profiles.")
    p.add_argument("--profile", default=str(DEFAULT_PROFILE_PATH))
    sub = p.add_subparsers(dest="action", required=True)

    sub.add_parser("show", help="Print the active profile.")
    sub.add_parser("locate", help="Print the path of the active profile.")
    r = sub.add_parser("redact", help="Print a copy of the profile with personal paths removed.")
    r.add_argument("-o", "--output", help="Optional output path; otherwise prints to stdout.")
    sub.add_parser("reproduce", help="Print commands to rerun this install on another machine.")

    args = p.parse_args(argv)
    return {
        "show": cmd_show,
        "locate": cmd_locate,
        "redact": cmd_redact,
        "reproduce": cmd_reproduce,
    }[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
