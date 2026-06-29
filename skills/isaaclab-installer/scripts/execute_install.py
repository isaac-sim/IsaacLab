#!/usr/bin/env python3
"""execute_install.py — run a plan produced by plan_install.py, with explicit
per-step user confirmation, structured logging, auth-token prompting, and
failure handling.

Usage:
    python3 scripts/execute_install.py --plan plan.json
    python3 scripts/execute_install.py --plan plan.json --yes   # auto-confirm

Behavior:
- Shows the full plan and asks for top-level confirmation. STOPS if declined.
- For each step:
    * shows the command,
    * runs skip_if and skips when the predicate is already satisfied,
    * pauses for confirmation on every state-changing step (unless --yes),
    * pauses for manual steps until the user marks them done,
    * streams output to the terminal AND a log file,
    * on failure: warn-and-continue or abort, per the step config.
- After all steps succeed, runs the verify command and writes
  ~/.isaaclab/install_profile.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    DEFAULT_PROFILE_PATH, colorize, confirm, now_iso, open_log,
    print_err, print_header, print_info, print_ok, print_step, print_warn,
)


# Tokens the skill knows how to prompt for. Add new ones here.
KNOWN_AUTH_TOKENS = {
    "NGC_API_KEY": {
        "prompt": "Enter your NGC API key (from https://ngc.nvidia.com/setup/api-key)",
        "env_var": "NGC_API_KEY",
        "secret": True,
    },
}


def _ask_secret(prompt):
    import getpass
    return getpass.getpass(prompt + ": ").strip()


def _prepare_env_for_step(step, base_env, auth_cache):
    env = dict(base_env)
    for key, value in (step.get("env") or {}).items():
        env[key] = value
    if step.get("needs_auth"):
        token_id = step["needs_auth"]
        spec = KNOWN_AUTH_TOKENS.get(token_id)
        if not spec:
            print_warn(f"Step references unknown auth token '{token_id}' — skipping prompt.")
            return env
        if token_id in auth_cache:
            env[spec["env_var"]] = auth_cache[token_id]
            return env
        existing = os.environ.get(spec["env_var"])
        if existing:
            print_info(f"Using {spec['env_var']} from your shell environment.")
            auth_cache[token_id] = existing
            env[spec["env_var"]] = existing
            return env
        print()
        print_warn(f"This step needs an auth token: {token_id}")
        value = _ask_secret(spec["prompt"]) if spec.get("secret") else input(spec["prompt"] + ": ").strip()
        if not value:
            raise SystemExit(f"Aborting — {token_id} is required for step '{step['id']}'.")
        auth_cache[token_id] = value
        env[spec["env_var"]] = value
    return env


def _expand(cmd, base_env):
    """Expand $VARS using os.path.expandvars against the current environment."""
    return os.path.expandvars(cmd)


def _run_streamed(cmd, cwd, env, log_fh, remote_runner=None, sudo_password=None):
    """Run a shell command, streaming its output to stdout and the log.

    If remote_runner is provided, the command runs over SSH instead of locally."""
    log_fh.write(f"\n=== {now_iso()}  $ {cmd}  (cwd={cwd})\n")
    log_fh.flush()
    if remote_runner is not None:
        try:
            return remote_runner.run(cmd, cwd=cwd, env=env, log_fh=log_fh,
                                     sudo_password=sudo_password)
        except Exception as e:  # noqa: BLE001
            log_fh.write(f"!!! remote exec error: {e}\n")
            return 1
    try:
        proc = subprocess.Popen(
            cmd, shell=True, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except OSError as e:
        log_fh.write(f"!!! could not start: {e}\n")
        return 127
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        log_fh.write(line)
    proc.wait()
    log_fh.write(f"=== exit code: {proc.returncode}\n")
    log_fh.flush()
    return proc.returncode


def _check_skip(skip_if, cwd, env, log_fh, remote_runner=None):
    log_fh.write(f"\n--- skip-if: {skip_if}\n")
    log_fh.flush()
    if remote_runner is not None:
        rc, _, _ = remote_runner.capture(skip_if, cwd=cwd, env=env)
        return rc == 0
    rc = subprocess.call(skip_if, shell=True, cwd=cwd, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return rc == 0


def execute(plan, assume_yes=False, abort_on_warn=False, remote_target=None):
    print_header(f"Executing plan: {plan['combo_id']}")
    print_info(plan["combo_title"])
    print()

    remote_runner = None
    sudo_password = None
    if remote_target:
        from _remote import open_runner, prompt_sudo_password, parse_target
        user, host, _ = parse_target(remote_target)
        # SSH connection (prompts for password).
        remote_runner = open_runner(remote_target)
        # Separate sudo password prompt. If any step is sudo, we'll need it.
        if any(s["requires_sudo"] for s in plan["steps"]):
            print_info("This plan contains sudo steps. Enter the sudo password for the remote user.")
            sudo_password = prompt_sudo_password(user, host) or None
            if sudo_password is None:
                print_info("No sudo password provided — assuming passwordless sudo on remote.")
        print_info(f"All steps will run on the REMOTE host {user}@{host}.")
        print()

    # Display the plan first
    print("The following steps will be run:\n")
    for i, s in enumerate(plan["steps"], 1):
        sudo = "  [sudo]" if s["requires_sudo"] else ""
        manual = "  [MANUAL]" if s.get("manual_step") else ""
        loc = "  [remote]" if remote_runner else ""
        print(f"  {i:2d}. {s['title']}{sudo}{manual}{loc}")
    print()
    print(f"  Verify: {plan['verify']['cmd']}")
    print()

    if plan["notes"]:
        print(colorize("Important notes:", "yellow"))
        for n in plan["notes"]:
            print(f"  - {n}")
        print()

    if not assume_yes:
        if not confirm("Proceed with the installation?", default=False):
            print_warn("Aborted by user. Nothing has been changed.")
            if remote_runner:
                remote_runner.disconnect()
            return 4

    log_path, log_fh = open_log(name=f"install-{plan['combo_id']}")
    print_info(f"Logging to {log_path}")

    base_env = os.environ.copy()
    auth_cache = {}
    failures = []
    for i, step in enumerate(plan["steps"], 1):
        print_header(f"Step {i}/{len(plan['steps'])}: {step['title']}")
        print(f"  cwd: {step['cwd']}")
        print(f"  $ {step['cmd']}")
        if step.get("notes"):
            print(f"  note: {step['notes']}")

        # Skip-if check
        if step.get("skip_if"):
            if _check_skip(step["skip_if"], step["cwd"], base_env, log_fh, remote_runner=remote_runner):
                print_ok("Skip condition already satisfied — skipping this step.")
                continue

        # Manual steps pause for the user to do something before continuing
        if step.get("manual_step"):
            if remote_runner:
                print_warn("This step requires manual action on the REMOTE host before continuing.")
            else:
                print_warn("This step requires manual action before continuing.")
            if not confirm("Have you completed the manual action?", default=False, assume_yes=False):
                print_err("Aborting — manual step not completed.")
                log_fh.close()
                if remote_runner:
                    remote_runner.disconnect()
                return 5

        # Confirm before any state-changing step (always, unless --yes)
        if not assume_yes:
            sudo_warn = " (will use sudo)" if step["requires_sudo"] else ""
            if not confirm(f"Run this step?{sudo_warn}", default=True):
                if step["on_failure"] == "warn":
                    print_warn("Skipped by user. Continuing.")
                    continue
                print_err("Aborted by user at this step.")
                log_fh.close()
                return 6

        try:
            env = _prepare_env_for_step(step, base_env, auth_cache)
        except SystemExit as e:
            print_err(str(e))
            log_fh.close()
            if remote_runner:
                remote_runner.disconnect()
            return 7

        cwd = step["cwd"]
        if remote_runner is None:
            try:
                Path(cwd).mkdir(parents=True, exist_ok=True)
            except (FileNotFoundError, NotADirectoryError):
                pass  # cwd may legitimately reference a path that doesn't exist yet

        rc = _run_streamed(step["cmd"], cwd, env, log_fh,
                           remote_runner=remote_runner,
                           sudo_password=sudo_password if step["requires_sudo"] else None)
        if rc != 0:
            failures.append(step["id"])
            print_err(f"Step '{step['id']}' failed (exit {rc}).")
            if step["on_failure"] == "abort":
                print_err("on_failure=abort — stopping.")
                log_fh.close()
                if remote_runner:
                    remote_runner.disconnect()
                return 1
            elif step["on_failure"] == "warn":
                print_warn(f"on_failure=warn — continuing despite failure.")
                if abort_on_warn:
                    log_fh.close()
                    if remote_runner:
                        remote_runner.disconnect()
                    return 1
            else:
                print_err("Unknown on_failure policy — treating as abort.")
                log_fh.close()
                if remote_runner:
                    remote_runner.disconnect()
                return 1
        else:
            print_ok(f"Step '{step['id']}' completed.")

    # Verify
    print_header("Verification")
    print(f"  $ {plan['verify']['cmd']}")
    print(f"  cwd: {plan['verify']['cwd']}")
    verify_env = os.environ.copy() if remote_runner is None else {}
    if not assume_yes:
        if not confirm("Run the verification smoke test now?", default=True):
            print_warn("Skipping verification.")
            verify_rc = -1
            verify_passed = False
        else:
            verify_rc = _run_streamed(plan["verify"]["cmd"], plan["verify"]["cwd"],
                                       verify_env, log_fh, remote_runner=remote_runner)
            verify_passed = (verify_rc == 0)
    else:
        verify_rc = _run_streamed(plan["verify"]["cmd"], plan["verify"]["cwd"],
                                   verify_env, log_fh, remote_runner=remote_runner)
        verify_passed = (verify_rc == 0)

    if verify_passed:
        print_ok("Verification PASSED.")
    elif verify_rc == -1:
        print_warn("Verification skipped — you can run it later with scripts/verify.py.")
    else:
        if plan["verify"]["headless_ok"]:
            print_warn(
                "Verification FAILED in headless mode. Install steps reported success, "
                "but the smoke test could not import or launch Isaac Lab. Common causes:"
            )
            print_info("- NVIDIA driver too old. See docs/source/setup/installation/index.rst.")
            print_info("- GPU not accessible inside container / WSL.")
            print_info("- Conda env not activated (run `conda activate {env}`).".format(env=plan["env_name"]))
            print_info(f"Full log: {log_path}")
        else:
            print_err("Verification failed.")

    # Save profile (local always; remote also writes a copy on the remote host)
    try:
        profile_path = _write_profile(plan, verify_passed, log_path, failures,
                                       remote_target=remote_target)
        print_ok(f"Saved install profile (local): {profile_path}")
        if remote_runner is not None:
            try:
                remote_home = remote_runner.remote_home()
                remote_profile_dir = f"{remote_home}/.isaaclab"
                remote_runner.remote_mkdirs(remote_profile_dir)
                runner_sftp_path = f"{remote_profile_dir}/install_profile.yaml"
                remote_runner.put_file(profile_path, runner_sftp_path)
                print_ok(f"Saved install profile (remote): {runner_sftp_path}")
            except Exception as e:  # noqa: BLE001
                print_warn(f"Could not copy profile to remote: {e}")
    except Exception as e:  # noqa: BLE001
        print_warn(f"Could not write install profile: {e}")

    log_fh.close()
    if remote_runner is not None:
        remote_runner.disconnect()

    if failures and verify_passed is False:
        return 1
    return 0


def _detect_versions_for_profile(plan):
    """Best-effort: read VERSION and probe pip for isaacsim package version."""
    info = {}
    isaaclab_dir = Path(plan["isaaclab_dir"])
    version_file = isaaclab_dir / "VERSION"
    if version_file.is_file():
        try:
            info["isaaclab_version"] = version_file.read_text().strip()
        except OSError:
            pass

    env_python = plan["placeholders"].get("ENV_PYTHON")
    if env_python and not env_python.startswith("$(") and Path(env_python.split()[0]).is_file():
        try:
            rc = subprocess.run(
                [env_python, "-c", "import isaacsim, sys; print(getattr(isaacsim, '__version__', 'unknown'))"],
                capture_output=True, text=True, timeout=20,
            )
            if rc.returncode == 0:
                info["isaacsim_version"] = rc.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def _write_profile(plan, verify_passed, log_path, failures, remote_target=None):
    """Write ~/.isaaclab/install_profile.yaml capturing this install."""
    from _lib import write_yaml

    facts = {}
    # Load preflight from the original input if recorded alongside the plan
    versions = _detect_versions_for_profile(plan) if not remote_target else {}

    profile = {
        "schema_version": 1,
        "timestamp": now_iso(),
        "skill_version": "0.1.0",
        "combo_id": plan["combo_id"],
        "combo_title": plan["combo_title"],
        "install_method": {
            "isaaclab": plan["isaaclab_source"],
            "isaacsim": plan["isaacsim_source"],
            "env_manager": plan["env_manager"],
            "env_name": plan["env_name"],
        },
        "paths": {
            "isaaclab_dir": plan["isaaclab_dir"],
            "isaacsim_path": plan["isaacsim_path"],
            "env_python": plan["placeholders"].get("ENV_PYTHON"),
        },
        "versions": {
            "isaaclab": versions.get("isaaclab_version"),
            "isaacsim": versions.get("isaacsim_version"),
            "isaacsim_pinned_in_combo": plan["placeholders"].get("ISAACSIM_VERSION"),
        },
        "verification": {
            "smoke_test": "pass" if verify_passed else "fail",
            "headless": True,
        },
        "failed_steps": failures,
        "log_path": str(log_path),
        "remote_target": remote_target,
    }
    return write_yaml(DEFAULT_PROFILE_PATH, profile)


def main(argv=None):
    p = argparse.ArgumentParser(description="Execute an Isaac Lab installation plan.")
    p.add_argument("--plan", required=True, help="Path to plan.json produced by plan_install.py")
    p.add_argument("-y", "--yes", action="store_true", help="Skip per-step confirmations.")
    p.add_argument("--abort-on-warn", action="store_true",
                   help="Treat any warn-level failure as fatal.")
    p.add_argument("--remote", metavar="USER@HOST[:PORT]",
                   help="Run every step on a remote Linux host over SSH (password-prompted).")
    args = p.parse_args(argv)

    plan = json.loads(Path(args.plan).read_text())
    return execute(plan, assume_yes=args.yes, abort_on_warn=args.abort_on_warn,
                   remote_target=args.remote)


if __name__ == "__main__":
    sys.exit(main())
