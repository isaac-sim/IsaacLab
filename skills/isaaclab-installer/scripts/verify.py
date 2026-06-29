#!/usr/bin/env python3
"""verify.py — run the post-install smoke test against an Isaac Lab install.

Resolves which python to use from an install profile (or from --env-python),
then runs resources/smoke_tests/hello_isaaclab.py through that interpreter.

Reports PASS/FAIL, prints a clear warning when headless mode fails, and
suggests next steps.

Usage:
    python3 scripts/verify.py                          # uses ~/.isaaclab/install_profile.yaml
    python3 scripts/verify.py --profile some/profile.yaml
    python3 scripts/verify.py --env-python /path/to/python --isaaclab-dir /path --kitless
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    DEFAULT_PROFILE_PATH, RESOURCES_DIR, colorize, print_err, print_header,
    print_info, print_ok, print_warn, read_yaml,
)


SMOKE = RESOURCES_DIR / "smoke_tests" / "hello_isaaclab.py"


def _resolve_python(profile, env_python_arg):
    if env_python_arg:
        return env_python_arg
    if profile is None:
        raise SystemExit("No profile found and no --env-python given. Cannot determine which Python to use.")
    paths = (profile.get("paths") or {})
    env_python = paths.get("env_python")
    if not env_python:
        raise SystemExit("Profile is missing paths.env_python.")
    if env_python.startswith("$(") and env_python.endswith(")"):
        # Resolve via subshell (conda)
        cmd = env_python[2:-1]
        rc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
        if rc.returncode != 0 or not rc.stdout.strip():
            raise SystemExit(f"Could not resolve env_python via `{cmd}`: {rc.stderr}")
        return rc.stdout.strip()
    if not Path(env_python.split()[0]).is_file():
        raise SystemExit(f"env_python does not exist on disk: {env_python}")
    return env_python


def _is_kitless(profile):
    if not profile:
        return False
    return (profile.get("install_method") or {}).get("isaacsim") == "kitless"


def _isaaclab_sh_path(profile, isaaclab_dir_arg):
    if isaaclab_dir_arg:
        d = Path(isaaclab_dir_arg).expanduser()
    elif profile and (profile.get("paths") or {}).get("isaaclab_dir"):
        d = Path(profile["paths"]["isaaclab_dir"])
    else:
        d = None
    if d and (d / "isaaclab.sh").is_file():
        return d
    return None


def run_smoke(env_python, kitless, isaaclab_dir):
    args = [env_python, str(SMOKE)]
    if kitless:
        args.append("--kitless")
    env = os.environ.copy()
    cwd = str(isaaclab_dir) if isaaclab_dir else None
    print_info(f"$ {' '.join(args)}")
    if cwd:
        print_info(f"  cwd: {cwd}")
    try:
        rc = subprocess.run(args, env=env, cwd=cwd).returncode
    except FileNotFoundError as e:
        print_err(str(e))
        return 127
    return rc


def run_smoke_remote(remote_target, env_python, kitless, isaaclab_dir):
    """Upload the smoke test to the remote and run it under env_python there."""
    from _remote import open_runner
    runner = open_runner(remote_target)
    try:
        remote_tmp = f"/tmp/isaaclab-installer-verify-{os.getpid()}"
        runner.remote_mkdirs(remote_tmp)
        runner.put_file(SMOKE, f"{remote_tmp}/hello_isaaclab.py")
        flag = " --kitless" if kitless else ""
        cmd = f"{env_python} {remote_tmp}/hello_isaaclab.py{flag}"
        print_info(f"$ {cmd}")
        if isaaclab_dir:
            print_info(f"  cwd: {isaaclab_dir}")
        return runner.run(cmd, cwd=str(isaaclab_dir) if isaaclab_dir else None,
                          env=None, log_fh=None)
    finally:
        runner.disconnect()


def main(argv=None):
    p = argparse.ArgumentParser(description="Run the Isaac Lab post-install smoke test.")
    p.add_argument("--profile", default=str(DEFAULT_PROFILE_PATH),
                   help="Path to install_profile.yaml.")
    p.add_argument("--env-python",
                   help="Override: absolute path to python in the install env.")
    p.add_argument("--isaaclab-dir",
                   help="Override: absolute path to the IsaacLab repo.")
    p.add_argument("--kitless", action="store_true",
                   help="Force kitless mode (skip Isaac Sim checks).")
    p.add_argument("--remote", metavar="USER@HOST[:PORT]",
                   help="Run the smoke test on a remote Linux host over SSH.")
    args = p.parse_args(argv)

    profile = None
    if Path(args.profile).is_file():
        try:
            profile = read_yaml(args.profile)
        except Exception as e:  # noqa: BLE001
            print_warn(f"Could not read profile {args.profile}: {e}")

    env_python = _resolve_python(profile, args.env_python)
    kitless = args.kitless or _is_kitless(profile)
    isaaclab_dir = _isaaclab_sh_path(profile, args.isaaclab_dir)

    print_header("Isaac Lab — Post-Install Verification")
    print_info(f"Python:     {env_python}")
    print_info(f"Mode:       {'kitless (no Isaac Sim)' if kitless else 'full (with Isaac Sim)'}")
    if isaaclab_dir:
        print_info(f"IsaacLab:   {isaaclab_dir}")
    if args.remote:
        print_info(f"Target:     remote ({args.remote})")
    print()

    if args.remote:
        rc = run_smoke_remote(args.remote, env_python, kitless, isaaclab_dir)
    else:
        rc = run_smoke(env_python, kitless, isaaclab_dir)

    print()
    if rc == 0:
        print_ok("VERIFICATION PASSED.")
        return 0

    print_err(f"VERIFICATION FAILED (exit {rc}).")
    print_warn(
        "The headless smoke test could not finish. Common causes (in order of likelihood):"
    )
    print_info("1. Environment is not activated. Re-run the script from inside the env.")
    print_info("2. NVIDIA driver is too old. Check `nvidia-smi` against the docs minimum (580.95.05+).")
    print_info("3. CUDA-incompatible PyTorch was installed by mistake. Re-run the install_torch step.")
    print_info("4. Running inside a container without GPU passthrough (`--gpus all`).")
    print_info("5. WSL without WSLg / GPU support enabled.")
    print()
    print_warn("Headless failure does NOT necessarily mean a broken install. If you intend "
               "to use Isaac Lab only with a display, try the GUI verify command:")
    print_info(f"  cd {isaaclab_dir or '<isaaclab-dir>'}")
    print_info("  ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --viz kit")
    return rc


if __name__ == "__main__":
    sys.exit(main())
