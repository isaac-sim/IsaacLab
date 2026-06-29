#!/usr/bin/env python3
"""doctor.py — diagnose a broken or partial Isaac Lab install.

Detects common breakage patterns and prints prioritized findings with
suggested fixes. Does NOT modify anything. Run when:
- the user's install seems broken,
- they want a second opinion before filing a GitHub issue,
- they're sharing a setup with a teammate and want to confirm sanity.

Usage:
    python3 scripts/doctor.py                 # uses ~/.isaaclab/install_profile.yaml
    python3 scripts/doctor.py --isaaclab-dir /path/to/IsaacLab --env-python /path/to/python
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    DEFAULT_PROFILE_PATH, colorize, print_err, print_header, print_info,
    print_ok, print_warn, read_yaml, run, version_ge, which,
)

import preflight  # type: ignore  # noqa: E402


SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _add(findings, severity, title, detail, fix=None):
    findings.append({
        "severity": severity,
        "title": title,
        "detail": detail,
        "fix": fix,
    })


def check_repo(findings, isaaclab_dir):
    print_info(f"Checking IsaacLab repo at {isaaclab_dir} ...")
    if not (isaaclab_dir / "isaaclab.sh").is_file():
        _add(findings, "critical",
             "isaaclab.sh not found",
             f"{isaaclab_dir} does not look like a clone of IsaacLab "
             "(missing isaaclab.sh).",
             "Re-clone: git clone https://github.com/isaac-sim/IsaacLab.git")
        return
    if not (isaaclab_dir / "VERSION").is_file():
        _add(findings, "warning",
             "VERSION file missing",
             "Cannot determine the Isaac Lab version.",
             "Pull the latest main: cd IsaacLab && git pull origin main")
    else:
        version = (isaaclab_dir / "VERSION").read_text().strip()
        _add(findings, "info", "Isaac Lab version", version)
    # Detect symlink
    sym = isaaclab_dir / "_isaac_sim"
    if sym.is_symlink():
        target = os.readlink(sym)
        if not Path(target).exists():
            _add(findings, "critical",
                 "_isaac_sim symlink is broken",
                 f"Points to {target} which does not exist.",
                 f"Re-create the link: ln -sfn /path/to/isaacsim {sym}")
        else:
            _add(findings, "info", "_isaac_sim symlink", f"-> {target}")
    elif sym.exists() and sym.is_dir():
        _add(findings, "info", "_isaac_sim", "directory (not a symlink)")
    else:
        # OK for kitless and pip-Isaac-Sim combos
        _add(findings, "info", "_isaac_sim", "absent (OK for kitless / pip Isaac Sim combos)")


def check_env_python(findings, env_python):
    print_info(f"Checking env Python at {env_python} ...")
    if env_python.startswith("$("):
        # conda-resolved path
        rc = subprocess.run(["bash", "-lc", env_python[2:-1]], capture_output=True, text=True)
        if rc.returncode != 0 or not rc.stdout.strip():
            _add(findings, "critical",
                 "Could not resolve conda env python",
                 f"`{env_python}` did not produce a path: {rc.stderr.strip()}",
                 "Activate the env (conda activate <name>) and retry, or pass --env-python explicitly.")
            return None
        env_python = rc.stdout.strip()

    if not Path(env_python.split()[0]).is_file():
        _add(findings, "critical",
             "Environment Python missing",
             f"{env_python} does not exist on disk.",
             "Re-create the env: ./isaaclab.sh -u  (or -c for conda).")
        return None

    rc = subprocess.run([env_python, "--version"], capture_output=True, text=True)
    ver = (rc.stdout + rc.stderr).strip()
    _add(findings, "info", "Env python", f"{env_python} ({ver})")
    if "3.12" not in ver:
        _add(findings, "critical",
             "Python version mismatch",
             f"Isaac Sim 6.x requires Python 3.12; env has {ver}.",
             "Recreate the env with Python 3.12.")
    return env_python


def check_imports(findings, env_python):
    print_info("Probing imports inside the env ...")
    probe = (
        "import json, sys, importlib\n"
        "out = {}\n"
        "for name in ['isaaclab', 'isaacsim', 'torch', 'numpy', 'omni.client']:\n"
        "    try:\n"
        "        m = importlib.import_module(name)\n"
        "        out[name] = getattr(m, '__version__', 'imported')\n"
        "    except Exception as e:\n"
        "        out[name] = f'ERROR: {type(e).__name__}: {e}'\n"
        "print(json.dumps(out))\n"
    )
    rc = subprocess.run([env_python, "-c", probe], capture_output=True, text=True, timeout=60)
    if rc.returncode != 0:
        _add(findings, "critical",
             "Env python could not run import probe",
             rc.stderr.strip()[:500],
             "Activate the env first, or repair the install.")
        return
    try:
        data = json.loads(rc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        _add(findings, "critical",
             "Could not parse import probe output",
             rc.stdout[:500],
             "")
        return
    for mod, val in data.items():
        if str(val).startswith("ERROR"):
            severity = "warning" if mod == "omni.client" else "critical"
            fix = None
            if mod == "isaaclab":
                fix = "Re-run `./isaaclab.sh -i` from the repo root with the env activated."
            elif mod == "isaacsim":
                fix = "Re-install: `uv pip install \"isaacsim[all,extscache]==6.0.1.0\" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow`"
            elif mod == "torch":
                fix = "Re-install matching CUDA build: `uv pip install -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128`"
            _add(findings, severity, f"{mod} import failed", val, fix)
        else:
            _add(findings, "info", f"{mod}", val)


def check_driver(findings):
    print_info("Checking NVIDIA driver ...")
    if not which("nvidia-smi"):
        _add(findings, "critical",
             "nvidia-smi not found",
             "No NVIDIA driver detected.",
             "Install the latest NVIDIA production-branch driver (>= 580.95.05).")
        return
    gpu = preflight.detect_gpus()
    driver = gpu.get("driver")
    if not driver:
        _add(findings, "warning", "Could not read driver version",
             "nvidia-smi returned no driver string.", None)
        return
    _add(findings, "info", "NVIDIA driver", driver)
    if not version_ge(driver, "580.95.05"):
        _add(findings, "warning",
             "NVIDIA driver below recommended",
             f"Detected {driver}; docs recommend 580.95.05+.",
             "Upgrade your driver from the Unix Driver Archive.")


def check_glibc(findings):
    g = preflight.detect_glibc()
    _add(findings, "info", "GLIBC", g or "unknown")
    if g and not version_ge(g, "2.35"):
        _add(findings, "warning",
             "GLIBC < 2.35",
             f"Detected {g}. pip Isaac Sim requires GLIBC 2.35+.",
             "Use the binary combo instead, or upgrade your distro.")


def print_findings(findings):
    findings.sort(key=lambda f: SEVERITY_ORDER.get(f["severity"], 99))
    counts = {"critical": 0, "warning": 0, "info": 0}
    for f in findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1

    print_header("Doctor Report")
    for f in findings:
        sev = f["severity"]
        if sev == "critical":
            tag = colorize("[CRIT]", "red")
        elif sev == "warning":
            tag = colorize("[WARN]", "yellow")
        else:
            tag = colorize("[INFO]", "blue")
        print(f"{tag} {colorize(f['title'], 'bold')}")
        if f["detail"]:
            print(f"       {f['detail']}")
        if f["fix"]:
            print(f"       fix: {f['fix']}")
    print()
    print(colorize(
        f"Summary: {counts['critical']} critical, {counts['warning']} warning, {counts['info']} info",
        "bold",
    ))


def run_remote_doctor(target):
    """Upload the skill bundle to the remote and exec doctor.py there."""
    from _remote import open_runner
    scripts_dir = Path(__file__).resolve().parent
    runner = open_runner(target)
    try:
        remote_tmp = f"/tmp/isaaclab-installer-doctor-{os.getpid()}"
        runner.remote_mkdirs(remote_tmp)
        runner.put_directory(scripts_dir, remote_tmp + "/scripts")
        # Run doctor.py remotely. Use --profile pointing at the remote default.
        cmd = f"python3 {remote_tmp}/scripts/doctor.py"
        rc = runner.run(cmd, cwd=None, env=None, log_fh=None)
        return rc
    finally:
        runner.disconnect()


def main(argv=None):
    p = argparse.ArgumentParser(description="Diagnose an Isaac Lab install.")
    p.add_argument("--profile", default=str(DEFAULT_PROFILE_PATH))
    p.add_argument("--isaaclab-dir")
    p.add_argument("--env-python")
    p.add_argument("--skip-imports", action="store_true",
                   help="Skip the import-probe step (useful if you can't activate the env).")
    p.add_argument("--remote", metavar="USER@HOST[:PORT]",
                   help="Diagnose an install on a remote Linux host over SSH.")
    args = p.parse_args(argv)

    if args.remote:
        return run_remote_doctor(args.remote)

    profile = None
    if Path(args.profile).is_file():
        try:
            profile = read_yaml(args.profile)
        except Exception:
            profile = None

    isaaclab_dir = None
    if args.isaaclab_dir:
        isaaclab_dir = Path(args.isaaclab_dir).expanduser()
    elif profile:
        d = (profile.get("paths") or {}).get("isaaclab_dir")
        if d:
            isaaclab_dir = Path(d)
    if not isaaclab_dir or not (isaaclab_dir / "isaaclab.sh").is_file():
        # Fall back to current directory if it looks like a repo
        cwd = Path.cwd()
        if (cwd / "isaaclab.sh").is_file():
            isaaclab_dir = cwd
        else:
            isaaclab_dir = None

    env_python = args.env_python
    if not env_python and profile:
        env_python = (profile.get("paths") or {}).get("env_python")

    findings = []
    if isaaclab_dir:
        check_repo(findings, isaaclab_dir)
    else:
        _add(findings, "warning",
             "No IsaacLab directory found",
             "Pass --isaaclab-dir or run this from inside the IsaacLab repo.",
             None)
    check_driver(findings)
    check_glibc(findings)
    if env_python and not args.skip_imports:
        resolved = check_env_python(findings, env_python)
        if resolved:
            check_imports(findings, resolved)
    elif not env_python:
        _add(findings, "warning",
             "No env Python known",
             "Cannot probe imports. Pass --env-python or activate the env and re-run.",
             None)
    print_findings(findings)

    # Exit code: 1 if any critical findings, else 0
    return 1 if any(f["severity"] == "critical" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
