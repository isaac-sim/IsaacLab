#!/usr/bin/env python3
"""preflight.py — detect system facts relevant to an Isaac Lab install.

Runs on stock Python 3.6+ (no third-party deps). Emits a JSON document on
stdout suitable for piping into recommend.py:

    python3 scripts/preflight.py > preflight.json
    python3 scripts/recommend.py --preflight preflight.json

Also prints a human-readable summary on stderr unless --quiet is given.

Detects: OS, distro, kernel, arch, glibc, GPU model(s), NVIDIA driver, CUDA
runtime, Python interpreters present (3.10/3.11/3.12), conda/uv presence,
disk free, RAM, existing Isaac Lab installs, existing Isaac Sim installs,
docker/wsl detection, and network reachability of key endpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    disk_free_gb, get_version, now_iso, print_header, print_info,
    print_ok, print_warn, run, total_ram_gb, which,
)


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def detect_os():
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "arch": platform.machine(),
    }
    # /etc/os-release for distro details
    distro = {}
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    distro[k] = v.strip('"')
    except OSError:
        pass
    info["distro_id"] = distro.get("ID")
    info["distro_version"] = distro.get("VERSION_ID")
    info["distro_pretty"] = distro.get("PRETTY_NAME")
    # WSL detection
    info["is_wsl"] = "microsoft" in info["release"].lower() or os.environ.get("WSL_DISTRO_NAME") is not None
    return info


def detect_glibc():
    # Try ldd --version (works on most Linux).
    v = get_version(["ldd", "--version"], r"(?:GLIBC|GNU libc|ldd)\s+\S*?(\d+\.\d+)")
    if v:
        return v
    # Fallback: parse output line that mentions "libc"
    v = get_version(["ldd", "--version"], r"(\d+\.\d+)")
    return v


def detect_gpus():
    """Use nvidia-smi if available."""
    if not which("nvidia-smi"):
        return {"available": False, "gpus": [], "driver": None, "cuda_driver_api": None}
    res = run([
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version,uuid",
        "--format=csv,noheader,nounits",
    ], timeout=10)
    gpus = []
    if res.returncode == 0:
        for line in (res.stdout or "").strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "name": parts[0],
                    "vram_mb": int(parts[1]) if parts[1].isdigit() else None,
                    "driver": parts[2],
                    "uuid": parts[3] if len(parts) > 3 else None,
                })
    driver = gpus[0]["driver"] if gpus else None
    # CUDA from nvidia-smi header
    cuda = None
    res2 = run(["nvidia-smi"], timeout=10)
    m = re.search(r"CUDA Version:\s*([\d.]+)", res2.stdout or "")
    if m:
        cuda = m.group(1)
    return {
        "available": bool(gpus),
        "gpus": gpus,
        "driver": driver,
        "cuda_driver_api": cuda,
    }


def detect_cuda_toolkit():
    nvcc = which("nvcc")
    if not nvcc:
        return {"present": False, "version": None, "path": None}
    v = get_version(["nvcc", "--version"], r"release\s+([\d.]+)")
    return {"present": True, "version": v, "path": nvcc}


def detect_pythons():
    candidates = ["python3", "python3.10", "python3.11", "python3.12", "python3.13", "python"]
    found = {}
    for cmd in candidates:
        path = which(cmd)
        if not path:
            continue
        v = get_version([cmd, "--version"], r"Python\s+([\d.]+)")
        if v:
            found[v] = path
    return found


def detect_env_managers():
    managers = {}
    # conda
    conda_path = which("conda")
    if conda_path:
        v = get_version(["conda", "--version"], r"conda\s+([\d.]+)")
        managers["conda"] = {"path": conda_path, "version": v}
    # uv
    uv_path = which("uv")
    if uv_path:
        v = get_version(["uv", "--version"], r"uv\s+([\d.]+)")
        managers["uv"] = {"path": uv_path, "version": v}
    # mamba (also conda-compatible)
    mamba_path = which("mamba")
    if mamba_path:
        managers["mamba"] = {"path": mamba_path}
    return managers


def detect_disk(paths):
    out = {}
    for p in paths:
        gb = disk_free_gb(p)
        out[str(p)] = round(gb, 1) if gb is not None else None
    return out


def detect_existing_isaaclab():
    """Look for an IsaacLab repo near $HOME and the script location."""
    hits = []
    candidates = [
        Path.cwd(),
        Path.home() / "IsaacLab",
        Path.home() / "Isaac_Projects" / "IsaacLab",
    ]
    # Also check the parent of the skill (this repo)
    skill_root = Path(__file__).resolve().parents[2]
    candidates.append(skill_root)
    seen = set()
    for c in candidates:
        try:
            c = c.resolve()
        except OSError:
            continue
        if c in seen:
            continue
        seen.add(c)
        if (c / "isaaclab.sh").is_file() and (c / "VERSION").is_file():
            try:
                version = (c / "VERSION").read_text().strip()
            except OSError:
                version = None
            hits.append({"path": str(c), "version": version})
    return hits


def detect_existing_isaacsim():
    hits = []
    candidates = [
        Path.home() / "isaacsim",
        Path("/opt/isaacsim"),
    ]
    for c in candidates:
        try:
            if (c / "isaac-sim.sh").is_file():
                hits.append({"path": str(c), "kind": "binary"})
        except OSError:
            continue
    # pip-installed isaacsim is detectable per-environment; we don't probe envs here.
    return hits


def detect_network():
    """Quick reachability check for endpoints we'll hit."""
    endpoints = {
        "github.com": 443,
        "pypi.org": 443,
        "pypi.nvidia.com": 443,
        "download.pytorch.org": 443,
    }
    results = {}
    for host, port in endpoints.items():
        try:
            with socket.create_connection((host, port), timeout=3):
                results[host] = True
        except OSError:
            results[host] = False
    return results


def detect_in_docker():
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup") as f:
            return "docker" in f.read() or "containerd" in f.read()
    except OSError:
        return False


def detect_display():
    return {
        "DISPLAY": os.environ.get("DISPLAY"),
        "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY"),
        "headless_only": not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_preflight():
    os_info = detect_os()
    home = str(Path.home())
    facts = {
        "schema_version": 1,
        "timestamp": now_iso(),
        "os": os_info,
        "glibc": detect_glibc(),
        "gpu": detect_gpus(),
        "cuda_toolkit": detect_cuda_toolkit(),
        "python_interpreters": detect_pythons(),
        "env_managers": detect_env_managers(),
        "ram_gb": round(total_ram_gb() or 0, 1) if total_ram_gb() else None,
        "disk_free_gb": detect_disk([home, "/"]),
        "existing_isaaclab": detect_existing_isaaclab(),
        "existing_isaacsim": detect_existing_isaacsim(),
        "network": detect_network(),
        "in_docker": detect_in_docker(),
        "display": detect_display(),
        "user": os.environ.get("USER") or os.environ.get("LOGNAME"),
        "home": home,
        "shell": os.environ.get("SHELL"),
    }
    return facts


def print_summary(facts, stream):
    """Render a human-readable summary on `stream`."""
    def w(line=""):
        stream.write(line + "\n")

    w()
    w("============================================================")
    w(" Isaac Lab Installer — Preflight Summary")
    w("============================================================")
    osi = facts["os"]
    w(f"  OS:        {osi.get('distro_pretty') or osi.get('system')}  ({osi.get('machine')})")
    w(f"  Kernel:    {osi.get('release')}")
    w(f"  WSL:       {osi.get('is_wsl')}")
    w(f"  Docker:    {facts['in_docker']}")
    w(f"  GLIBC:     {facts.get('glibc') or 'unknown'}")
    w(f"  RAM:       {facts.get('ram_gb')} GB")
    disk_home = facts["disk_free_gb"].get(facts["home"])
    w(f"  Disk free: {disk_home} GB at {facts['home']}")
    gpu = facts["gpu"]
    if gpu["available"]:
        for g in gpu["gpus"]:
            w(f"  GPU:       {g['name']} ({g.get('vram_mb')} MB)")
        w(f"  Driver:    {gpu['driver']}")
        w(f"  CUDA API:  {gpu.get('cuda_driver_api')}")
    else:
        w("  GPU:       NOT DETECTED (nvidia-smi missing or no NVIDIA GPU)")
    cuda = facts["cuda_toolkit"]
    if cuda["present"]:
        w(f"  nvcc:      {cuda['version']} ({cuda['path']})")
    else:
        w("  nvcc:      not installed (only needed for source builds)")
    pys = facts["python_interpreters"]
    w(f"  Python:    {', '.join(sorted(pys.keys())) or 'none found'}")
    mgrs = facts["env_managers"]
    w(f"  uv:        {'yes' if 'uv' in mgrs else 'no'}")
    w(f"  conda:     {'yes' if 'conda' in mgrs else 'no'}")
    if facts["existing_isaaclab"]:
        for h in facts["existing_isaaclab"]:
            w(f"  IsaacLab:  found at {h['path']} (v{h.get('version')})")
    if facts["existing_isaacsim"]:
        for h in facts["existing_isaacsim"]:
            w(f"  IsaacSim:  found at {h['path']} ({h['kind']})")
    net = facts["network"]
    ok = sum(1 for v in net.values() if v)
    w(f"  Network:   {ok}/{len(net)} endpoints reachable")
    w("============================================================")
    w()


def run_preflight_remote(target):
    """Upload the skill's scripts dir to a temp location on `target` and run
    preflight.py there. Returns the parsed JSON facts dict."""
    from _remote import open_runner

    scripts_dir = Path(__file__).resolve().parent
    runner = open_runner(target)
    try:
        # Pick a unique remote workspace under the remote $HOME or /tmp.
        remote_tmp = f"/tmp/isaaclab-installer-{os.getpid()}"
        runner.remote_mkdirs(remote_tmp)
        runner.put_directory(scripts_dir, remote_tmp + "/scripts")
        # preflight imports _lib by sibling path; uploading the whole scripts/
        # dir preserves that layout. We do NOT need resources/combos.py for
        # preflight, but uploading it keeps the bundle self-contained for any
        # follow-up doctor/verify run on the same temp dir.
        rc, out, err = runner.capture(
            f"python3 {remote_tmp}/scripts/preflight.py --quiet",
            timeout=120,
        )
        if rc != 0:
            sys.stderr.write(err)
            raise SystemExit(f"Remote preflight exited {rc}.")
        return json.loads(out)
    finally:
        # Best-effort cleanup. Leave the scripts on disk so doctor/verify can
        # reuse them if the user runs them next; preflight JSON is tiny.
        runner.disconnect()


def main(argv=None):
    p = argparse.ArgumentParser(description="Detect system facts for Isaac Lab install.")
    p.add_argument("-o", "--output", help="Write JSON to this path instead of stdout.")
    p.add_argument("--quiet", action="store_true", help="Suppress human summary on stderr.")
    p.add_argument("--remote", metavar="USER@HOST[:PORT]",
                   help="Run preflight on a remote Linux host over SSH (password-prompted).")
    args = p.parse_args(argv)

    if args.remote:
        facts = run_preflight_remote(args.remote)
    else:
        facts = run_preflight()

    text = json.dumps(facts, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(text)
        if not args.quiet:
            print_summary(facts, sys.stderr)
            print(f"[preflight] wrote {args.output}", file=sys.stderr)
    else:
        print(text)
        if not args.quiet:
            print_summary(facts, sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
