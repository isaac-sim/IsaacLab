# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Set up and run the Shadow Hand Cosmos 3 Transfer demo.

Every download location and pin is read from ``sources.json``; this file contains none.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEMOS = Path(__file__).resolve().parent
REPO = DEMOS.parents[1]
LOCAL = DEMOS / ".local"


MAX_FRAME_GAP_S = 0.05
"""Largest gap between depth frames sent to Cosmos that is expected to stay clean. A 0.033 s gap (every step at 30 Hz)
stayed clean; larger gaps degrade the result."""


def load_sources() -> dict:
    return json.loads((DEMOS / "sources.json").read_text())


def load_config() -> dict:
    """Machine-local paths written by setup (ignored by Git)."""
    path = LOCAL / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


class Report:
    def __init__(self):
        self.failed = False

    def ok(self, text):
        print(f"ok:   {text}")

    def warn(self, text):
        print(f"WARN: {text}", file=sys.stderr)

    def fail(self, text):
        print(f"FAIL: {text}", file=sys.stderr)
        self.failed = True


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    path = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
    return path.read_text().strip() if path.exists() and path.read_text().strip() else None


def hf_access(repo: str, revision: str | None, token: str | None) -> tuple[bool, str]:
    """Ask the Hub whether this token can read the repo; this is the real access check, not a token-file check."""
    url = f"https://huggingface.co/api/models/{repo}" + (f"/revision/{revision}" if revision else "")
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403, 404):
            return False, f"HTTP {error.code}: no access (accept the license on the model page, or fix the token)"
        return False, f"HTTP {error.code}"
    except OSError as error:
        return False, f"network error: {error}"
    if body.get("gated") and not token:
        return False, "gated model and no token"
    return True, "readable"


def git_access(repo: str) -> tuple[bool, str]:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--exit-code", repo, "HEAD"], capture_output=True, text=True, env=env, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)
    if result.returncode == 0:
        return True, "readable"
    return False, "cannot read the repository (no access, or git has no credentials for it)"


def gpus() -> list[tuple[str, int | None, str]]:
    """(name, memory in MiB or None when the driver hides it, driver). MIG hosts report no memory for the parent GPU."""
    if not shutil.which("nvidia-smi"):
        return []
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        name, memory, driver = [part.strip() for part in line.split(",")]
        rows.append((name, int(memory) if memory.isdigit() else None, driver))
    return rows


INSTALL_HINTS = {
    "uv": 'curl -LsSf https://astral.sh/uv/install.sh | sh   (then: export PATH="$HOME/.local/bin:$PATH")',
    "git": "sudo apt-get install -y git",
    "ffmpeg": "sudo apt-get install -y ffmpeg",
    "git-lfs": "sudo apt-get install -y git-lfs && git lfs install",
    "hf": 'uv tool install "huggingface_hub[cli]"   (or set HF_TOKEN instead)',
}


EULA_URL = "https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html"


HF_TOKEN_URL = "https://huggingface.co/settings/tokens"


ACCEPTED = LOCAL / "accepted.env"


def eula_accepted() -> bool:
    if os.environ.get("OMNI_KIT_ACCEPT_EULA") == "YES":
        return True
    return ACCEPTED.exists() and "OMNI_KIT_ACCEPT_EULA=YES" in ACCEPTED.read_text()


def accepted_environment() -> dict:
    """Environment variables recording the user's license acceptance, for the processes we start."""
    return {"OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y"} if eula_accepted() else {}


def platform_environment() -> dict:
    """Environment for the processes we start: PCI GPU order everywhere, and the OpenMP preload on aarch64."""
    environment = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}  # GPU indexes then match nvidia-smi and --sim-gpu/--cosmos-gpu
    libgomp = Path("/lib/aarch64-linux-gnu/libgomp.so.1")
    if platform.machine() == "aarch64" and libgomp.exists():
        preload = os.environ.get("LD_PRELOAD")
        environment["LD_PRELOAD"] = f"{libgomp}:{preload}" if preload else str(libgomp)
    return environment


def save_hf_token(token: str) -> None:
    path = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token)
    path.chmod(0o600)
    os.environ["HF_TOKEN"] = token


def hf_token_valid(token: str) -> bool:
    request = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2", headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=20):
            return True
    except (urllib.error.URLError, OSError):
        return False


def pending_access(sources: dict) -> list[dict]:
    """Every access item that is still missing, each with the link a person needs and how to re-check it."""
    items = []
    token = hf_token()
    if not token:
        items.append({"id": "hf-token", "url": HF_TOKEN_URL, "what": "Hugging Face login (a Read access token)"})
    for name, model in sources["models"].items():
        allowed, _ = hf_access(model["repo"], model.get("revision"), token)
        if not allowed and token:
            items.append(
                {
                    "id": f"model:{name}",
                    "url": f"https://huggingface.co/{model['repo']}",
                    "what": f"license for {model['repo']} (open the page and click 'Agree and access repository')",
                }
            )
    framework = sources["frameworks"][sources["active_framework"]]
    if not git_access(framework["repo"])[0]:
        items.append(
            {
                "id": "framework",
                "url": framework["repo"],
                "what": "read access to the Cosmos Framework repository (request access, then sign in with git, "
                "for example 'gh auth login' or a credential helper with a personal access token)",
            }
        )
    if not eula_accepted():
        items.append({"id": "eula", "url": EULA_URL, "what": "the Isaac Sim license agreement (yes/no)"})
    return items


def access(args) -> int:
    """Walk through each missing access item with its web link; the person answers, we re-check."""
    import getpass

    sources = load_sources()
    if getattr(args, "accept_eula", False) and not eula_accepted():
        LOCAL.mkdir(exist_ok=True)
        ACCEPTED.write_text(f"OMNI_KIT_ACCEPT_EULA=YES\n# accepted via --accept-eula on {time.strftime('%F %T')}\n")
    interactive = sys.stdin.isatty() and not getattr(args, "non_interactive", False)
    for attempt in range(1, 4):
        items = pending_access(sources)
        if not items:
            print("All access items are in place.")
            return 0
        print(f"\n{len(items)} access item(s) still need a person:")
        for number, item in enumerate(items, 1):
            print(f"  {number}. {item['what']}\n     {item['url']}")
        if not interactive:
            print("\nACTION NEEDED: ask the user to complete the items above, then re-run this command.")
            print("For the license, the user can accept with:  ./demos/cosmos3-transfer/setup.sh --accept-eula")
            return 3
        for item in items:
            if item["id"] == "hf-token":
                print(f"\nOpen {item['url']} , create a token of type Read, and paste it here (input is hidden).")
                token = getpass.getpass("Hugging Face token (Enter to skip): ").strip()
                if token and hf_token_valid(token):
                    save_hf_token(token)
                    print("Token saved.")
                elif token:
                    print("That token was not accepted by Hugging Face.")
            elif item["id"] == "eula":
                print(f"\nRead the license: {item['url']}")
                answer = input("Do you accept the NVIDIA Omniverse license agreement? Type 'yes' to accept: ")
                if answer.strip().lower() == "yes":
                    LOCAL.mkdir(exist_ok=True)
                    ACCEPTED.write_text(
                        f"OMNI_KIT_ACCEPT_EULA=YES\n# accepted by the user on {time.strftime('%F %T')}\n"
                    )
                    print("Recorded.")
                else:
                    print("Not accepted. The simulator will not run until you accept.")
            else:
                input(f"\nOpen {item['url']}\nComplete: {item['what']}\nPress Enter when done (or Ctrl-C to stop). ")
    return 3 if pending_access(sources) else 0


def check(args) -> int:
    sources = load_sources()
    need = sources["requirements"]
    report = Report()

    if platform.system() == "Linux" and platform.machine() == "x86_64":
        report.ok("Linux x86_64")
    elif platform.system() == "Linux" and platform.machine() == "aarch64":
        report.ok("Linux aarch64")
        report.warn("aarch64 (ARM) is less tested; see APPENDIX.md.")
    else:
        report.fail(f"Linux x86_64 or aarch64 is required (found {platform.system()} {platform.machine()})")

    for tool in ("uv", "git", "ffmpeg"):
        if shutil.which(tool):
            report.ok(f"{tool} found")
        else:
            report.fail(f"{tool} is not installed. Install: {INSTALL_HINTS[tool]}")
    if shutil.which("git") and subprocess.run(["git", "lfs", "version"], capture_output=True).returncode == 0:
        report.ok("git lfs found")
    else:
        report.fail(f"Git LFS is not installed. Install: {INSTALL_HINTS['git-lfs']}")
    if not shutil.which("hf"):
        report.warn(f"hf CLI not found (needed once for 'hf auth login'). Install: {INSTALL_HINTS['hf']}")

    devices = gpus()
    if not devices:
        report.fail("nvidia-smi not found or lists no GPU; an NVIDIA driver is required")
    else:
        driver = devices[0][2]
        if int(driver.split(".")[0]) >= need["min_driver_major"]:
            report.ok(f"NVIDIA driver {driver}")
        else:
            report.fail(f"NVIDIA driver {driver} is too old; {need['min_driver_major']} or newer is required")
        report.ok(
            f"{len(devices)} GPU(s): "
            + "; ".join(
                f"{n} {m // 1024} GB" if m else f"{n} (memory hidden by the driver, e.g. MIG)" for n, m, _ in devices
            )
        )
        if len(devices) < need["min_gpu_count_full_demo"]:
            report.warn("One GPU: Cosmos and the simulator would share it. The tested layout is Cosmos on GPU 1.")
        elif devices[1][1] is not None and devices[1][1] < 46000:
            report.warn(f"GPU 1 has {devices[1][1]} MiB; the tested Cosmos runs peaked at about 45,700 MiB.")

    if Path("/proc/meminfo").exists():
        lines = Path("/proc/meminfo").read_text().splitlines()
        ram_gb = int(next(line for line in lines if "MemTotal" in line).split()[1]) // 1048576
        if ram_gb >= need["min_ram_gb"]:
            report.ok(f"{ram_gb} GB RAM")
        else:
            report.warn(f"{ram_gb} GB RAM; {need['min_ram_gb']} GB is recommended")
    free_gb = shutil.disk_usage(REPO).free // 1024**3
    if free_gb >= need["min_disk_gb"]:
        report.ok(f"{free_gb} GB free disk")
    else:
        report.fail(f"{free_gb} GB free disk; about {need['min_disk_gb']} GB is needed (installed files: 60 to 70 GB)")

    if eula_accepted():
        report.ok("Isaac Sim license accepted")
    else:
        report.warn(f"Isaac Sim license not accepted yet. Run ./demos/cosmos3-transfer/setup.sh --access ({EULA_URL})")

    if args.offline:
        print("Skipping access checks (--offline).")
    else:
        token = hf_token()
        report.ok("Hugging Face token found") if token else report.fail(
            "no Hugging Face login: run 'hf auth login' or export HF_TOKEN"
        )
        for name, model in sources["models"].items():
            allowed, detail = hf_access(model["repo"], model.get("revision"), token)
            (report.ok if allowed else report.fail)(f"{name}: {model['repo']} {detail}")
        active = sources["active_framework"]
        framework = sources["frameworks"][active]
        allowed, detail = git_access(framework["repo"])
        label = f"framework '{active}' ({framework['access']}): {framework['repo']} {detail}"
        (report.ok if allowed else report.fail)(label)

    if report.failed:
        print("Check failed. Fix the items above and run it again.", file=sys.stderr)
        return 1
    print("Check passed.")
    return 0


def wait_for_socket(path: str, process: subprocess.Popen, timeout: float, log: Path) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Cosmos exited early (code {process.returncode}); see {log}")
        if Path(path).exists():
            return
        time.sleep(2)
    raise RuntimeError(f"Cosmos was not ready after {timeout:.0f} s; see {log}")


def _nvml_used_mib() -> list[int | None]:
    """Used memory per visible device through NVML, which still works when nvidia-smi hides it (MIG hosts)."""
    try:
        import re

        import pynvml

        listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
        mig = re.findall(r"UUID: (MIG-[0-9a-fA-F-]+)", listing)
        uuids = mig or re.findall(r"UUID: (GPU-[0-9a-fA-F-]+)", listing)
        pynvml.nvmlInit()
        used = []
        for uuid in uuids:
            handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            used.append(int(pynvml.nvmlDeviceGetMemoryInfo(handle).used) // 2**20)
        return used
    except Exception:  # NVML is optional; any failure means "unavailable"
        return []


def gpu_memory_mib() -> list[int | None]:
    """Used memory per GPU in MiB; None where the driver hides it (MIG hosts report [Insufficient Permissions])."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True
        ).stdout
    except OSError:
        return []
    used = [int(line) if line.strip().isdigit() else None for line in out.splitlines() if line.strip()]
    if any(value is None for value in used):
        nvml = _nvml_used_mib()
        if nvml:
            return nvml
    return used


def choose_settings(sources: dict, demo: str, cosmos_gpu: int) -> tuple[dict, str]:
    """The reference recipe, or a lower tier when the Cosmos GPU has too little memory for the tier above."""
    tiers = sources["cosmos_settings"][demo]
    devices = gpus()
    total = devices[cosmos_gpu][1] if cosmos_gpu < len(devices) else None
    hidden = total is None
    if hidden:
        total = sources["cosmos_settings"]["assumed_gpu_mib_when_hidden"]
    for index, tier in enumerate(tiers):
        if total >= tier["min_gpu_mib"]:
            memory = f"{total} MiB" + (" (hidden by the driver, assumed)" if hidden else "")
            if index == 0:
                return tier, f"{tier['label']}. Cosmos GPU memory: {memory}."
            return tier, f"{tier['label']}. Fallback because the Cosmos GPU has {memory}, less than the recipe needs."
    return tiers[-1], tiers[-1]["label"]


SHADOW_TASK = "Isaac-Reorient-Cube-Shadow-Camera-Direct"


SHADOW_STEPS = 120
"""Environment steps recorded: 120 captures at 60 Hz, 2 s of simulation, plus the initial capture."""


SHADOW_SOCKET = f"/tmp/isaaclab-cosmos-shadow-{os.getuid() if hasattr(os, 'getuid') else 0}.sock"


def shadow_capture_command(output: Path, endpoint: str, steps: int = SHADOW_STEPS) -> list[str]:
    """The recording run: the Shadow Hand camera task with Cosmos on its camera modifier (shadow-hand capture.py)."""
    return [
        "uv", "run", "--project", str(REPO), "--no-sync", "--extra", "video", "python",
        str(DEMOS / "shadow-hand/runtime/capture.py"),
        "--output", str(output), "--endpoint", endpoint, "--steps", str(steps),
    ]  # fmt: skip


def parse_capture_gap(log_text: str) -> dict:
    """What the Cosmos camera modifier reported in the run's log: a too-large gap warning, or the mean/max gap."""
    import re

    found = {"warned": "Cosmos camera frames are" in log_text}
    match = re.search(r"Cosmos capture gap: mean ([0-9.]+) s, max ([0-9.]+) s over (\d+) captures", log_text)
    if match:
        found.update(mean_s=float(match.group(1)), max_s=float(match.group(2)), captures=int(match.group(3)))
    transport = re.search(r"Cosmos (keeps images on the GPU with CUDA IPC|sends images through the socket)", log_text)
    if transport:
        found["transport"] = "cuda_ipc" if "IPC" in transport.group(1) else "socket"
    return found


def shadow_verdict(checks: dict) -> tuple[bool, list[str]]:
    """PASS only when every required check holds; the second value lists what failed, in plain words."""
    labels = {
        "capture_completed": "the recording run did not complete",
        "frames_recorded": "fewer frames were recorded than steps were run",
        "cosmos_not_black": "the generated frames include black frames",
        "differs_from_plain": "the generated frames do not differ from the plain camera (is Cosmos on the camera?)",
        "service_saw_session": "the Cosmos service never reported an active camera session",
        "capture_gap_ok": "the camera captures were too far apart",
    }
    failed = [text for key, text in labels.items() if not checks.get(key)]
    return not failed, failed


def cosmos_status(endpoint: str) -> dict | None:
    """The Cosmos service's status, read with the integration's own protocol; None when it cannot be reached."""
    try:
        import importlib.util

        path = REPO / "source/isaaclab_experimental/isaaclab_experimental/cosmos/_protocol.py"
        spec = importlib.util.spec_from_file_location("demo_cosmos_protocol", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reply, _ = module.request(endpoint, {"op": "status"}, timeout=5.0)
        return reply
    except Exception:  # the service may be busy or just starting; the caller only needs to know it answered or not
        return None


def shadow_analysis(frames: Path) -> dict:
    """Facts about the recorded frames: how many, how many generated ones are black, how far they differ from plain."""
    import h5py
    import numpy as np

    with h5py.File(frames) as data:
        count = len(data["step"])
        plain, cosmos = data["plain"], data["cosmos"]
        black = int(sum(float(cosmos[i].mean()) < 1.0 for i in range(count)))
        step = max(1, count // 30)
        difference = (
            float(
                np.mean(
                    [abs(plain[i].astype("int16") - cosmos[i].astype("int16")).mean() for i in range(0, count, step)]
                )
            )
            if count
            else None
        )
    return {"paired_frames": count, "cosmos_black_frames": black, "mean_difference_from_plain": difference}


def write_raw_video(frames, width: int, height: int, fps: float, target: Path) -> bool:
    """Encode RGB frames (an iterable of height x width x 3 arrays) to an mp4 with ffmpeg; False when it cannot."""
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-framerate", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(target),
    ]  # fmt: skip
    try:
        writer = subprocess.Popen(command, stdin=subprocess.PIPE)
    except OSError:
        return False
    for frame in frames:
        writer.stdin.write(frame.tobytes())
    writer.stdin.close()
    return writer.wait() == 0


def caption_band(labels: list[str], panel_width: int, height: int = 48):
    """A dark band with one label above each panel."""
    import numpy as np
    from PIL import Image, ImageDraw

    band = Image.new("RGB", (panel_width * len(labels), height), (20, 20, 20))
    draw = ImageDraw.Draw(band)
    for index, label in enumerate(labels):
        draw.text((index * panel_width + 12, 16), label, fill=(235, 235, 235))
    return np.asarray(band)


def shadow_videos(frames: Path, media: Path, fps: float, renderer: str) -> list[str]:
    """Plain, Cosmos and depth videos plus one captioned three-panel video, all from the same recorded captures."""
    import h5py
    import numpy as np

    made = []
    with h5py.File(frames) as data:
        count = len(data["step"])
        size = data["plain"].shape[1]
        for name in ("plain", "cosmos", "depth_control"):
            target = media / f"shadow-{name.replace('_control', '')}.mp4"
            if write_raw_video((data[name][i] for i in range(count)), size, size, fps, target):
                made.append(str(target))
        band = caption_band([f"Plain camera ({renderer})", "Cosmos (generated live)", "Depth sent to Cosmos"], size)
        target = media / "shadow-plain-cosmos-depth.mp4"
        panels = (
            np.concatenate([band, np.concatenate([data[n][i] for n in ("plain", "cosmos", "depth_control")], axis=1)])
            for i in range(count)
        )
        if write_raw_video(panels, 3 * size, size + band.shape[0], fps, target):
            made.append(str(target))
    return made


def run_shadow(args, sources: dict, framework_root: str) -> int:
    """Record the Shadow Hand camera task with Cosmos on its camera modifier, and check that the modifier path works."""
    import json as _json

    output = Path(args.output or REPO.parent / "output" / f"shadow-{time.strftime('%Y%m%dT%H%M%S')}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f"Output folder: {output}")
    settings, why = choose_settings(sources, "shadow", args.cosmos_gpu)
    settings = {**settings, "history": args.history or settings["history"]}
    print(f"Cosmos settings: {why}")
    endpoint = f"unix://{SHADOW_SOCKET}"
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.sim_gpu),
        PYTHONUNBUFFERED="1",
        **platform_environment(),
        **accepted_environment(),
    )
    environment.pop("DISPLAY", None)
    cosmos_command = [
        "uv", "run", "--no-sync", "python", str(DEMOS / "run_cosmos.py"),
        "--device", f"cuda:{args.cosmos_gpu}", "--endpoint", endpoint, "--kv-window", str(settings["history"]),
        "--attention-sink", str(settings["attention_sink"]), "--fps", "30",
    ]  # fmt: skip
    if not (settings["compile"] if args.compile is None else args.compile):
        cosmos_command.append("--no-compile")
    result: dict = {
        "demo": "shadow", "task": SHADOW_TASK, "steps": SHADOW_STEPS, "cosmos_settings": {**settings, "reason": why},
        "completed": False, "mode": "live: the camera modifier sends depth to Cosmos during the run",
    }  # fmt: skip
    peak: dict[str, int | None] = {"cosmos_gpu": None, "sim_gpu": None}
    started = time.monotonic()
    cosmos = None
    Path(SHADOW_SOCKET).unlink(missing_ok=True)
    try:
        with (output / "cosmos.log").open("w") as cosmos_log:
            cosmos = subprocess.Popen(
                cosmos_command, cwd=framework_root, stdout=cosmos_log, stderr=subprocess.STDOUT,
                env=dict(os.environ, LD_LIBRARY_PATH="", **platform_environment()), start_new_session=True,
            )  # fmt: skip
            wait_for_socket(SHADOW_SOCKET, cosmos, args.ready_timeout, output / "cosmos.log")
            print("Recording the Shadow Hand task with Cosmos on the camera modifier...", flush=True)
            with (output / "capture.log").open("w") as log:
                process = subprocess.Popen(
                    shadow_capture_command(output, endpoint), cwd=REPO, env=environment, stdout=log,
                    stderr=subprocess.STDOUT,
                )  # fmt: skip
                while process.poll() is None:
                    status = cosmos_status(endpoint)
                    if status and status.get("session_active"):
                        result["service_saw_session"] = True
                    used = gpu_memory_mib()
                    for key, index in (("sim_gpu", args.sim_gpu), ("cosmos_gpu", args.cosmos_gpu)):
                        if index < len(used) and used[index] is not None:
                            peak[key] = max(peak[key] or 0, used[index])
                    time.sleep(2)
            result["capture_exit_code"] = process.returncode
    except Exception as error:  # report every failure in the result file, then clean up
        result["error"] = str(error)
        print(f"ERROR: {error}", file=sys.stderr)
    finally:
        if cosmos is not None and cosmos.poll() is None:
            os.killpg(cosmos.pid, signal.SIGTERM)
            try:
                cosmos.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(cosmos.pid, signal.SIGKILL)
        Path(SHADOW_SOCKET).unlink(missing_ok=True)

    captured = output / "capture-result.json"
    capture = _json.loads(captured.read_text()) if captured.exists() else {}
    frames = output / "paired-frames.h5"
    analysis = shadow_analysis(frames) if frames.exists() else {}
    log_text = (output / "capture.log").read_text() if (output / "capture.log").exists() else ""
    gap = parse_capture_gap(log_text)
    gaps = capture.get("capture_gap") or []
    gap["max_s"] = max([g["max"] for g in gaps], default=gap.get("max_s"))
    renderer = capture.get("renderer") or "unknown renderer"
    videos = shadow_videos(frames, output / "comparison", 60.0, renderer) if frames.exists() else []
    checks = {
        "capture_completed": result.get("capture_exit_code") == 0 and bool(capture.get("completed")),
        "frames_recorded": analysis.get("paired_frames", 0) >= SHADOW_STEPS,
        "cosmos_not_black": bool(analysis) and analysis["cosmos_black_frames"] == 0,
        "differs_from_plain": (analysis.get("mean_difference_from_plain") or 0) > 3.0,
        "service_saw_session": bool(result.get("service_saw_session")),
        "capture_gap_ok": not gap["warned"] and (gap.get("max_s") or 0) <= MAX_FRAME_GAP_S + 1e-6,
    }
    passed, failed = shadow_verdict(checks)
    result.update(
        completed=checks["capture_completed"],
        proof={"passed": passed, "failed": failed, "checks": checks, **analysis},
        capture_gap=gap, renderer=renderer, videos=videos, peak_gpu_memory_mib=peak,
        generation_seconds_per_chunk=summary([e["seconds"] for e in capture.get("events", [])]),
        wall_s=round(time.monotonic() - started, 1), components=collect_versions(),
    )  # fmt: skip
    (output / "result.json").write_text(_json.dumps(result, indent=2) + "\n")
    (output / "metrics.md").write_text(shadow_markdown(result))
    verdict = "PASS" if passed else "FAIL: " + "; ".join(failed)
    print(f"Camera modifier path: {verdict}")
    print(f"Report: {output / 'metrics.md'}")
    return 0 if passed else 1


def shadow_markdown(result: dict) -> str:
    proof = result["proof"]
    lines = [
        "# Shadow Hand camera task with Cosmos on the camera modifier",
        "",
        f"Result: **{'PASS' if proof['passed'] else 'FAIL'}**"
        + ("" if proof["passed"] else " (" + "; ".join(proof["failed"]) + ")"),
        "",
        f"Task: `{result['task']}`, {result['steps']} steps, one run: {result['mode']}.",
        f"Plain camera renderer: {result['renderer']}. A random agent with the environment and its actions seeded 42.",
        "The plain image, the exact depth sent to Cosmos and every generated frame come from the same capture.",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for key, ok in proof["checks"].items():
        lines.append(f"| {key.replace('_', ' ')} | {'yes' if ok else 'NO'} |")
    gap = result["capture_gap"]
    lines += [
        "",
        f"- Frames recorded: {proof.get('paired_frames')}, black generated frames: {proof.get('cosmos_black_frames')}",
        f"- Mean absolute difference, generated against plain (0 to 255): {proof.get('mean_difference_from_plain')}",
        f"- Capture gap: max {gap.get('max_s')} s, warning raised: {gap.get('warned')}",
        f"- Generation seconds per chunk: {result.get('generation_seconds_per_chunk')}",
        f"- Wall time: {result.get('wall_s')} s; peak GPU memory (MiB): {result['peak_gpu_memory_mib']}",
        "",
        "## Videos",
        "",
    ]
    lines += [f"- `{path}`" for path in result["videos"]] or ["- none written"]
    return "\n".join(lines) + "\n"


def use_physical_gpu_order() -> None:
    """Make ``--sim-gpu`` and ``--cosmos-gpu`` mean the GPU numbers ``nvidia-smi`` lists.

    A login shell can export ``CUDA_VISIBLE_DEVICES`` (even as a reversed list of GPU UUIDs). The Cosmos server would
    inherit it, so ``cuda:1`` could be another card than the one the launcher measured and chose settings for.
    """
    inherited = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if inherited:
        print(
            f"Ignoring the inherited CUDA_VISIBLE_DEVICES={inherited}; GPU numbers follow nvidia-smi.",
            file=sys.stderr,
        )


def step(title: str, command: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print(f"\n==> {title}\n    $ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd or REPO, env=env, check=True)


def clone_at(repo: str, revision: str, target: Path) -> None:
    """Clone once, then only move to the pinned revision; reruns reuse the checkout."""
    if not (target / ".git").exists():
        step(f"Clone {repo}", ["git", "clone", "--filter=blob:none", repo, str(target)])
    step(f"Check out {revision[:9]}", ["git", "-C", str(target), "checkout", "--quiet", revision])


def bootstrap(args) -> int:
    """Install the user-space tools (no sudo). System packages are printed, not installed."""
    if not shutil.which("uv"):
        step("Install uv", ["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
        os.environ["PATH"] = f"{Path.home() / '.local/bin'}{os.pathsep}{os.environ['PATH']}"
    if not shutil.which("hf"):
        step("Install the Hugging Face CLI", ["uv", "tool", "install", "huggingface_hub[cli]"])
    missing = [t for t in ("git", "ffmpeg") if not shutil.which(t)]
    if subprocess.run(["git", "lfs", "version"], capture_output=True).returncode != 0:
        missing.append("git-lfs")
    if missing:
        print("Install these system packages with sudo, then run bootstrap again:")
        print(f"    sudo apt-get update && sudo apt-get install -y {' '.join(missing)}")
        if "git-lfs" in missing:
            print("    git lfs install")
    print("If uv was just installed, run:  export PATH=$HOME/.local/bin:$PATH")
    print(
        "Still manual (guided): run ./demos/cosmos3-transfer/setup.sh --access for the links and the license question."
    )
    return 0


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _hf_snapshots(repo: str) -> list[str]:
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    folder = cache / ("models--" + repo.replace("/", "--")) / "snapshots"
    return sorted(path.name for path in folder.iterdir()) if folder.is_dir() else []


def summary(values: list[float]) -> dict | None:
    """Count, mean, median, 95th percentile and max of a list of seconds; None when nothing was recorded."""
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return {
        "count": len(ordered), "mean": round(sum(ordered) / len(ordered), 4), "median": round(median, 4),
        "p95": round(p95, 4), "max": round(ordered[-1], 4),
    }  # fmt: skip


def collect_versions(resolved: bool = True) -> list[dict]:
    """Each component with its source, access level, pinned snapshot and what is actually present."""
    sources = load_sources()
    config = load_config() if resolved else {}
    active = sources["active_framework"]
    framework = sources["frameworks"][active]
    cosmos = sources["models"]["cosmos_checkpoint"]
    remote = subprocess.run(["git", "-C", str(REPO), "remote", "get-url", "origin"], capture_output=True, text=True)
    patch = framework.get("patch")
    framework_present = _git_head(Path(config["framework_root"])) if config.get("framework_root") else None
    snapshots = _hf_snapshots(cosmos["repo"]) if resolved else []
    return [
        {
            "component": "Demo code (this checkout)", "source": remote.stdout.strip() or "origin", "access": "public",
            "pinned": "branch head", "present": _git_head(REPO) if resolved else None,
        },
        {
            "component": f"Cosmos Framework ({active})", "source": framework["repo"], "access": framework["access"],
            "pinned": framework["revision"] + (f" + patch {Path(patch).name}" if patch else " (no patch)"),
            "present": framework_present,
        },
        {
            "component": "Cosmos Transfer checkpoint", "source": "https://huggingface.co/" + cosmos["repo"],
            "access": cosmos["access"], "pinned": cosmos["revision"], "present": ", ".join(snapshots) or None,
        },
    ]  # fmt: skip


def versions(args) -> int:
    for row in collect_versions(resolved=not args.pins_only):
        print(row["component"])
        print(f"    source : {row['source']}  [{row['access']}]")
        print(f"    pinned : {row['pinned']}")
        if row.get("present"):
            print(f"    present: {row['present']}")
    return 0


def setup(args) -> int:
    """Install the Cosmos Framework, the simulator environment and the Cosmos checkpoint."""
    sources = load_sources()
    access(args)  # walks a person through missing links and answers; never blocks the install steps
    if not args.skip_check and check(argparse.Namespace(offline=False)) != 0:
        return 1
    LOCAL.mkdir(exist_ok=True)
    config = load_config()
    framework = sources["frameworks"][sources["active_framework"]]
    root = Path(args.framework_root or config.get("framework_root") or LOCAL / "cosmos-framework").resolve()

    # Cosmos Framework: its own Python 3.13 / Torch environment, with this checkout's Cosmos server installed into it.
    clone_at(framework["repo"], framework["revision"], root)
    if framework.get("patch"):
        patch = DEMOS / framework["patch"]
        done = subprocess.run(
            ["git", "-C", str(root), "apply", "--reverse", "--check", str(patch)], capture_output=True
        )
        if done.returncode != 0:
            step("Apply the Framework compatibility patch", ["git", "-C", str(root), "apply", str(patch)])
    step("Install the Framework environment", ["uv", "sync", "--locked", "--all-extras", "--group=cu130-train"], root)
    step(
        "Install this checkout's Cosmos server into it",
        ["uv", "run", "--no-sync", "uv", "pip", "install", "--editable",
         str(REPO / "source/isaaclab_experimental/isaaclab_experimental/cosmos")],
        root,
    )  # fmt: skip
    step(
        "Download and verify the Cosmos checkpoint",
        ["uv", "run", "--no-sync", "python", str(DEMOS / "run_cosmos.py"), "--download-only"],
        root,
        dict(os.environ, LD_LIBRARY_PATH=""),
    )
    step("Install the simulator environment", ["uv", "sync", "--extra", "video", "--extra", "isaacsim"])

    config.update(framework_root=str(root))
    (LOCAL / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"\nSetup finished. Paths saved to {LOCAL / 'config.json'}.")
    print("Run the demo: ./demos/cosmos3-transfer/run.sh")
    return 0


def run(args) -> int:
    use_physical_gpu_order()
    sources = load_sources()
    config = load_config()
    framework_root = args.framework_root or config.get("framework_root") or os.environ.get("COSMOS_FRAMEWORK_ROOT")
    if not framework_root or not Path(framework_root).is_dir():
        print(
            "Cosmos Framework checkout not found. Pass --framework-root, set COSMOS_FRAMEWORK_ROOT, or run setup.",
            file=sys.stderr,
        )
        return 1
    if not eula_accepted():
        print(
            f"The Isaac Sim license has not been accepted. Read {EULA_URL} and run: "
            "./demos/cosmos3-transfer/setup.sh --access",
            file=sys.stderr,
        )
        return 1
    print("What this run uses:")
    versions(argparse.Namespace(pins_only=False))
    return run_shadow(args, sources, framework_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="Check the machine and your access to every download")
    c.add_argument("--offline", action="store_true", help="Skip Hugging Face and Git access checks")
    r = sub.add_parser("run", help="Start Cosmos, run the Shadow Hand demo, save the results, stop everything")
    r.add_argument("--framework-root", help="Cosmos Framework checkout (default: from setup config)")
    r.add_argument("--output", help="Output folder (default: ../output/shadow-<timestamp>)")
    r.add_argument("--sim-gpu", type=int, default=0)
    r.add_argument("--cosmos-gpu", type=int, default=1)
    r.add_argument("--history", type=int, help="Advanced: Cosmos history window")
    r.add_argument("--compile", dest="compile", action="store_true", default=None, help="Advanced: force compile on")
    r.add_argument("--no-compile", dest="compile", action="store_false", help="Advanced: eager inference (less memory)")
    r.add_argument("--ready-timeout", type=float, default=900)
    sub.add_parser("bootstrap", help="Install uv and the Hugging Face CLI; print the sudo packages still needed")
    v = sub.add_parser("versions", help="Show every source, access level and pinned snapshot we use")
    v.add_argument("--pins-only", action="store_true", help="Do not look at this machine, only at the pins")
    a = sub.add_parser("access", help="Walk through Hugging Face, repository and license steps with web links")
    a.add_argument("--accept-eula", action="store_true", help="Record that YOU accept the Isaac Sim license")
    a.add_argument("--non-interactive", action="store_true", help="Only list what is missing")
    s = sub.add_parser("setup", help="Install what the demo needs (Cosmos, simulator environment, checkpoint)")
    s.add_argument("--accept-eula", action="store_true", help="Record that YOU accept the Isaac Sim license")
    s.add_argument("--non-interactive", action="store_true")
    s.add_argument("--framework-root", help="Where to put the Cosmos Framework checkout (default: .local)")
    s.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()
    return {
        "check": check,
        "run": run,
        "setup": setup,
        "bootstrap": bootstrap,
        "access": access,
        "versions": versions,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
