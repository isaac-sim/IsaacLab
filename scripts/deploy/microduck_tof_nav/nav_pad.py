# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.24", "onnxruntime>=1.16", "onnx>=1.14"]
# ///

"""Drive a MicroDuck with a gamepad while a learned ToF policy steers it around obstacles.

You hold the stick; the navigation policy reads the head's 8x8 time-of-flight (ToF) depth sensor and
the robot's body state, and adjusts your command before it reaches the walking policy. Runs on a
laptop on the same network as the robot.

  uv run nav_pad.py --robot ROBOT_HOST --dry-run     # sensor + policy check, no movement commands
  uv run nav_pad.py --robot ROBOT_HOST               # live: drive with the gamepad, the policy steers

Controls (the robot's gamepad, as usual):
  * stick forward / turn  -> the policy steers (forward speed capped, see --max-vx)
  * stick released        -> the robot stops
  * stick pulled back     -> plain gamepad driving (the sensor only looks forward)
  * X                     -> policy off / on (the robot coos / chirps); off = plain gamepad driving
  * Ctrl-C                -> end the run; the robot stops and the gamepad goes back to normal
Stand/sit, head modes and the other buttons work as usual.

Safety, on top of the policy:
  * anything within --stop-dist (20 cm) in the centre of the view stops forward motion
  * on the robot, a watchdog stops it if no policy command arrives for 0.25 s
    (Wi-Fi drop, laptop asleep, this script crashing)
  * the robot-side helper caps speed at 0.2 m/s whatever this script sends

How it works: two small helpers run on the robot over SSH. relay_on_duck.py streams body state
(50 Hz) and ToF frames to this script. padproxy_on_duck.py sits between the gamepad daemon and the
robot daemon (robotd): it reports your stick to this script and sends the policy's command in its
place. Starting the run needs the robot's sudo password once, to restart the gamepad daemon pointed at
the proxy; when the run ends the normal gamepad daemon is restored automatically.
"""

import argparse
import collections
import csv
import json
import math
import pathlib
import shlex
import subprocess
import sys
import threading
import time

import numpy as np
import onnxruntime as ort

HERE = pathlib.Path(__file__).resolve().parent

# ---------------------------------------------------------------------------------------------
# The navigation policy's contract: obs[1, 1343] -> action[1, 3], at 50 Hz.
#   0-2 gyro · 3-5 projected gravity · 6-19 joint positions minus default · 20-33 joint velocities
#   (one tick late) · 34-36 previous action · 37-39 requested velocity · 40-49 zeros (head/body pose)
#   50-52 previous applied command · 53-1332 ten control-tick ToF samples, oldest first, each
#   64 ranges (m) then 64 valid flags · 1333-1342 their ages (s). Normalisation is inside the ONNX.
# The action is a residual [forward, lateral, yaw] in [-1, 1], turned into a velocity command by
# command_from_action(), exactly as in training.
# ---------------------------------------------------------------------------------------------
OBS_LEN = 1343
MOUTH_INDEX = 9  # robotd reports 15 joints; the policy uses 14
DEFAULT_POSE = np.array(
    [
        0.0,
        -0.0873,
        -0.4579,
        -0.0049,
        0.4530,  # left leg
        0.3491,
        0.3491,
        0.0,
        0.0,  # neck / head
        0.0,
        0.0873,
        0.4579,
        0.0049,
        -0.4530,
    ],  # right leg
    dtype=np.float32,
)
TOF_HISTORY, TOF_ZONES = 10, 64
TOF_VALID_STATUS = {5, 9}  # VL53L5CX target status codes for a valid range
TOF_MIN_M, TOF_MAX_M = 0.12, 4.0  # readings outside this count as invalid
TOF_MAX_AGE_S = 0.25  # older than this and the command is zeroed
CMD_LIMITS = np.array([0.35, 0.12, 0.75], np.float32)
CMD_SLEW = np.array([1.0, 1.0, 3.0], np.float32)  # per second

CAP = np.array([0.15, 0.05, 0.75], np.float32)  # vx, vy, wz while the policy drives (vx from --max-vx)
BRAKE_IGNORE_ROWS = (6, 7)  # the floor just ahead of the feet: not an obstacle
ROBOT_DIR = "tof-nav"  # where the helpers are copied, in the robot's home
PROXY_SOCK = "/tmp/tof-nav/robotd.sock"  # padproxy_on_duck.py listens here for the gamepad daemon
GAMEPAD_DAEMON = "padd"  # codespell:ignore  (the robot's gamepad daemon and its systemd unit)
NAV_UNIT = GAMEPAD_DAEMON + "-nav"  # the temporary copy of it that talks to the proxy
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR"]


def command_from_action(action, requested, prev_applied, have_tof, tof_age, dt):
    """The training command processing: residual on top of the request, limits, slew, speed cap."""
    if not all(np.isfinite(x).all() for x in (action, requested, prev_applied)):
        raise ValueError("Non-finite action or velocity command")
    cmd = requested + np.clip(action, -1.0, 1.0) * CMD_LIMITS
    cmd = np.clip(cmd, [0.0, -CMD_LIMITS[1], -CMD_LIMITS[2]], CMD_LIMITS)
    req_speed, speed = np.linalg.norm(requested[:2]), max(np.linalg.norm(cmd[:2]), 1e-6)
    cmd[:2] *= min(req_speed / speed, 1.0)  # never faster than you asked
    cmd = prev_applied + np.clip(cmd - prev_applied, -CMD_SLEW * dt, CMD_SLEW * dt)
    cmd[1] = np.clip(cmd[1], -0.35 * cmd[0], 0.35 * cmd[0])
    if not requested.any() or not have_tof or tof_age > TOF_MAX_AGE_S:
        cmd[:] = 0.0
    return cmd.astype(np.float32)


def load_normalizer(path):
    """The input statistics baked into the ONNX, to warn when live inputs are far from training.
    Optional: returns None if the export does not carry them under the expected name."""
    try:
        import onnx
        from onnx import numpy_helper

        graph = onnx.load(str(path)).graph
        w = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
        mean = w["obs_normalizer._mean"]
        if "obs_normalizer._std" in w:
            std = w["obs_normalizer._std"]
        else:
            # New exports fold std + epsilon into a differently named initializer.
            centered = next(
                n.output[0] for n in graph.node if n.op_type == "Sub" and n.input[1] == "obs_normalizer._mean"
            )
            denominator = next(n.input[1] for n in graph.node if n.op_type == "Div" and n.input[0] == centered)
            std = w[denominator]
        if mean.shape != (1, OBS_LEN) or std.shape != mean.shape or not np.all(np.isfinite(std) & (std > 0)):
            return None
        return mean[0], std[0]
    except (KeyError, OSError, ValueError, StopIteration):
        return None


def policy_joints(values):
    """robotd's 15 joints without the mouth, in the policy's order."""
    values = np.asarray(values, np.float32)
    if values.shape != (15,) or not np.isfinite(values).all():
        raise ValueError("Expected 15 finite robotd joint values, including the mouth")
    return np.delete(values, MOUTH_INDEX)


class Navigator:
    """Builds the observation from live robot state and ToF frames, runs the policy, returns a command."""

    def __init__(self, model_path):
        self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        n = self.sess.get_inputs()[0].shape[-1]
        if isinstance(n, int) and n != OBS_LEN:
            sys.exit(f"{model_path} takes {n} inputs; this script feeds the {OBS_LEN}-input ToF navigation policy")
        if len(self.sess.get_outputs()) != 1 or self.sess.get_outputs()[0].shape[-1] != 3:
            raise ValueError("Expected one navigation output with three residuals")
        self.norm = load_normalizer(model_path)
        self.prev_action = np.zeros(3, np.float32)
        self.prev_applied = np.zeros(3, np.float32)
        self.prev_dq = None
        self.prev_rx = None
        self.tof_hist = collections.deque(maxlen=TOF_HISTORY)
        self.tof = None  # (ranges_m, valid, receive time)
        self.tof_received_at = 0.0
        self.max_z = 0.0

    def on_tof(self, msg):
        m = np.asarray(msg["mm"], np.float32) / 1000.0
        status = np.asarray(msg["status"])
        if m.shape != (TOF_ZONES,) or status.shape != (TOF_ZONES,) or not math.isfinite(msg["rx"]):
            raise ValueError("Expected 64 ToF ranges/statuses and a finite robot timestamp")
        valid = np.isfinite(m) & np.isin(status, list(TOF_VALID_STATUS)) & (m >= TOF_MIN_M) & (m <= TOF_MAX_M)
        self.tof = (np.where(valid, m, 0.0).astype(np.float32), valid.astype(np.float32), msg["rx"])
        self.tof_received_at = time.monotonic()

    def step(self, st, requested):
        q, dq_now = policy_joints(st["q"]) - DEFAULT_POSE, policy_joints(st["dq"])
        dq = self.prev_dq if self.prev_dq is not None else dq_now  # velocity is one tick late
        self.prev_dq = dq_now
        ranges, valid, rx = self.tof
        now = time.monotonic()
        age = max(st["rx"] - rx, now - self.tof_received_at, 0.0)
        self.tof_hist.append((ranges, valid, age))
        while len(self.tof_hist) < TOF_HISTORY:  # first ticks: empty frames aged 1 s, as at a training reset
            self.tof_hist.appendleft((np.zeros(TOF_ZONES, np.float32), np.zeros(TOF_ZONES, np.float32), 1.0))
        obs = np.concatenate(
            [
                np.asarray(st["gyro"], np.float32),
                np.asarray(st["grav"], np.float32),
                q,
                dq,
                self.prev_action,
                requested,
                np.zeros(10, np.float32),
                self.prev_applied,
                *[np.concatenate([r, v]) for r, v, _ in self.tof_hist],
                np.array([a for _, _, a in self.tof_hist], np.float32),
            ]
        ).astype(np.float32)
        if obs.shape != (OBS_LEN,) or not np.isfinite(obs).all():
            raise ValueError("Navigation observations must contain 1343 finite values")
        if self.norm is not None:
            self.max_z = float(np.abs((obs - self.norm[0]) / self.norm[1]).max())
        action = self.sess.run(None, {self.input_name: obs[None, :]})[0][0].astype(np.float32)
        dt = 0.02 if self.prev_rx is None else min(max(st["rx"] - self.prev_rx, 0.01), 0.1)
        self.prev_rx = st["rx"]
        cmd = command_from_action(action, requested, self.prev_applied, True, age, dt)
        if now - st.get("received_at", now) > TOF_MAX_AGE_S:
            cmd[:] = 0.0
        self.prev_action = action
        return action, cmd, age, ranges, valid


def centre_min(ranges, valid):
    """Nearest valid reading in the centre four columns, floor rows left out: what the brake checks."""
    rows = [r for r in range(8) if r not in BRAKE_IGNORE_ROWS]
    r, v = ranges.reshape(8, 8)[rows][:, 2:6], valid.reshape(8, 8)[rows][:, 2:6] > 0
    return float(r[v].min()) if v.any() else None


def front_summary(ranges, valid):
    """Nearest reading in the left, centre and right of the view, for the console."""
    r, v = ranges.reshape(8, 8), valid.reshape(8, 8) > 0
    out = []
    for name, cols in (("L", slice(0, 3)), ("C", slice(3, 5)), ("R", slice(5, 8))):
        vals = r[:, cols][v[:, cols]]
        out.append(f"{name}{vals.min():.2f}" if vals.size else f"{name} --")
    return " ".join(out)


# --- the robot side ---------------------------------------------------------------------------


class Robot:
    """SSH access to the robot, and handing its gamepad to the proxy and back."""

    def __init__(self, host, user):
        self.target = f"{user}@{host}"

    def ssh(self, cmd, **kw):
        return subprocess.run([*SSH, "-o", "ConnectTimeout=8", self.target, cmd], **kw)

    def stream(self, script, *args):
        """Run one of the helpers on the robot; its stdout is JSON lines."""
        cmd = f"python3 -u {ROBOT_DIR}/{script} " + " ".join(shlex.quote(a) for a in args)
        return subprocess.Popen([*SSH, self.target, cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def install_helpers(self):
        """Copy the two robot-side helpers to ~/tof-nav on the robot (every run, so they stay current)."""
        self.ssh(f"mkdir -p {ROBOT_DIR}", check=True)
        helpers = [str(HERE / name) for name in ("relay_on_duck.py", "padproxy_on_duck.py")]
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", *helpers, f"{self.target}:{ROBOT_DIR}/"], check=True)

    def hand_gamepad_to_proxy(self):
        """Stop the gamepad daemon and start a copy of it that talks to the proxy. The copy exits when the
        proxy does, and its unit then starts the normal daemon again. Asks for the robot's sudo password."""
        d, unit = GAMEPAD_DAEMON, NAV_UNIT
        run_copy = (
            f"for i in $(seq 1 60); do [ -S {PROXY_SOCK} ] && break; sleep 1; done; "
            f"exec /opt/robot/daemon/current/bin/{d} --socket {PROXY_SOCK} --tap-socket /run/{unit}/pad.sock"
        )
        inner = (
            f"systemctl stop {unit} 2>/dev/null; systemctl reset-failed {unit} 2>/dev/null; "
            f"systemctl stop {d} && systemd-run --quiet --collect --unit={unit} --uid={d} --gid={d} "
            f"-p 'SupplementaryGroups=input robot' -p RuntimeDirectory={unit} "
            f"-p 'ExecStopPost=+/bin/systemctl start {d}' /bin/sh -c "
            + shlex.quote(run_copy)
            + f" || {{ systemctl start {d}; exit 1; }}"
        )
        ssh_t = ["ssh", "-t", "-o", "LogLevel=ERROR", self.target, "sudo sh -c " + shlex.quote(inner)]
        return subprocess.run(ssh_t).returncode == 0

    def gamepad_daemon_active(self):
        out = self.ssh(f"systemctl is-active {GAMEPAD_DAEMON}", capture_output=True, text=True).stdout
        return out.strip() == "active"


def read_lines(proc, on_msg):
    """Read a helper's JSON lines on a thread, so the control loop only sees the newest state."""

    def run():
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                print("  robot:", line.rstrip())
                continue
            try:
                msg["received_at"] = time.monotonic()
                on_msg(msg)
            except Exception as e:  # one bad message must not stop the stream
                print(f"  error handling {msg.get('k')!r}: {e!r}")

    threading.Thread(target=run, daemon=True).start()


# --- one run ----------------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, help="the robot's hostname or IP address")
    ap.add_argument("--user", default="microduck", help="SSH user on the robot")
    ap.add_argument("--model", default=str(HERE / "policy.onnx"), help="the ToF navigation policy (.onnx)")
    ap.add_argument("--max-vx", type=float, default=0.15, help="forward speed cap while the policy drives, m/s (≤ 0.2)")
    ap.add_argument("--stop-dist", type=float, default=0.20, help="stop forward motion when anything is this close, m")
    ap.add_argument(
        "--stick-turn-override",
        action="store_true",
        help="while you turn the stick, your turn is sent instead of the policy's (it still sets the speed)",
    )
    ap.add_argument("--override-deadband", type=float, default=0.1, help="stick turn (0-1) where the override starts")
    ap.add_argument(
        "--toggle-button",
        default="BTN_NORTH",
        help="button that switches the policy on/off (evdev name; BTN_NORTH is X on a Switch Pro controller)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="check only: stream the sensor, run the policy and log what it would send; no movement commands, "
        "the gamepad is untouched and no sudo is needed",
    )
    ap.add_argument(
        "--request",
        nargs=3,
        type=float,
        default=[0.15, 0.0, 0.0],
        metavar=("VX", "VY", "WZ"),
        help="with --dry-run: the stick command the policy is given (default: forward at 0.15 m/s)",
    )
    ap.add_argument("--seconds", type=float, default=0, help="end the run after this long (0 = until Ctrl-C)")
    ap.add_argument("--log-dir", default=str(HERE / "runs"), help="where the per-tick CSV log goes")
    args = ap.parse_args()
    if not math.isfinite(args.max_vx) or not 0.0 < args.max_vx <= 0.2:
        ap.error("--max-vx must be finite and in (0, 0.2] m/s")
    if not math.isfinite(args.stop_dist) or args.stop_dist < 0.0:
        ap.error("--stop-dist must be finite and non-negative")
    if not math.isfinite(args.seconds) or args.seconds < 0.0:
        ap.error("--seconds must be finite and non-negative")
    if not math.isfinite(args.override_deadband) or not 0.0 <= args.override_deadband <= 1.0:
        ap.error("--override-deadband must be in [0, 1]")
    if not np.isfinite(args.request).all():
        ap.error("--request values must be finite")
    return args


def connect(args):
    """Load the policy and reach the robot, with a clear message for each thing that can be wrong."""
    CAP[0] = min(args.max_vx, 0.2)
    if not pathlib.Path(args.model).exists():
        sys.exit(f"no policy at {args.model}; put it there or pass --model path/to/policy.onnx")
    nav = Navigator(args.model)
    robot = Robot(args.robot, args.user)
    check = robot.ssh("true", capture_output=True, text=True)
    if check.returncode != 0:
        if "Permission denied" in check.stderr:
            sys.exit(f"cannot SSH to {robot.target} without a password; run: ssh-copy-id {robot.target}")
        sys.exit(f"cannot reach {robot.target}: is the robot on and on this network? ({check.stderr.strip()})")
    robot.install_helpers()
    return nav, robot


def start_proxy(robot, args, pad):
    """Start padproxy_on_duck.py. Its stdin takes our commands; its stdout reports your stick and buttons."""
    ready = threading.Event()

    def on_proxy(m):
        k = m.get("k")
        if k == "ready":
            ready.set()
        elif k == "req":
            pad["req"], pad["policy"] = np.array(m["v"], np.float32), m["policy"]
        elif k == "toggle":
            pad["policy"] = m["policy"]
            print(f"  >>> policy {'ON' if m['policy'] else 'OFF (plain gamepad)'}")
        elif k == "button" and m["name"] != args.toggle_button:
            print(f"  pressed {m['name']}  (--toggle-button {m['name']} would make it the policy switch)")
        elif k == "gamepad_daemon":
            print("  gamepad connected through the policy")
        elif k == "pad":
            print(f"  gamepad {m['event']}: {m.get('name') or m.get('why')}")
        elif k == "watchdog":
            print(f"  robot watchdog: {m['msg']}")
        elif k == "err" and "No such file" not in str(m.get("error")):  # the input tap appears a moment later
            print(f"  robot: {m.get('error')}")

    proxy = robot.stream("padproxy_on_duck.py", args.toggle_button)
    read_lines(proxy, on_proxy)
    if not ready.wait(15):
        stop_proxy(proxy)
        sys.exit("the robot-side proxy did not start")
    return proxy


def start_relay(robot, nav):
    """Start relay_on_duck.py: ToF frames go straight to the policy, the newest body state waits in `latest`."""
    latest, new_state = {"st": None}, threading.Condition()

    def on_relay(m):
        if m["k"] == "tof":
            nav.on_tof(m)
        elif m["k"] == "st" and m.get("q") is not None and m.get("grav") is not None:
            with new_state:
                latest["st"] = m
                new_state.notify()
        elif m["k"] == "err":
            print(f"  robot: {m.get('src')}: {m.get('error')}")

    relay = robot.stream("relay_on_duck.py")
    read_lines(relay, on_relay)
    return relay, latest, new_state


def preflight(nav, latest):
    """Wait for body state and ToF frames; return a problem to report, or None."""
    t0 = time.time()
    while time.time() - t0 < 10 and (nav.tof is None or latest["st"] is None):
        time.sleep(0.1)
    if latest["st"] is None:
        return "no body state from robotd: is the robot software running?"
    if nav.tof is None:
        return "no frames from the ToF sensor: is it fitted and is tofd running (robotctl monitor shows it)?"
    mode = latest["st"].get("policy")
    print(f"robot OK: mode {mode!r}, ToF {int(nav.tof[1].sum())}/64 zones valid")
    if mode not in ("walk", "stand"):
        print("  the robot is not standing: stand it up with the gamepad first (the policy expects it upright)")
    return None


LOG_COLUMNS = [
    "t", "tof_age", "valid_zones", "stick_vx", "stick_vy", "stick_wz", "policy_on", "policy_drives",
    "action_fwd", "action_lat", "action_yaw", "policy_vx", "policy_vy", "policy_wz",
    "sent_vx", "sent_vy", "sent_wz", "override", "safety_stop", "centre_min_m", "mode", "fallen",
]  # fmt: skip


def tick(args, nav, st, stick, pad):
    """One control step: run the policy, apply the override and the safety stop. Returns what to send."""
    requested = np.clip(stick, [0.0, -CAP[1], -CAP[2]], CAP).astype(np.float32)
    action, cmd, age, ranges, valid = nav.step(st, requested)
    sent = np.clip(cmd, -CAP, CAP)
    override = args.stick_turn_override and abs(float(stick[2])) > args.override_deadband
    if override:
        sent[2] = float(np.clip(stick[2], -CAP[2], CAP[2]))
    if age > TOF_MAX_AGE_S or time.monotonic() - st.get("received_at", time.monotonic()) > TOF_MAX_AGE_S:
        sent[:] = 0.0
    cmin, safety = centre_min(ranges, valid), ""
    if cmin is not None and cmin < args.stop_dist:
        sent[0] = sent[1] = 0.0
        safety = f"stop: {cmin:.2f} m ahead"
    drives = pad["policy"] and stick.any() and stick[0] >= 0  # the proxy only uses our command then
    # Feed back the final command, including operator overrides and the extra deployment brake.
    nav.prev_applied = (sent if drives else stick).astype(np.float32).copy()
    return sent, drives, override, safety, cmin, (action, cmd, age, ranges, valid)


def control_loop(args, nav, pad, proxy, latest, new_state, writer):
    """Run until Ctrl-C or --seconds. Returns (ticks, why it ended)."""
    start, n = time.monotonic(), 0
    try:
        while True:
            if args.seconds and time.monotonic() - start >= args.seconds:
                return n, "time limit"
            with new_state:
                new_state.wait_for(lambda: latest["st"] is not None, timeout=1.0)
                st, latest["st"] = latest["st"], None
            if st is None or nav.tof is None:
                return n, "telemetry unavailable"
            stick = np.array(args.request, np.float32) if args.dry_run else pad["req"].copy()
            sent, drives, override, safety, cmin, (action, cmd, age, ranges, valid) = tick(args, nav, st, stick, pad)
            if proxy:
                proxy.stdin.write(f"{sent[0]:.4f} {sent[1]:.4f} {sent[2]:.4f}\n")
                proxy.stdin.flush()
            el, out = time.monotonic() - start, (sent if drives else stick)
            writer.writerow([
                f"{el:.3f}", f"{age:.3f}", int(valid.sum()), *np.round(stick, 3), int(pad["policy"]), int(drives),
                *np.round(action, 3), *np.round(cmd, 3), *np.round(out, 3), int(override and drives),
                safety if drives else "", "" if cmin is None else round(cmin, 3), st.get("policy"),
                int(bool(st.get("fallen"))),
            ])  # fmt: skip
            n += 1
            if n % 10 == 0:
                if drives:
                    who = "DRY RUN   " if args.dry_run else "POLICY+YOU" if override else "POLICY    "
                else:
                    who = "gamepad   " if pad["policy"] else "policy off"
                far = "  (inputs far from training)" if nav.max_z > 5 else ""
                print(
                    f"{el:6.1f}s {who} {front_summary(ranges, valid):24s} stick [{stick[0]:+.2f} {stick[2]:+.2f}]"
                    f" -> [{out[0]:+.2f} {out[2]:+.2f}]  {safety if drives else ''}{far}"
                )
            if args.seconds and el >= args.seconds:
                return n, "time limit"
    except KeyboardInterrupt:
        return n, "Ctrl-C"


def stop_proxy(proxy):
    """Tell the proxy to stop the robot and exit; the gamepad daemon copy then hands the gamepad back."""
    try:
        proxy.stdin.write("stop\n")
        proxy.stdin.flush()
        proxy.stdin.close()
    except OSError:
        pass
    try:
        proxy.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proxy.kill()
        proxy.wait()


def main():
    args = parse_args()
    nav, robot = connect(args)
    pad = {"req": np.zeros(3, np.float32), "policy": True}
    proxy = relay = None
    try:
        proxy = None if args.dry_run else start_proxy(robot, args, pad)
        relay, latest, new_state = start_relay(robot, nav)
        problem = preflight(nav, latest)
        if problem is None and proxy:
            print("Routing the gamepad through the policy (the robot's sudo password, once):")
            if not robot.hand_gamepad_to_proxy():
                problem = "could not restart the gamepad daemon; restoring the normal daemon was requested"
        if problem:
            sys.exit(problem)

        log_dir = pathlib.Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"nav-{time.strftime('%Y%m%d-%H%M%S')}.csv"
        if args.dry_run:
            print(f"DRY RUN: no movement commands are sent. The policy is given the request {args.request}.")
            print("  Put an obstacle in front of the sensor and watch the turn it would make.")
        else:
            print(f"LIVE. Hold the stick forward and the policy steers; {args.toggle_button} switches it off/on.")
        print(
            f"  model {pathlib.Path(args.model).name}  speed cap {CAP[0]:.2f} m/s  stop distance {args.stop_dist:.2f} m"
        )
        if args.stick_turn_override:
            print("  stick-turn override ON: your turns win, the policy keeps the speed")
        print(f"  log {log}\n  Ctrl-C to end.")

        with open(log, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_COLUMNS)
            n, reason = control_loop(args, nav, pad, proxy, latest, new_state, writer)
    finally:
        if proxy:
            stop_proxy(proxy)
        if relay:
            relay.terminate()
            try:
                relay.wait(timeout=5)
            except subprocess.TimeoutExpired:
                relay.kill()
                relay.wait()
    print(f"stopped ({reason}): {n} ticks logged to {log}")
    if proxy:
        time.sleep(2)  # the daemon copy sees the proxy go and hands the gamepad back
        if robot.gamepad_daemon_active():
            print("  the gamepad is back to normal")
        else:
            print("  the gamepad daemon is not back yet; if the gamepad does nothing in a few seconds:")
            print(f"    ssh -t {robot.target} 'sudo systemctl start {GAMEPAD_DAEMON}'")


if __name__ == "__main__":
    main()
