# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runs on the robot (Python standard library only), started by nav_pad.py over SSH.

Sits between the gamepad daemon and the robot daemon, so the gamepad is the operator and the
navigation policy adjusts what it asks for:

  gamepad daemon --socket /tmp/tof-nav/robotd.sock  ->  this proxy  ->  /run/robotd.sock

Everything the gamepad daemon sends passes through unchanged (stand/sit, Start, Select, head modes, sounds, the
sit when the gamepad is lost) EXCEPT robot.move while the policy is on and the stick asks for forward
or turning motion. Then the stick is reported to the laptop ({"k": "req"}) and the laptop's command
(stdin lines "vx vy vyaw") is sent instead.

It falls back to stopping, never to an unchecked command:
  * policy on, stick deflected, no laptop command for WATCHDOG_S  ->  robot.stop
  * stdin closed or a "stop" line  ->  robot.stop and exit (the gamepad daemon then loses its socket)
Stick centred, pulled back (the sensor only looks forward), or policy off: the gamepad drives as usual.

The toggle button (argv[1], an evdev name) is read from the gamepad daemon's raw input tap: each press switches the
policy, and the robot chirps (on) or coos (off).
"""

import contextlib
import grp
import json
import math
import os
import select
import socket
import sys
import threading
import time

ROBOTD = "/run/robotd.sock"
SOCK_DIR = "/tmp/tof-nav"
SOCK = SOCK_DIR + "/robotd.sock"
TAP = "/run/padd-nav/pad.sock"  # codespell:ignore  (the daemon copy's input tap)
SOUND_OFF = "coo"  # codespell:ignore  (the robot's soft "off" sound)
WATCHDOG_S = 0.25
REQUEST_FRESH_S = 0.3  # the daemon repeats a deflected stick at 50 Hz; older means centred
MAX_VX, MAX_VY, MAX_VYAW = 0.2, 0.08, 0.8  # last line of defence; the laptop caps tighter
TOGGLE = sys.argv[1] if len(sys.argv) > 1 else "BTN_NORTH"

out_lock, state_lock = threading.Lock(), threading.Lock()
state = {"policy": True, "req": (0.0, 0.0, 0.0), "req_t": 0.0, "cmd_t": 0.0, "nav_driving": False}


def emit(obj):
    with out_lock:
        sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
        sys.stdout.flush()


def clamp(v, lim):
    if not math.isfinite(v):
        raise ValueError("Command values must be finite")
    return max(-lim, min(lim, v))


def rpc(method, params=None):
    msg = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return (json.dumps(msg) + "\n").encode()


nav = None  # Connected by main(), so importing this module never touches the robot.
nav_lock = threading.Lock()


def nav_send(method, params=None):
    with nav_lock:
        nav.sendall(rpc(method, params))


def policy_drives():
    """True when the policy is on and the stick asks for forward or turning motion."""
    vx, vy, vyaw = state["req"]
    fresh = time.monotonic() - state["req_t"] < REQUEST_FRESH_S
    return state["policy"] and fresh and (abs(vx) + abs(vy) + abs(vyaw) > 0) and vx >= 0


def pump(src, dst, on_line=None):
    buf = b""
    try:
        while True:
            chunk = src.recv(65536)
            if not chunk:
                break
            if on_line is None:
                dst.sendall(chunk)
                continue
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if on_line(line):
                    dst.sendall(line + b"\n")
    except OSError:
        pass
    finally:
        if on_line is not None:
            with state_lock:
                state["req"], state["nav_driving"] = (0.0, 0.0, 0.0), False
            with contextlib.suppress(OSError):
                nav_send("robot.stop")
        for s in (src, dst):
            with contextlib.suppress(OSError):
                s.close()


def from_gamepad(line):
    """Return True to pass the line on to robotd unchanged."""
    try:
        m = json.loads(line)
    except ValueError:
        return True
    method = m.get("method")
    if method == "robot.move":
        p = m.get("params") or {}
        try:
            req = (float(p.get("vx", 0)), float(p.get("vy", 0)), float(p.get("vyaw", 0)))
            if not all(math.isfinite(v) for v in req):
                raise ValueError("Non-finite gamepad command")
        except (TypeError, ValueError):
            with state_lock:
                state["req"], state["nav_driving"] = (0.0, 0.0, 0.0), False
            nav_send("robot.stop")
            return False
        with state_lock:
            state["req"], state["req_t"] = req, time.monotonic()
            drives = policy_drives()
            if not drives:
                state["nav_driving"] = False
        emit({"k": "req", "v": req, "policy": state["policy"], "policy_drives": drives})
        return not drives
    if method == "robot.stop":
        with state_lock:
            state["req"], state["req_t"], state["nav_driving"] = (0.0, 0.0, 0.0), time.monotonic(), False
        emit({"k": "req", "v": (0.0, 0.0, 0.0), "policy": state["policy"], "policy_drives": False})
    return True


def serve():
    os.makedirs(SOCK_DIR, exist_ok=True)
    robot_gid = grp.getgrnam("robot").gr_gid  # the gamepad daemon runs in the robot group
    os.chown(SOCK_DIR, -1, robot_gid)
    os.chmod(SOCK_DIR, 0o750)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(SOCK)
    os.chown(SOCK, -1, robot_gid)
    os.chmod(SOCK, 0o660)
    srv.listen(4)
    emit({"k": "ready"})
    while True:
        client, _ = srv.accept()
        upstream = socket.socket(socket.AF_UNIX)
        upstream.connect(ROBOTD)
        emit({"k": "gamepad_daemon", "event": "connected"})
        threading.Thread(target=pump, args=(client, upstream, from_gamepad), daemon=True).start()
        threading.Thread(target=pump, args=(upstream, client), daemon=True).start()


def tap():
    """Switch the policy from a gamepad button, through the gamepad daemon's raw input tap."""
    last_err = None
    while True:
        try:
            s = socket.socket(socket.AF_UNIX)
            s.connect(TAP)
            s.sendall(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "pad.input"}).encode() + b"\n")
            buf = b""
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    try:
                        p = json.loads(line).get("params") or {}
                    except ValueError:
                        continue
                    if p.get("report") == "attached":
                        emit({"k": "pad", "event": "attached", "name": p["device"]["name"]})
                    elif p.get("report") == "detached":
                        emit({"k": "pad", "event": "detached", "why": p.get("why")})
                    for ev in p.get("events") or ():
                        if ev.get("kind") == 1 and ev.get("value") == 1:  # a key press
                            emit({"k": "button", "name": ev.get("name")})
                            if ev.get("name") == TOGGLE:
                                with state_lock:
                                    state["policy"] = not state["policy"]
                                    on = state["policy"]
                                nav_send("robot.sound", {"tag": "chirp" if on else SOUND_OFF})
                                emit({"k": "toggle", "policy": on})
        except OSError as e:
            if str(e) != last_err:  # the daemon copy may start after the proxy: say so once, keep trying
                emit({"k": "err", "src": "tap", "error": str(e)})
            last_err = str(e)
        time.sleep(1)


def main():
    """Forward navigation commands until the laptop exits or its input stream closes."""
    global nav
    nav = socket.socket(socket.AF_UNIX)
    nav.connect(ROBOTD)
    threading.Thread(target=serve, daemon=True).start()
    threading.Thread(target=tap, daemon=True).start()

    fd, pending = sys.stdin.fileno(), b""
    try:
        while True:
            r, _, _ = select.select([fd], [], [], 0.05)
            now = time.monotonic()
            if r:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                pending += chunk
                *lines, pending = pending.split(b"\n")
                if any(line.strip() == b"stop" for line in lines):
                    break
                if lines:
                    try:
                        vx, vy, vyaw = (float(x) for x in lines[-1].split())
                        if not all(math.isfinite(v) for v in (vx, vy, vyaw)):
                            raise ValueError("Non-finite navigation command")
                    except ValueError:
                        with state_lock:
                            state["nav_driving"] = False
                        nav_send("robot.stop")
                        continue
                    with state_lock:
                        state["cmd_t"] = now
                        drives = state["nav_driving"] = policy_drives()
                        if drives:
                            nav_send(
                                "robot.move",
                                {"vx": clamp(vx, MAX_VX), "vy": clamp(vy, MAX_VY), "vyaw": clamp(vyaw, MAX_VYAW)},
                            )
            with state_lock:
                stale = state["nav_driving"] and (not policy_drives() or now - state["cmd_t"] > WATCHDOG_S)
                if stale:
                    state["nav_driving"] = False
                    nav_send("robot.stop")
            if stale:
                emit({"k": "watchdog", "msg": "stale request or policy command, stopped"})
    finally:
        with contextlib.suppress(OSError):
            nav_send("robot.stop")
        nav.close()
        emit({"k": "bye"})


if __name__ == "__main__":
    main()
