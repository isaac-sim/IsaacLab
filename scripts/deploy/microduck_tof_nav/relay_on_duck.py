# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runs on the robot (Python standard library only), started by nav_pad.py over SSH.

Read-only: streams body state at 50 Hz and every ToF frame to stdout as JSON lines. It subscribes to
robotd and tofd and never sends a command.
"""

import json
import socket
import sys
import threading
import time

ROBOTD = "/run/robotd.sock"
TOFD = "/run/tofd/tof.sock"
HZ = 50

out_lock = threading.Lock()


def emit(obj):
    with out_lock:
        sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
        sys.stdout.flush()


def stream(path, method, params, want, on_msg):
    while True:
        try:
            with socket.socket(socket.AF_UNIX) as s:
                s.connect(path)
                s.sendall((json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}) + "\n").encode())
                buf = b""
                while True:
                    chunk = s.recv(1 << 20)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        try:
                            m = json.loads(line)
                        except ValueError:
                            continue
                        if m.get("method") == want:
                            on_msg(m["params"])
        except Exception as e:  # reconnect rather than die: daemon restarts are normal
            emit({"k": "err", "src": method, "error": str(e)})
        time.sleep(1)


def on_tof(p):
    # Stamped on the robot's own clock when the frame arrives, so frame age needs no clock sync.
    emit({"k": "tof", "rx": time.monotonic(), "mm": p.get("distance_mm"), "status": p.get("status")})


def on_state(p):
    safety, imu = p.get("safety") or {}, p.get("imu") or {}
    emit(
        {
            "k": "st",
            "rx": time.monotonic(),
            "q": p.get("joints"),
            "dq": p.get("velocities"),
            "gyro": imu.get("gyro"),
            "grav": safety.get("gravity"),
            "fallen": safety.get("fallen"),
            "policy": p.get("policy"),
        }
    )


def main():
    threading.Thread(target=stream, args=(TOFD, "tof.stream", {}, "tof.frame", on_tof), daemon=True).start()
    stream(ROBOTD, "robot.subscribe", {"hz": HZ}, "robot.state", on_state)


if __name__ == "__main__":
    main()
