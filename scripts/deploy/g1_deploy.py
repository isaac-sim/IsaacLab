# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a LEAPP-exported policy on a physical Unitree G1.

Inference is the official ``leapp.InferenceManager``. This script supplies the two ends the graph
cannot reach by itself: the robot's state and camera on the way in, its motors on the way out. No
simulator is launched and Isaac Sim is never imported.

Needs ``unitree_sdk2py`` with ``cyclonedds==0.10.2``, and a depth publisher on the robot's onboard
computer sending 64x38 frames as unsigned 16-bit millimetres over UDP.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import select
import socket
import struct
import sys
import termios
import threading
import time
import tty

import numpy as np

CONTROL_DT = 0.02
"""Control period [s]. The exported graph carries its history at this rate."""

DEPTH_SHAPE = (38, 64)
"""Depth frames the publisher sends and the graph expects, as ``(height, width)``."""

DEPTH_MAX_RANGE = 3.0
"""Range the graph clips at [m], and what a pixel with no return is filled with."""

DEPTH_MAX_AGE = 0.2
"""Refuse to step on a frame older than this [s]. Ten control periods."""

FALL_GRAVITY_Z = -0.7
"""Stop above this projected gravity. Upright is -1.0; this is about 45 degrees over."""

UPRIGHT_GRAVITY_Z = -0.9
"""Require this projected gravity or lower before engaging the policy."""

MOTOR_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
"""The 29 body joints ``rt/lowcmd`` carries, in its own order.

The exported graph reads and writes every joint of the articulation, fingers included, so its
tensors are matched to these by name rather than by position -- the two orders agree for the first
22 joints and then diverge, which is the kind of mismatch that drives an arm from a hand's target.
"""

DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.20,
    ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_joint": 0.87,
    "left_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.16,
    "right_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16,
}
"""The pose the ramp ends at [rad], by joint-name pattern; joints not listed go to zero.

This is the task's own initial state, and it is fixed on purpose. The operator squares the robot up
in this pose, so it has to be the same on every run.
"""

BUTTON_BITS = {"R1": 0, "L1": 1, "start": 2, "select": 3, "R2": 4, "L2": 5, "A": 8, "B": 9}
"""Bits of the 16-bit button mask at offset 2 of ``wireless_remote``, as Unitree's own parser reads
them."""

ABORT_COMBOS = (("L2", "B"), ("L1", "A"), ("select",))
"""Combinations that stop the run.

**The G1 has no hardware emergency stop.** Unitree's documented stop is a remote combination handled
by the *factory* controller, and once this program owns ``rt/lowcmd`` nothing guarantees that still
applies. So the combination is decoded here, out of every ``rt/lowstate``, and the damping is this
program's own.

The mapping changed at Motion Control 8.2.0.0: damping is ``L2 + B`` from 8.2.0.0 and ``L1 + A``
before it, where ``L1 + B`` is instead *zero torque* and drops a hanging robot. All three are
accepted, because guessing the firmware wrong must not be what decides whether the stop works.
"""


class OperatorAbort(RuntimeError):
    """The operator pressed an abort combination on the remote."""


def abort_pressed(wireless_remote) -> str | None:
    """Name the abort combination being held, or ``None``."""
    data = bytes(bytearray(wireless_remote))
    if len(data) < 4 or not any(data):
        return None
    mask = struct.unpack_from("<H", data, 2)[0]
    down = {name for name, bit in BUTTON_BITS.items() if mask & (1 << bit)}
    return next(("+".join(c) for c in ABORT_COMBOS if down.issuperset(c)), None)


class DepthStream:
    """Newest depth frame from the publisher, in metres.

    The publisher sends a whole frame in one datagram, so a frame either arrives intact or not at
    all. What this guards is staleness: a dead publisher leaves the last frame in place, and the
    array alone cannot say how old it is.
    """

    def __init__(self, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("", port))
        self._sock.settimeout(0.2)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._at = 0.0
        threading.Thread(target=self._receive, daemon=True).start()

    def _receive(self) -> None:
        count = DEPTH_SHAPE[0] * DEPTH_SHAPE[1]
        while True:
            try:
                payload = self._sock.recv(1 << 16)
            except TimeoutError:
                continue
            if len(payload) < count * 2:
                continue
            raw = np.frombuffer(payload, dtype=np.uint16, count=count, offset=len(payload) - count * 2)
            frame = raw.reshape(DEPTH_SHAPE).astype(np.float32) / 1000.0
            # A real camera reports no return as zero. The graph reads zero as a surface against the
            # lens, so the clip range is what says "nothing within range" instead.
            frame = np.where(np.isfinite(frame) & (frame > 0.0), frame, DEPTH_MAX_RANGE)
            with self._lock:
                self._frame, self._at = np.clip(frame, 0.0, DEPTH_MAX_RANGE), time.monotonic()

    def wait(self, timeout: float = 20.0) -> None:
        """Block until the first frame arrives, so a missing camera stops the run before the ramp."""
        deadline = time.monotonic() + timeout
        while self.latest() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if self.latest() is None:
            raise TimeoutError(f"no depth frame in {timeout:.0f}s; is the publisher running?")

    def latest(self) -> np.ndarray | None:
        """Newest frame [m], or ``None`` when it is missing or stale."""
        with self._lock:
            if self._frame is None or time.monotonic() - self._at > DEPTH_MAX_AGE:
                return None
            return self._frame


def heading_from_quat_xyzw(quat_xyzw: np.ndarray) -> float:
    """Body heading in the world frame [rad]."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Teleop:
    """Forward speed and heading from single keypresses, without waiting for Enter."""

    HELP = "w/s forward +/-   q/e turn left/right   space stop   r face current heading   x quit   ? help"
    SPEED_STEP = 0.1
    HEADING_STEP = math.radians(15.0)

    def __init__(self) -> None:
        self.vx = 0.0
        self.heading = 0.0
        self.running = True
        self._terminal_settings = None

    def __enter__(self) -> Teleop:
        if sys.stdin.isatty():
            self._terminal_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno(), termios.TCSANOW)
        return self

    def __exit__(self, *exc) -> None:
        if self._terminal_settings is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, self._terminal_settings)

    def command(self, quat_xyzw: np.ndarray) -> np.ndarray:
        """``(vx, vy, yaw_rate)`` [m/s, m/s, rad/s], with the yaw rate steering toward a heading.

        The command's third element is a rate, so the heading target is turned into one here against
        the measured orientation. Steering a heading rather than a rate keeps a teleoperated run on
        the distribution a fixed command trains on.
        """
        keys = ""
        while select.select([sys.stdin], [], [], 0.0)[0]:
            data = os.read(sys.stdin.fileno(), 64)
            if not data:
                break
            keys += data.decode(errors="ignore")
        for sequence, key in (("\x1b[A", "w"), ("\x1b[B", "s"), ("\x1b[D", "a"), ("\x1b[C", "d")):
            keys = keys.replace(sequence, key)
        for key in keys.lower():
            if key == "w":
                self.vx = min(1.0, self.vx + self.SPEED_STEP)
            elif key == "s":
                self.vx = max(0.0, self.vx - self.SPEED_STEP)
            elif key == "q":
                self.heading += self.HEADING_STEP
            elif key == "e":
                self.heading -= self.HEADING_STEP
            elif key == " ":
                self.vx = 0.0
            elif key == "r":
                self.heading = heading_from_quat_xyzw(quat_xyzw)
            elif key == "x":
                self.running = False
            elif key == "?":
                print(self.HELP)
        if keys:
            print(f"[cmd] vx={self.vx:+.2f} m/s  heading={math.degrees(self.heading):+.0f} deg")

        error = self.heading - heading_from_quat_xyzw(quat_xyzw)
        wrapped = (error + math.pi) % (2.0 * math.pi)
        error = math.pi if wrapped == 0.0 and error > 0.0 else wrapped - math.pi
        return np.array([self.vx, 0.0, np.clip(0.5 * error, -1.0, 1.0)], dtype=np.float32)


class G1Link:
    """``rt/lowstate`` in, ``rt/lowcmd`` out, over the Unitree SDK."""

    def __init__(self, domain_id: int, interface: str) -> None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(domain_id, interface)
        self.state = None
        ChannelSubscriber("rt/lowstate", LowState_).Init(self._on_state, 10)
        self._publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._publisher.Init()
        self._command = unitree_hg_msg_dds__LowCmd_()
        self._crc = CRC()
        self._next = None
        while self.state is None:
            time.sleep(0.02)

    def _on_state(self, message) -> None:
        self.state = message

    def read(self) -> dict:
        """Joint state and IMU, in the frames the graph expects."""
        state = self.state
        motors = state.motor_state
        quat = np.asarray(state.imu_state.quaternion[:4], dtype=np.float32)
        return {
            "joint_pos": np.array([m.q for m in motors[: len(MOTOR_NAMES)]], dtype=np.float32),
            "joint_vel": np.array([m.dq for m in motors[: len(MOTOR_NAMES)]], dtype=np.float32),
            # The IMU publishes (w, x, y, z); the graph declares (x, y, z, w).
            "root_quat_xyzw": quat[[1, 2, 3, 0]],
            "ang_vel": np.asarray(state.imu_state.gyroscope[:3], dtype=np.float32),
            # Gravity in the pelvis frame; -1.0 upright.
            "gravity_z": -float(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)),
            "mode_machine": int(state.mode_machine),
        }

    def check_abort(self) -> None:
        """Raise if the operator is holding an abort combination.

        Called from every loop that commands the robot -- the ramp, the hold and the policy -- so
        the stop works during the motion that most needs it rather than only once walking starts.
        """
        combo = abort_pressed(self.state.wireless_remote) if self.state is not None else None
        if combo is not None:
            raise OperatorAbort(f"operator pressed {combo} on the remote")

    def write(self, joint_pos, kp, kd) -> None:
        """Send one ``rt/lowcmd``. The gains are the policy's own outputs, not configuration."""
        for i in range(len(MOTOR_NAMES)):
            motor = self._command.motor_cmd[i]
            motor.mode, motor.q, motor.dq, motor.tau = 1, float(joint_pos[i]), 0.0, 0.0
            motor.kp, motor.kd = float(kp[i]), float(kd[i])
        self._command.mode_machine = self.state.mode_machine
        self._command.crc = self._crc.Crc(self._command)
        self._publisher.Write(self._command)

    def damp(self, kd, seconds: float = 1.5) -> None:
        """Bleed off energy with no stiffness, then stop commanding. Every exit goes through here."""
        print("[..] damping down")
        zero = np.zeros(len(MOTOR_NAMES), dtype=np.float32)
        for _ in range(int(seconds / CONTROL_DT)):
            self.write(zero, zero, kd)
            time.sleep(CONTROL_DT)

    def wait(self, period: float) -> None:
        """Keep the control cadence, discarding overdue deadlines after a stall."""
        if self._next is None:
            self._next = time.monotonic()
        self._next += period
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
            now = time.monotonic()
            if now - self._next < period:
                return
        if now > self._next:
            self._next = now


class GraphPolicy:
    """The exported graph, with the robot's arrays converted to and from its layout.

    The conversions are the point of this class, and each is silent when wrong -- the shapes still
    match and the robot simply walks badly:

    * the graph reads and writes **all** 43 articulation joints, so the 29 motors are placed by
      name and the fingers stay at their default pose and zero velocity;
    * depth arrives from the camera rolled 180 degrees against what the graph was trained on;
    * depth is a rank-4 ``(1, H, W, 1)`` tensor, not an image-shaped array.
    """

    def __init__(self, pipeline: str, depth_rotation: int, imu_tilt: tuple[float, float] = (0.0, 0.0)) -> None:
        import torch
        from leapp import InferenceManager

        self._torch = torch
        self._rotate = depth_rotation == 180
        pitch, roll = (math.radians(angle) / 2.0 for angle in imu_tilt)
        cp, sp, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
        self._imu_tilt = np.array([cp * cr, cp * sr, sp * cr, -sp * sr], dtype=np.float32)
        torch.set_num_threads(1)
        self.inference = InferenceManager(pipeline)

        self._ports = {}
        for direction, keys in (("inputs", self.inference.inputs), ("outputs", self.inference.outputs)):
            for key in keys:
                node, name = key.split("/", 1)
                spec = next(io for io in self.inference.models[node][direction] if io["name"] == name)
                connection = spec.get("isaaclab_connection")
                if not connection or connection in self._ports:
                    raise ValueError(f"missing or duplicate isaaclab_connection for {key}")
                self._ports[connection] = (key, spec)
        names = self._ports["write:robot:set_joint_position_target_index"][1]["element_names"][0]
        order = {name: i for i, name in enumerate(names)}
        missing = [name for name in MOTOR_NAMES if name not in order]
        if missing:
            raise KeyError(f"the export does not drive {missing}")
        self._motors = np.array([order[name] for name in MOTOR_NAMES], dtype=np.int64)
        self._default = np.array(
            [next((v for p, v in DEFAULT_JOINT_POS.items() if re.fullmatch(p, n)), 0.0) for n in names],
            dtype=np.float32,
        )
        self.default_pose = self._default[self._motors]

        # Warm the graph on synthetic inputs, before anything owns rt/lowcmd. This pays the first
        # call's cost outside the control loop and leaves self.kp and self.kd holding the gains the
        # ramp uses, which are the policy's own rather than numbers repeated here.
        for _ in range(3):
            self.step(
                self.default_pose,
                np.zeros(len(MOTOR_NAMES), np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], np.float32),
                np.zeros(3, np.float32),
                np.zeros(3, np.float32),
                np.full(DEPTH_SHAPE, DEPTH_MAX_RANGE, np.float32),
            )
        with torch.inference_mode():
            self.inference.reset()

    def step(self, joint_pos, joint_vel, quat_xyzw, ang_vel, command, depth):
        """One inference step. Returns motor targets, Kp and Kd, each ``(29,)``."""
        q, dq = self._default.copy(), np.zeros_like(self._default)
        q[self._motors], dq[self._motors] = joint_pos, joint_vel
        if self._imu_tilt[0] != 1.0:
            # R_pelvis_world = R_imu_world * inverse(R_imu_to_pelvis).
            quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
            ang_vel = np.asarray(ang_vel, dtype=np.float32)
            w, xyz = quat_xyzw[3], quat_xyzw[:3]
            offset_w, offset_xyz = self._imu_tilt[0], -self._imu_tilt[1:]
            quat_xyzw = np.concatenate(
                (
                    w * offset_xyz + offset_w * xyz + np.cross(xyz, offset_xyz),
                    [w * offset_w - np.dot(xyz, offset_xyz)],
                )
            )
            cross = np.cross(self._imu_tilt[1:], ang_vel)
            ang_vel = ang_vel + 2.0 * (self._imu_tilt[0] * cross + np.cross(self._imu_tilt[1:], cross))
        frame = np.asarray(depth, dtype=np.float32)
        if self._rotate:
            frame = frame[::-1, ::-1]

        inputs = {
            self._key(connection): self._torch.from_numpy(np.ascontiguousarray(value, np.float32))
            for connection, value in {
                "state:robot:joint_pos": q[None],
                "state:robot:joint_vel": dq[None],
                "state:robot:root_quat_w": np.asarray(quat_xyzw, np.float32)[None],
                "state:robot:root_ang_vel_b": np.asarray(ang_vel, np.float32)[None],
                "command:base_velocity": np.asarray(command, np.float32)[None],
                "state:depth_camera:output:distance_to_image_plane": frame[None, :, :, None],
            }.items()
        }
        with self._torch.inference_mode():
            outputs = self.inference.run_policy(inputs)

        target = self._motors_of(outputs, "write:robot:set_joint_position_target_index")
        self.kp = self._motors_of(outputs, "write:robot:write_joint_stiffness_to_sim_index")
        self.kd = self._motors_of(outputs, "write:robot:write_joint_damping_to_sim_index")
        # The one check worth making here: these three go straight into motor commands, and a NaN
        # from a bad joint or IMU reading would be written to the robot as a target.
        if not np.isfinite([target, self.kp, self.kd]).all():
            raise ValueError("the graph returned a non-finite target or gain")
        return target, self.kp, self.kd

    def _key(self, connection: str) -> str:
        """The graph's own name for a tensor, given the Isaac Lab connection it carries."""
        return self._ports[connection][0]

    def _motors_of(self, outputs: dict, connection: str) -> np.ndarray:
        """One of the graph's 43-wide outputs, reduced to the 29 motors in ``rt/lowcmd`` order."""
        return outputs[self._key(connection)].detach().cpu().numpy().reshape(-1)[self._motors]


def take_lowcmd(domain_id: int, probe_s: float = 3.0) -> None:
    """Refuse to start while something else is writing ``rt/lowcmd``.

    The factory controller publishes at about 1 kHz whenever the robot is not in debug mode, and a
    run started against it fights for the motors. Measured on this robot: in debug mode the topic is
    silent, so a nonzero count means the remote never took it out of its normal mode.
    """
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

    seen = [0]
    subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
    subscriber.Init(lambda _: seen.__setitem__(0, seen[0] + 1), 10)
    time.sleep(probe_s)
    subscriber.Close()
    if seen[0]:
        raise RuntimeError(
            f"{seen[0]} rt/lowcmd samples in {probe_s:.0f}s -- something else is commanding the"
            " robot. Put it in debug mode with the remote (L2+B, then L2+R2) and try again."
        )
    print("[ok] rt/lowcmd is free")


def confirm(word: str, prompt: str) -> None:
    """Block until the operator types ``word``. Anything else aborts."""
    print(prompt)
    if input(f">>> type '{word}' and press Enter: ").strip() != word:
        raise SystemExit("[..] aborted by the operator")


def bring_up(link: G1Link, pose: np.ndarray, kp, kd, ramp_s: float) -> None:
    """Interpolate to the start pose and hold it until the operator engages the policy.

    Two steps rather than one, and ``rt/lowcmd`` keeps flowing through both: the robot is lowered
    onto its feet between them, and a gap in the command stream there would drop it.
    """
    start = link.read()["joint_pos"].copy()
    steps = int(ramp_s / CONTROL_DT)
    print(f"[..] ramping to the start pose over {ramp_s:.1f}s")
    for step in range(steps):
        link.check_abort()
        link.write(start + (pose - start) * (step + 1) / steps, kp, kd)
        time.sleep(CONTROL_DT)
    if link.read()["gravity_z"] > UPRIGHT_GRAVITY_Z:
        raise RuntimeError("the robot is not upright; not engaging the policy")

    print("\n>>> HOLDING the start pose. Square the robot up and lower it onto its feet.")
    print(">>> KEEP IT SUPPORTED: this is PD with no policy, and the pose is not statically stable.")
    interactive = sys.stdin.isatty()
    if interactive:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    print(">>> type 'go2' and press Enter within 300s:")
    deadline = time.monotonic() + 300.0
    while True:
        link.check_abort()
        link.write(pose, kp, kd)
        if interactive and select.select([sys.stdin], [], [], 0.0)[0]:
            line = sys.stdin.readline()
            if not line:
                interactive = False
            elif line.strip() == "go2":
                break
            elif line.strip():
                print(f"[..] ignored {line.strip()!r}; type 'go2' to proceed")
        if time.monotonic() > deadline:
            raise TimeoutError("'go2' not received within 300s of holding")
        time.sleep(CONTROL_DT)

    print("[ok] engaging in 3s -- hands clear")
    for _ in range(int(3.0 / CONTROL_DT)):
        link.check_abort()
        link.write(pose, kp, kd)
        time.sleep(CONTROL_DT)
    if link.read()["gravity_z"] > UPRIGHT_GRAVITY_Z:
        raise RuntimeError("the robot is not upright; not engaging the policy")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pipeline", required=True, help="Exported LEAPP .yaml.")
    parser.add_argument("--interface", required=True, help="Network interface on the robot's subnet.")
    parser.add_argument("--domain_id", type=int, default=0, help="DDS domain.")
    parser.add_argument("--depth_port", type=int, default=5601, help="UDP port the publisher sends to.")
    parser.add_argument(
        "--imu_tilt",
        default="0,0",
        metavar="PITCH,ROLL",
        help="IMU pitch and roll [deg] measured with the pelvis level; default 0,0.",
    )
    parser.add_argument(
        "--depth_rotation",
        type=int,
        choices=(0, 180),
        default=180,
        help="Rotate incoming depth before the graph [deg]. The default compensates the export's"
        " own 180-degree roll against the camera stream.",
    )
    args = parser.parse_args()
    try:
        pitch, roll = (float(value) for value in args.imu_tilt.split(","))
        if not math.isfinite(pitch) or not math.isfinite(roll):
            raise ValueError
    except ValueError:
        parser.error("--imu_tilt requires finite PITCH,ROLL values in degrees")

    policy = GraphPolicy(args.pipeline, args.depth_rotation, (pitch, roll))
    link = G1Link(args.domain_id, args.interface)
    depth = DepthStream(args.depth_port)
    depth.wait()
    take_lowcmd(args.domain_id)

    print(f"[ok] mode_machine={link.read()['mode_machine']}  policy at {1 / CONTROL_DT:.0f} Hz")
    print(f"[..] teleop: {Teleop.HELP}")
    teleop = Teleop()
    confirm("go", "\n>>> Robot supported, area clear, e-stop in hand.")

    try:
        bring_up(link, policy.default_pose, policy.kp, policy.kd, ramp_s=3.0)
        teleop.heading = heading_from_quat_xyzw(link.read()["root_quat_xyzw"])
        with teleop:
            while teleop.running:
                link.check_abort()
                state = link.read()
                if state["gravity_z"] > FALL_GRAVITY_Z:
                    raise RuntimeError(f"fell (projected gravity {state['gravity_z']:+.2f})")
                command = teleop.command(state["root_quat_xyzw"])
                if not teleop.running:
                    break
                frame = depth.latest()
                if frame is None:
                    raise RuntimeError(f"no depth frame within {DEPTH_MAX_AGE * 1e3:.0f} ms")
                joint_pos, kp, kd = policy.step(
                    state["joint_pos"],
                    state["joint_vel"],
                    state["root_quat_xyzw"],
                    state["ang_vel"],
                    command,
                    frame,
                )
                link.write(joint_pos, kp, kd)
                link.wait(CONTROL_DT)
    except (KeyboardInterrupt, OperatorAbort, RuntimeError, TimeoutError) as error:
        print(f"\n[..] {error or 'interrupted'}")
    finally:
        link.damp(policy.kd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
