# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check that a task behaves the same way on two physics backends, before spending a training run.

A policy trained on one backend and evaluated on another can differ for two very different reasons:
the dynamics genuinely differ, or one backend never applied part of the configuration. The second is
much more common and much harder to see, because the config that was ignored still reads back
correctly from ``env_cfg`` -- only the simulation disagrees.

This runs four checks that need no policy and no training, on whichever backend ``physics=NAME``
selects, and writes a JSON report. Run it once per backend and pass both reports to ``--compare``.

1. **Drive stability.** With the action held at zero the robot should stand still. Mean and peak
   joint speed measure whether it does; a drive the solver cannot hold shows up here as speeds no
   motor could produce, and the worst joints name themselves.
2. **Authored velocity limits in force.** Every joint whose measured peak exceeds the limit its own
   asset authors. A limit that is present in USD but not enforced after spawn lets a joint run away,
   and because the limit is right there in the file the runaway looks impossible on inspection.
3. **Contact reporting.** The fraction of the time the contact sensor reports its load-bearing
   bodies as touching while the robot stands, and the force it reports against the robot's weight.
   Air-time rewards and ground-contact terminations read this sensor rather than the geometry, so a
   backend that reports contact intermittently is training against a different objective.
4. **Joint-property read-back.** Writes a joint property and reads it straight back. A backend that
   returns the old value cannot be trusted to report its own state, which matters because every
   other check here -- and every domain randomization -- is read through the same path.

.. code-block:: bash

    # once per backend
    ./isaaclab.sh -p scripts/tools/check_physics_parity.py --task <TASK> \
        --report /tmp/newton.json physics=newton_mjwarp
    ./isaaclab.sh -p scripts/tools/check_physics_parity.py --task <TASK> \
        --report /tmp/physx.json physics=physx

    # then the side-by-side
    ./isaaclab.sh -p scripts/tools/check_physics_parity.py --compare /tmp/newton.json /tmp/physx.json

The launch order matters and is the reason this script does not use :class:`~isaaclab.app.AppLauncher`
directly. ``AppLauncher(args).app`` starts Kit before Hydra has resolved anything, so the launcher's
scan sees no physics configuration and falls back to Isaac Sim -- which makes ``physics=physx``
resolve to the Kit-based ``PhysxCfg``. A headless training run resolves the config first, finds
nothing that needs Kit, and gets the kitless ``OvPhysxCfg`` instead. Those are two different PhysX
backends with different defaults, so a probe built the first way says nothing about a run built the
second way. This resolves the config first, exactly as the trainer does.
"""

import argparse
import json
import sys

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default=None, help="Task id to check. Required unless --compare is used.")
parser.add_argument("--report", default=None, help="Write the JSON report here.")
parser.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None, help="Print two reports side by side.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--settle", type=int, default=150, help="Steps to discard before measuring.")
parser.add_argument("--measure", type=int, default=100)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--device", default="cuda:0")
parser.add_argument(
    "--contact_bodies",
    default=None,
    help="Regex naming the bodies whose contact to report. Defaults to those carrying at least"
    " --bearing_pct of the robot's weight, which on a standing biped is its feet.",
)
parser.add_argument("--bearing_pct", type=float, default=2.0)
args, hydra_args = parser.parse_known_args()


def _fmt(v, spec=""):
    return "n/a" if v is None else f"{v:{spec}}" if spec else str(v)


def compare(path_a: str, path_b: str) -> int:
    """Print two reports side by side and return a shell exit code."""
    reports = []
    for path in (path_a, path_b):
        with open(path) as handle:
            reports.append(json.load(handle))
    a, b = reports
    print(f"\n{'':38s} {a['backend']:>22s} {b['backend']:>22s}")
    print("-" * 86)
    rows = [
        ("mean |joint speed| holding [rad/s]", "hold_mean_joint_speed", ".4f"),
        ("peak |joint speed| holding [rad/s]", "hold_peak_joint_speed", ".1f"),
        ("joints over their authored limit", "over_limit", "d"),
        ("contact reported [% of steps]", "contact_duty_pct", ".1f"),
        ("contact force / weight [%]", "contact_force_pct", ".1f"),
        ("settled root height [m]", "hold_root_height", ".5f"),
        ("resets while holding [% env-steps]", "reset_pct", ".2f"),
        ("joint property reads back as written", "readback_honest", ""),
    ]
    for label, key, spec in rows:
        print(f"{label:38s} {_fmt(a.get(key), spec):>22s} {_fmt(b.get(key), spec):>22s}")

    print()
    findings = []
    for rep in (a, b):
        if (rep.get("hold_peak_joint_speed") or 0) > 100.0:
            findings.append(
                f"{rep['backend']}: peak joint speed {rep['hold_peak_joint_speed']:.0f} rad/s while merely"
                f" holding the default pose -- worst: {', '.join(rep['hold_worst_joints'][:3])}"
            )
        if rep.get("over_limit"):
            findings.append(
                f"{rep['backend']}: {rep['over_limit']} joint(s) exceed the velocity limit their own asset"
                f" authors, by up to {rep['over_limit_factor']:.0f}x -- {', '.join(rep['over_limit_names'][:4])}"
            )
        if rep.get("contact_duty_pct") is not None and rep["contact_duty_pct"] < 60.0:
            findings.append(
                f"{rep['backend']}: the contact sensor reports contact only"
                f" {rep['contact_duty_pct']:.0f}% of the time while the robot stands, so any air-time"
                " reward or ground-contact termination is reading chatter"
            )
        if rep.get("readback_honest") is False:
            findings.append(
                f"{rep['backend']}: a joint property read back as the old value after being written, so"
                " this backend's reported joint state cannot be taken at face value"
            )

    set_a, set_b = a.get("contact_bodies"), b.get("contact_bodies")
    if set_a is not None and set_b is not None and set(set_a) != set(set_b):
        findings.append(
            "the two runs measured contact on different bodies"
            f" ({len(set_a)} against {len(set_b)}), because the backends left the robot in different"
            " states -- re-run both with --contact_bodies to pin the set before reading row 4 and 5"
        )

    if findings:
        print("FINDINGS")
        for f in findings:
            print(f"  * {f}")
        print(
            "\nRemedy for a drive the solver cannot hold: take the stiffness off that actuator group"
            "\n(``stiffness=0.0`` and a small ``damping``). Joints no policy needs to position -- hands"
            "\nand fingers on a locomotion task -- cost nothing this way, and the change is inert on a"
            "\nbackend that was already stable, which is what makes it safe to apply to both."
        )
        return 1
    print("No parity problem found by these four checks.")
    return 0


if args.compare:
    raise SystemExit(compare(*args.compare))

if not args.task:
    parser.error("--task is required unless --compare is used")

sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import launch_simulation, scan  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def _t(x):
    """Return a torch view of a backend array, which may already be one."""
    return x.torch if hasattr(x, "torch") else x


def _authored_velocity_limits(robot):
    """Return the per-joint velocity limit the asset authors [rad/s], or ``None`` if unavailable."""
    for name in ("joint_velocity_limits", "joint_vel_limits"):
        value = getattr(robot.data, name, None)
        if value is not None:
            limits = value.torch if hasattr(value, "torch") else value
            return limits[0] if limits.ndim == 2 else limits
    return None


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point", play_mode=True)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    # Flat ground and no disturbance: the checks are about what the backend applied, and a rough
    # terrain or a push would put a real difference and an artefact in the same number.
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    if getattr(env_cfg, "curriculum", None) is not None:
        for term in [t for t in vars(env_cfg.curriculum) if not t.startswith("_")]:
            setattr(env_cfg.curriculum, term, None)
    events = getattr(env_cfg, "events", None)
    if events is not None:
        if getattr(events, "push_robot", None) is not None:
            events.push_robot = None
        if getattr(events, "reset_base", None) is not None:
            events.reset_base.params["pose_range"] = {}
            events.reset_base.params["velocity_range"] = {}
        if getattr(events, "reset_robot_joints", None) is not None:
            events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
            events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    launcher_args = {"device": args.device}
    scan(env_cfg, launcher_args)
    with launch_simulation(env_cfg, launcher_args):
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        robot = env.scene["robot"]
        joint_names = list(robot.joint_names)
        backend = type(env.sim.cfg.physics).__name__
        report: dict = {"task": args.task, "backend": backend, "num_envs": env.num_envs}
        print(f"\nbackend: {backend}   task: {args.task}   envs: {env.num_envs}\n")

        # 1 to 3. hold the default pose ---------------------------------------------------------
        contacts = env.scene.sensors.get("contact_forces")
        tracked = list(range(len(contacts.body_names))) if contacts is not None else []
        env.reset()
        zero = torch.zeros(env.num_envs, env.action_space.shape[1], device=env.device)
        acc_speed = torch.zeros((), device=env.device)
        peak_speed = torch.zeros((), device=env.device)
        peak_per_joint = torch.zeros(len(joint_names), device=env.device)
        acc_height = torch.zeros((), device=env.device)
        n_bodies = len(tracked)
        acc_force = torch.zeros(max(n_bodies, 1), device=env.device)
        acc_touch = torch.zeros(max(n_bodies, 1), device=env.device)
        peak_force = torch.zeros(max(n_bodies, 1), device=env.device)
        resets = 0
        steps = 0
        with torch.inference_mode():
            for step in range(args.settle + args.measure):
                _, _, terminated, truncated, _ = env.step(zero)
                if step < args.settle:
                    continue
                resets += int((_t(terminated) | _t(truncated)).sum().item())
                speed = _t(robot.data.joint_vel).abs()
                acc_speed += speed.mean()
                peak_speed = torch.maximum(peak_speed, speed.max())
                peak_per_joint = torch.maximum(peak_per_joint, speed.max(dim=0).values)
                acc_height += _t(robot.data.root_pos_w)[:, 2].mean()
                if contacts is not None:
                    force = _t(contacts.data.net_forces_w)[:, tracked, :].norm(dim=-1)
                    acc_force += force.mean(dim=0)
                    peak_force = torch.maximum(peak_force, force.max(dim=0).values)
                    acc_touch += (_t(contacts.data.current_air_time)[:, tracked] <= 0.0).float().mean(dim=0)
                steps += 1

        weight = _t(robot.data.default_mass).sum(dim=1).mean().item() * 9.81
        worst = torch.argsort(peak_per_joint, descending=True)[:5].tolist()
        report["hold_mean_joint_speed"] = acc_speed.item() / steps
        report["hold_peak_joint_speed"] = peak_speed.item()
        report["hold_worst_joints"] = [joint_names[i] for i in worst]
        report["hold_root_height"] = acc_height.item() / steps
        report["reset_pct"] = 100.0 * resets / (steps * env.num_envs)
        print(
            f"[1] holding the default pose: mean |joint speed| {report['hold_mean_joint_speed']:.4f} rad/s,"
            f" peak {report['hold_peak_joint_speed']:.1f} rad/s"
        )
        for i in worst:
            print(f"    {joint_names[i]:34s} peak {peak_per_joint[i].item():10.2f} rad/s")

        limits = _authored_velocity_limits(robot)
        if limits is None:
            report["over_limit"] = None
            print("[2] authored velocity limits: this backend does not expose them; skipped")
        else:
            ratio = peak_per_joint / limits.clamp(min=1e-9)
            over = (ratio > 1.0).nonzero().flatten().tolist()
            order = sorted(over, key=lambda i: -ratio[i].item())
            report["over_limit"] = len(over)
            report["over_limit_names"] = [joint_names[i] for i in order]
            report["over_limit_factor"] = ratio.max().item() if over else 1.0
            print(f"[2] authored velocity limits: {len(over)}/{len(joint_names)} joints exceed their own")
            for i in order[:5]:
                print(
                    f"    {joint_names[i]:34s} peak {peak_per_joint[i].item():9.1f} rad/s against a limit of"
                    f" {limits[i].item():8.1f} ({ratio[i].item():.0f}x)"
                )

        if contacts is not None:
            # Only the bodies that touch the ground at all can say anything about contact reporting;
            # a torso that never touches would otherwise drag the duty cycle down on both backends.
            if args.contact_bodies:
                import re

                bearing = [i for i in tracked if re.fullmatch(args.contact_bodies, contacts.body_names[i])]
            else:
                # Bodies that merely brush the ground are not evidence about contact reporting, and
                # on an unstable backend a flailing torso would otherwise join the set and make the
                # two backends' numbers describe different bodies. Take real load bearers only.
                floor = args.bearing_pct / 100.0 * weight * steps
                bearing = (acc_force > floor).nonzero().flatten().tolist()
            if bearing:
                report["contact_duty_pct"] = 100.0 * (acc_touch[bearing].mean().item() / steps)
                report["contact_force_pct"] = 100.0 * (acc_force[bearing].sum().item() / steps) / weight
                report["contact_bodies"] = [contacts.body_names[i] for i in bearing]
                print(
                    f"[3] contact on {len(bearing)} load-bearing body(ies)"
                    f" ({', '.join(report['contact_bodies'][:4])}): reported"
                    f" {report['contact_duty_pct']:.1f}% of steps, carrying"
                    f" {report['contact_force_pct']:.1f}% of the robot's {weight:.1f} N"
                )
            else:
                print("[3] contact: nothing touched the ground during the window; skipped")
        else:
            print("[3] contact: the scene has no sensor named 'contact_forces'; skipped")

        # 4. does a written joint property read back? ---------------------------------------------
        before = _t(robot.data.joint_armature).clone()
        probe = 0.05 if before.max().item() != 0.05 else 0.07
        robot.write_joint_armature_to_sim(torch.full_like(before, probe))
        env.sim.step()
        env.scene.update(env.sim.get_physics_dt())
        read = _t(robot.data.joint_armature)[0]
        honest = int((read - probe).abs().lt(1e-9).sum().item())
        report["readback_honest"] = honest == len(joint_names)
        print(f"[4] joint-property read-back: {honest}/{len(joint_names)} joints report the value just written")
        if honest != len(joint_names):
            print("    this backend's reported joint state does not track what was written to it")
        robot.write_joint_armature_to_sim(before)

        if args.report:
            with open(args.report, "w") as handle:
                json.dump(report, handle, indent=1)
            print(f"\nwrote {args.report}")
        env.close()


main()
