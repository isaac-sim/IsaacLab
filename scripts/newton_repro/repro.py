# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NaN reproduction script — standalone, no IsaacLab runtime dependencies."""

import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import newton.viewer
import torch
import warp as wp
import yaml
from pxr import Usd

_repro_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repro_dir)
from newton_replicate import _build_and_label
from newton_manager import NewtonSim


def main():
    parser = argparse.ArgumentParser(description="NaN repro — standalone")
    parser.add_argument("--env", type=str, required=True, help="Env name under envs/ (e.g. rough_anymal_d)")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record", type=str, default=None, help="Record to file (e.g. recording.bin)")
    args = parser.parse_args()

    device = "cuda:0"
    torch.cuda.set_device(0)

    env_dir = os.path.join(_repro_dir, "envs", args.env)
    sys.path.insert(0, env_dir)
    import mdp

    with open(os.path.join(env_dir, "env.yaml")) as f:
        cfg = yaml.unsafe_load(f)

    num_envs = args.num_envs or cfg["scene"]["num_envs"]
    decimation = cfg["decimation"]
    physics_dt = cfg["sim"]["dt"]
    episode_length_s = cfg["episode_length_s"]
    phys = cfg["sim"]["physics"]
    solver_cfg = SimpleNamespace(**phys["solver_cfg"])
    collision_cfg = SimpleNamespace(**phys["collision_cfg"]) if phys.get("collision_cfg") else None
    num_substeps = phys.get("num_substeps", 1)

    stage = Usd.Stage.Open(os.path.join(env_dir, "stage.usd"))
    info = json.load(open(os.path.join(env_dir, "cloner_info.json")))

    env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
    mapping = torch.ones(len(info["sources"]), num_envs, dtype=torch.bool, device=device)
    env_spacing = info.get("env_spacing", cfg["scene"]["env_spacing"]) * 2
    num_rows = int(math.ceil(num_envs / math.sqrt(num_envs)))
    num_cols = int(math.ceil(num_envs / num_rows))
    ii, jj = torch.meshgrid(
        torch.arange(num_rows, dtype=torch.float32, device=device),
        torch.arange(num_cols, dtype=torch.float32, device=device),
        indexing="ij",
    )
    positions = torch.stack([
        -(ii.flatten()[:num_envs] - (num_rows - 1) / 2) * env_spacing,
        (jj.flatten()[:num_envs] - (num_cols - 1) / 2) * env_spacing,
        torch.zeros(num_envs, device=device),
    ], dim=1)

    builder, _ = _build_and_label(
        stage,
        info["sources"],
        info["destinations"],
        env_ids,
        mapping,
        positions=positions,
        up_axis=info.get("up_axis", "Z"),
        simplify_meshes=info.get("simplify_meshes", True),
    )
    sim = NewtonSim(builder, solver_cfg, collision_cfg, physics_dt, num_substeps, device)

    viewer = newton.viewer.ViewerNull() if args.headless else newton.viewer.ViewerGL()
    viewer.set_model(sim.model)
    viewer.set_world_offsets((0.0, 0.0, 0.0))
    recorder = None
    if args.record:
        from newton._src.viewer.viewer_file import ViewerFile
        recorder = ViewerFile(args.record, max_history_size=200)
        recorder.set_model(sim.model)

    jc_starts = sim.model.joint_coord_world_start.numpy()
    jd_starts = sim.model.joint_dof_world_start.numpy()
    jc_per = int(jc_starts[1]) - int(jc_starts[0])
    jd_per = int(jd_starts[1]) - int(jd_starts[0])

    mdp_env = mdp.AnymalMDP(
        model=sim.model,
        state=sim.state,
        control=sim.control,
        env_origins=positions,
        num_envs=num_envs,
        jc_per=jc_per,
        jd_per=jd_per,
        physics_dt=physics_dt,
        episode_length_s=episode_length_s,
        decimation=decimation,
        device=device,
    )

    # AnymalMDP modifies model arrays (armature, stiffness, damping, etc.)
    # Notify the solver so it syncs these into the MuJoCo Warp model.
    sim.notify_model_changed()

    policy_path = args.policy or os.path.join(env_dir, "policy.pt")
    print(f"[INFO] Loading JIT policy: {policy_path}")
    policy = torch.jit.load(policy_path, map_location=device).eval()

    obs = mdp_env.get_observations()
    timestep = 0
    prev = {}
    try:
        while args.headless or viewer.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                mdp_env.set_actions(actions)
                for _ in range(decimation):
                    mdp_env.apply_lstm_torques()
                    sim.step()
                obs, terminated, truncated = mdp_env.forward()

                # NaN detection
                jq = wp.to_torch(sim.state.joint_q).reshape(num_envs, jc_per)
                nan_mask = torch.isnan(jq).any(dim=1)
                if nan_mask.any():
                    nan_envs = nan_mask.nonzero(as_tuple=False).squeeze(-1)
                    jqd = wp.to_torch(sim.state.joint_qd).reshape(num_envs, jd_per)
                    jf = wp.to_torch(sim.control.joint_f).reshape(num_envs, jd_per)
                    mjw = sim.solver.mjw_data
                    for eid in nan_envs[:3].tolist():
                        print(f"\n{'='*80}")
                        print(f"[NaN] timestep={timestep} env={eid}")
                        print(f"  jq:   {jq[eid].tolist()}")
                        print(f"  jqd:  {jqd[eid].tolist()}")
                        print(f"  jf:   {jf[eid].tolist()}")
                        if mjw is not None:
                            for name in ["qpos", "qvel", "qacc", "qfrc_applied", "qfrc_constraint", "qfrc_bias"]:
                                arr = getattr(mjw, name, None)
                                if arr is not None:
                                    val = wp.to_torch(arr)[eid]
                                    print(f"  mjw.{name}: {val.tolist()}")
                            for name in ["ncon", "nefc"]:
                                arr = getattr(mjw, name, None)
                                if arr is not None:
                                    print(f"  mjw.{name}: {wp.to_torch(arr)[eid].item()}")
                        if prev:
                            print(f"\n[Pre-NaN] env={eid}")
                            print(f"  prev_jq:   {prev['jq'][eid].tolist()}")
                            print(f"  prev_jqd:  {prev['jqd'][eid].tolist()}")
                            print(f"  prev_jf:   {prev['jf'][eid].tolist()}")
                            print(f"  |prev_jq|_max:  {prev['jq'][eid].abs().max().item():.6f}")
                            print(f"  |prev_jqd|_max: {prev['jqd'][eid].abs().max().item():.6f}")
                            print(f"  |prev_jf|_max:  {prev['jf'][eid].abs().max().item():.6f}")
                        print(f"{'='*80}\n")
                    if recorder:
                        recorder.save()
                        print(f"[INFO] Recording saved to {args.record}")
                    break

                prev = {
                    "jq": jq.clone(),
                    "jqd": wp.to_torch(sim.state.joint_qd).reshape(num_envs, jd_per).clone(),
                    "jf": wp.to_torch(sim.control.joint_f).reshape(num_envs, jd_per).clone(),
                }

                reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
                if len(reset_ids) > 0:
                    mdp_env.reset(reset_ids)
                    policy.reset()

                if timestep % 100 == 0:
                    print(f"Timestep: {timestep}")

            sim.forward()
            viewer.begin_frame(0.0)
            viewer.log_state(sim.state)
            viewer.end_frame()
            if recorder:
                recorder.begin_frame(timestep * physics_dt * decimation)
                recorder.log_state(sim.state)
                recorder.end_frame()
            timestep += 1
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
