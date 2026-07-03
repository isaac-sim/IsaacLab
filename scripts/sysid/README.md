# FR3 implicit-actuator sysid (stiffness/damping fitting)

Fits per-joint `{stiffness, damping}` of the `ImplicitActuatorCfg` for the
Franka FR3 by replaying chirp data recorded with the `isaac_ros_sysid` GUI,
on the Newton/mjwarp backend. Ported from the agile G1 sysid stack
(`scripts/sys_id/fit.py` + `CMAESOptimizer`), with these deltas:

- fitted params are the solver-side implicit PD gains (2N), not
  armature/friction/delay (3N+1);
- fixed-base 7-DOF arm, no biped scaffolding;
- gravity disabled by default (`sysid.zero_gravity`) — the FR3 firmware
  gravity-compensates on top of its impedance loop, so the zero-g sim
  approximates the compensated plant and fitted gains don't absorb
  gravity-holding torque;
- 1 kHz physics with the command shaping reproduced in the replay:
  `--shaping auto` (default) resolves the shaper PROVENANCE stamped in the
  dataset and hard-fails when it is absent or incomplete; `franka_fr3`
  reconstructs the driver's clamp→EMA→Ruckig pipeline offline
  (`fr3_target_shaping.py`) and feeds the resulting 1 kHz impedance targets to
  the sim, `none` is a plain ZOH of the raw commands via env decimation.

## Candidate real controllers (stack pin is a GM decision — do not hard-code)

Two stacks are on the table; gains and shaping are always read from run
metadata, never assumed:

**Stack A — isaac_ros_robots/franka_fr3** (`shaper_type: franka_fr3_ema_ruckig`):
`FrankaFr3SystemInterface` (ros2_control, CM at **200 Hz**): position commands
are consumed by a **1 kHz** internal FCI thread as
`latest 200 Hz command (ZOH) → clamp to limits−0.05 rad → EMA (α=0.02/tick,
τ≈49 ms, fc≈3.25 Hz) → Ruckig OTG (vel/acc/jerk × relative_dynamics, default
0.3 → vel caps 0.65 rad/s j1-4 / 0.78 rad/s j5-7) → libfranka
kJointImpedance` with `setJointImpedance` default
`[1500,1500,1500,1250,1250,1000,1000]` (internal damping not exposed; gravity
comp internal — hence the zero-g sim). This shaping is deterministic and
independent of the measured state, so `fit.py` reconstructs the impedance
targets from `des_dof_pos` (APPROXIMATE — see the burn-in section) — 2 Hz chirp
segments whose raw commands exceed the Ruckig caps (≈2.9× vel, ≈8.4× accel at
scale 0.15) remain usable *only* through this reconstruction.
`relative_dynamics` and the configured
`joint_impedance` MUST match the deployed hardware parameters (read them from
the run metadata; never hard-code — the 600/30 set floating around is the
mujoco sim-backend PID from `mujoco_pid.yaml`, not the real arm).

**Stack B — dex/robot-control MR!2 `FrankaDriver`** (`shaper_type: none`):
libfranka torque mode with a 1 kHz host PD (kp `[600×4,250,150,50]`,
kd `[30×4,10,10,5]`), SHM latest-value targets, no position shaping —
but also no ROS command path (a Float64MultiArray→SHM bridge would be needed).

The mujoco sim backend (`mujoco_pid.yaml`) uses the 600/30 PID and no shaper —
sim datasets stamp `shaper_type: none`.

Note for RL deployment: any policy commanding the stack-A driver faces the
same shaper. A GPU-parallel shaper action term for training envs is future
work; the exact CPU Ruckig reconstruction here is fit-only.

## Pipeline

1. **Asset (one-time).** `python scripts/sysid/prepare_fr3_asset.py` strips the
   unresolvable meshes out of `fr3.urdf`, then run the printed
   `convert_urdf.py` command (needs Kit) to produce `fr3.usd` next to it.
2. **Collect data** with the isaac_ros_sysid GUI (`robot:=franka_fr3`) — see
   `config/robots/franka_fr3.yaml` in that repo. Output: `chirp_data.pt`.
3. **Fit:**

   ```bash
   ./isaaclab.sh -p scripts/sysid/fit.py \
       --task Isaac-Sysid-Franka-FR3-v0 \
       --num_envs 256 \
       --data <run_dir>/chirp_data.pt
   ```

   Results land in `logs/sysid/franka_fr3/<stamp>/`: `fitted_parameters.txt`,
   `mean_*.pt` (physical-space CMA mean), `best_trajectory.pt`, tensorboard
   logs, and `fit_signals.png` (overlay plot via isaac_ros_sysid's
   `plot_chirp.py` when available).

Options: `--warmstart_from_data` seeds the CMA mean from the dataset's
`kp_used/kd_used` metadata (never written to sim otherwise);
`physics=physx` hydra token switches backend for A/B checks;
`--controller_update_rate` overrides the ZOH rate.

## Dataset contract (fail-closed; see `data_contract.py` + its tests)

REQUIRED: `time (T,)`, `des_dof_pos (T,N)`, `dof_pos (T,N)`, `dof_vel (T,N)`,
`dof_tau_est (T,N)` (diagnostics only — never in the loss), `joint_names [N]`,
`active_joint_names [K]`, `sample_rate`, `kp_used (N,)`, `kd_used (N,)`, and
the freshness pair `state_fresh (T,)` + `state_stamps (T,)` (float64,
self-consistent: `fresh[i] == stamps[i] > stamps[i-1]`). Stale rows reject by
default; `--allow_stale_fraction` (debug-only, hard-capped at 0.20) masks them
from the loss. Shaper provenance: top-level `shaper_type` (+
`shaper_ema_alpha`/`shaper_relative_dynamics`/`shaper_rate_hz` when
franka_fr3); `gains_provenance.command_shaping` is validated and cross-checked
— a conflict is a hard failure. Completion: `safety_controller.aborted` or a
short run vs `intended_duration_s` rejects unless `--allow_truncated`
(diagnostics only). Clamped runs always reject. Legacy escapes
(`--allow_missing_freshness`) are provenance-stamped into every artifact.

## Requirements

`pip install cmaes ruckig` on top of the IsaacLab env (Newton backend needs the
`newton` and `mujoco_warp` packages, present in `env_isaaclab`).

## Shaper reconstruction is APPROXIMATE (real data)

The hardware shaper's internal state at recording t=0 is unobserved (it
persisted through homing + dwell). The reconstruction seeds it settled at
`des_dof_pos[0]`, and the loss masks a burn-in window measured from an
ENVELOPE of plausible initial states (measured pose plus ± perturbations
beyond the observed mismatch), scaled by a 1.5× plant-settling factor, floor
0.25 s. This is a conservative diagnostic HEURISTIC, not a mathematical bound
— position-seeded streams do not bracket unknown EMA/Ruckig velocity/
acceleration state or candidate-dependent plant memory. State is preserved
end-to-end (the mask affects only the loss, never the replay), and no real-
data acceptance claim derives from it: real fitting stays gated on
driver-exported applied targets and timestamp alignment.
`fitted_parameters.txt` labels such fits `APPROXIMATE reconstruction`; exact
replay arrives when the driver exports applied targets
(`des_dof_pos_applied`). Sim datasets (`shaper_type: none`) are unaffected
(burn-in 0).

## Executable acceptance gates

| Gate | Command |
| --- | --- |
| Env + per-env gain writes (CUDA graph on/off) | `./isaaclab.sh -p scripts/sysid/smoke_test.py [--no_cuda_graph]` |
| Contract / shaper / ordering rules | `pytest scripts/sysid/tests/` |
| Synthetic known-gain recovery | `./isaaclab.sh -p scripts/sysid/make_synthetic_dataset.py --out ...` then `fit.py` then `recovery_report.py --max_mape 5` (JSON verdict + failing exit) — per-joint error vs stamped `kp_used/kd_used` |
| Held-out evaluation vs baselines | `./isaaclab.sh -p scripts/sysid/fit.py --data <heldout>.pt --eval_params <run>/best_candidate.pt` — needs >=3 envs; verdict = beats all baselines AND saturation <= ceiling; `eval_result.json`, nonzero exit on FAIL |
| USD audit (all joints/drives/limits/masses) | `python scripts/sysid/audit_fr3_usd.py` — per-named-joint drives/limits, fixed base, positive masses, MjcActuators; prints the USD content digest (recorded in every fit's run_metadata); mutation-tested; nonzero exit on failure |
| Saturation audit | logged per generation (`0_Episode/saturation_*`) and in the summary |
| Final-mean reroll | automatic — the summary's mean score is from an actual rollout |

## Excitation protocol (identifiability)

Excite **one joint per run** (GUI Independent mode); simultaneous identical
zero-phase chirps on all joints make the input rank-1 and the fit meaningless.
Multiple poses / amplitudes per joint improve conditioning; hold the other
joints with the recorded rig gains (`fit.py` writes `kp_used/kd_used` to held
joints automatically).

## Validation before trusting a fit (Newton backend was never used for sysid before)

- self-consistency: replay a **sim-generated** dataset with known gains and
  check CMA-ES recovers them (synthetic recovery gate);
- per-env write check: two envs with different stiffness must diverge under
  the same command (run with CUDA graphs on and off);
- USD sanity after conversion: position drives present on all 7 joints, fixed
  root, URDF inertias/effort limits (87/87/87/87/12/12/12 Nm) intact;
- torque-saturation audit: fraction of steps at the 12 Nm wrist limit — a
  saturated candidate makes high stiffness unidentifiable;
- the final CMA mean is rerolled AUTOMATICALLY (the summary's mean score is an
  evaluated rollout, alongside the best evaluated candidate);
- held-out trajectories (different frequencies/amplitudes/poses) must beat
  both the fixed default gains and the recorded-gain baseline;
- A/B `physics=physx` vs `newton_mjwarp` on the same dataset.

Deferred (v2): action-delay nuisance parameter (state/command timestamping is
collection-side work first), per-joint loss weighting, multi-seed CMA runs,
saturation-aware candidate rejection.
