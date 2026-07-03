# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-joint recovery error of a fit against a synthetic dataset's known gains.

The synthetic-recovery acceptance gate, machine-enforced: compares a run's best
evaluated candidate and latest checkpointed mean against the ``kp_used/kd_used``
the dataset was generated with, verifies joint-order provenance, writes
``recovery_result.json``, and exits nonzero when the MAPE threshold is missed.
Pure CPU — no isaaclab.

    python scripts/sysid/recovery_report.py --data <synth>/chirp_data.pt --run <log_dir> [--max_mape 5.0]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_contract import load_dataset, validate_contract  # noqa: E402


def _table(label: str, params: torch.Tensor, kp_true, kd_true, names: list[str]) -> dict:
    n = len(names)
    kp_fit, kd_fit = params[:n], params[n:]
    print(f"\n== {label}")
    print(f"{'joint':<14}{'kp true':>10}{'kp fit':>12}{'err%':>8}{'kd true':>10}{'kd fit':>12}{'err%':>8}")
    joints = {}
    for i, name in enumerate(names):
        kp_err = 100.0 * (kp_fit[i] - kp_true[i]) / kp_true[i]
        kd_err = 100.0 * (kd_fit[i] - kd_true[i]) / kd_true[i]
        joints[name] = {"kp_err_pct": kp_err.item(), "kd_err_pct": kd_err.item()}
        print(
            f"{name:<14}{kp_true[i]:>10.1f}{kp_fit[i]:>12.1f}{kp_err:>+8.2f}"
            f"{kd_true[i]:>10.1f}{kd_fit[i]:>12.1f}{kd_err:>+8.2f}"
        )
    kp_mape = float((100.0 * (kp_fit - kp_true).abs() / kp_true).mean())
    kd_mape = float((100.0 * (kd_fit - kd_true).abs() / kd_true).mean())
    print(f"{'MAPE':<14}{'':>10}{'':>12}{kp_mape:>8.3f}{'':>10}{'':>12}{kd_mape:>8.3f}")
    return {"joints": joints, "kp_mape_pct": kp_mape, "kd_mape_pct": kd_mape}


def _verify_order(artifact_path: str, blob, names: list[str]) -> torch.Tensor:
    if isinstance(blob, dict):
        stored = blob.get("joint_order")
        if stored is None:
            raise ValueError(f"{artifact_path} carries no joint_order provenance — regenerate the run.")
        if list(stored) != list(names):
            raise ValueError(f"{artifact_path} joint_order {list(stored)} != dataset active joints {list(names)}")
        return blob["sim_params"].float()
    raise ValueError(f"{artifact_path} is a legacy tensor artifact without provenance — regenerate the run.")


def _verify_provenance(artifact_path: str, blob: dict, bind_to: dict | None = None) -> None:
    """A gate artifact must carry full, typed run provenance — nulls fail.

    ``bind_to`` cross-binds a second artifact (the mean) to the best
    candidate's run: seed and USD digest must MATCH, not merely exist — a
    foreign mean beside a valid best is a FAIL.
    """
    if blob.get("seed") is None:
        raise ValueError(f"{artifact_path} carries no seed provenance — regenerate the run.")
    meta = blob.get("run_metadata")
    if not isinstance(meta, dict) or not meta:
        raise ValueError(f"{artifact_path} carries no run_metadata provenance — regenerate the run.")
    digest = meta.get("usd_digest")
    if not isinstance(digest, str) or not digest:
        raise ValueError(f"{artifact_path} run_metadata.usd_digest missing/empty — regenerate the run.")
    if not isinstance(meta.get("use_cuda_graph"), bool):
        raise ValueError(f"{artifact_path} run_metadata.use_cuda_graph missing or non-bool — regenerate the run.")
    if bind_to is not None:
        if blob["seed"] != bind_to.get("seed"):
            raise ValueError(f"{artifact_path} seed {blob['seed']} != best candidate seed {bind_to.get('seed')}.")
        bind_digest = (bind_to.get("run_metadata") or {}).get("usd_digest")
        if digest != bind_digest:
            raise ValueError(f"{artifact_path} usd_digest {digest} != best candidate digest {bind_digest}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, required=True, help="Synthetic dataset with known kp_used/kd_used.")
    parser.add_argument("--run", type=str, required=True, help="Fit log dir (best_candidate.pt, mean_*.pt).")
    parser.add_argument("--max_mape", type=float, default=5.0, help="Max kp/kd MAPE (%%) for a PASS.")
    args = parser.parse_args()

    ds = validate_contract(load_dataset(args.data))
    names = ds.active_joint_names
    cols = [ds.joint_names.index(n) for n in names]
    kp_true, kd_true = ds.kp_used[cols], ds.kd_used[cols]

    result: dict = {"data": args.data, "run": args.run, "max_mape_pct": args.max_mape}

    best_path = os.path.join(args.run, "best_candidate.pt")
    if not os.path.exists(best_path):
        print(f"[FAIL] no best_candidate.pt in {args.run}")
        return 1
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    print(f"run: {args.run}")
    print(f"best evaluated candidate: score {best['score']:.6e} rad² (generation {best['iteration']})")
    _verify_provenance(best_path, best)
    result["best_score_rad2"] = best["score"]
    result["seed"] = best["seed"]
    result["run_metadata"] = best["run_metadata"]
    result["best"] = _table("best evaluated candidate", _verify_order(best_path, best, names), kp_true, kd_true, names)

    # The rerolled mean is the fit's primary deliverable — its absence or a
    # threshold miss fails the gate just like the best candidate.
    means = sorted(glob.glob(os.path.join(args.run, "mean_*.pt")))
    if not means:
        print(f"[FAIL] no mean_*.pt checkpoint in {args.run}")
        return 1
    mean_blob = torch.load(means[-1], map_location="cpu", weights_only=False)
    _verify_provenance(means[-1], mean_blob, bind_to=best)
    mean_params = _verify_order(means[-1], mean_blob, names)
    result["mean"] = _table(f"CMA mean ({os.path.basename(means[-1])})", mean_params, kp_true, kd_true, names)

    result["pass"] = all(
        gate["kp_mape_pct"] <= args.max_mape and gate["kd_mape_pct"] <= args.max_mape
        for gate in (result["best"], result["mean"])
    )
    gate = result["best"]
    out = os.path.join(args.run, "recovery_result.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"\nverdict: {'PASS' if result['pass'] else 'FAIL'} "
        f"(best kp/kd MAPE {gate['kp_mape_pct']:.3f}%/{gate['kd_mape_pct']:.3f}% vs {args.max_mape}%) → {out}"
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
