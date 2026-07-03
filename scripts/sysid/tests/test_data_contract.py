# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the sysid dataset contract (pure CPU, no isaaclab)."""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_contract import (  # noqa: E402
    CANONICAL_JOINT_ORDER,
    ContractError,
    ShaperSpec,
    build_loss_mask,
    canonical_indices,
    compute_burn_in_s,
    measure_convergence_burn_in,
    normalize_joint_name,
    resolve_shaper,
    validate_contract,
)


def make_data(T: int = 100, rate: float = 200.0, names=None) -> dict:
    names = names or list(CANONICAL_JOINT_ORDER)
    n = len(names)
    return {
        "time": torch.arange(T, dtype=torch.float32) / rate,
        "des_dof_pos": torch.zeros(T, n),
        "dof_pos": torch.zeros(T, n),
        "dof_vel": torch.zeros(T, n),
        "dof_tau_est": torch.zeros(T, n),
        "joint_names": names,
        "active_joint_names": list(names),
        "kp_used": torch.full((n,), 600.0),
        "kd_used": torch.full((n,), 30.0),
        "sample_rate": rate,
        "state_fresh": torch.ones(T, dtype=torch.uint8),
        "state_stamps": torch.arange(T, dtype=torch.float64) / rate,
        "mode": "synthetic",
    }


class TestValidateContract:
    def test_valid_passes(self):
        ds = validate_contract(make_data())
        assert ds.dt == pytest.approx(1.0 / 200.0)
        assert ds.joint_names == CANONICAL_JOINT_ORDER

    def test_missing_key_fails(self):
        data = make_data()
        del data["sample_rate"]
        with pytest.raises(ContractError, match="sample_rate"):
            validate_contract(data)

    def test_panda_names_normalized(self):
        names = [f"panda_joint{i}" for i in range(1, 8)]
        ds = validate_contract(make_data(names=names))
        assert ds.joint_names == CANONICAL_JOINT_ORDER

    def test_duplicate_names_fail(self):
        names = ["fr3_joint1"] * 7
        with pytest.raises(ContractError, match="duplicate"):
            validate_contract(make_data(names=names))

    def test_mixed_duplicate_after_normalization_fails(self):
        names = ["panda_joint1", "fr3_joint1"] + [f"fr3_joint{i}" for i in range(2, 7)]
        with pytest.raises(ContractError, match="duplicate"):
            validate_contract(make_data(names=names))

    def test_time_jitter_fails(self):
        data = make_data()
        data["time"][50] += 0.002  # 40% of a 5 ms step
        with pytest.raises(ContractError, match="uniform|increasing"):
            validate_contract(data)

    def test_nonfinite_fails(self):
        data = make_data()
        data["dof_pos"][3, 2] = float("nan")
        with pytest.raises(ContractError, match="non-finite"):
            validate_contract(data)

    def test_shape_mismatch_fails(self):
        data = make_data()
        data["dof_pos"] = data["dof_pos"][:-1]
        with pytest.raises(ContractError, match="shape"):
            validate_contract(data)

    def test_endpoint_inclusive_linspace_passes(self):
        # The collector's chirp clock: linspace(0, 20, 4000) drifts 1/T from
        # 1/sample_rate — inside the 1% tolerance.
        data = make_data(T=4000)
        data["time"] = torch.linspace(0.0, 20.0, 4000)
        validate_contract(data)


class TestResolveShaper:
    def test_missing_provenance_fails(self):
        with pytest.raises(ContractError, match="shaper provenance"):
            resolve_shaper(make_data())

    def test_unknown_type_fails(self):
        data = make_data()
        data["shaper_type"] = "mystery_filter"
        with pytest.raises(ContractError, match="unknown shaper_type"):
            resolve_shaper(data)

    def test_none_passes(self):
        data = make_data()
        data["shaper_type"] = "none"
        assert resolve_shaper(data).type == "none"

    def test_auto_franka_requires_every_param(self):
        data = make_data()
        data["shaper_type"] = "franka_fr3_ema_ruckig"
        data["shaper_ema_alpha"] = 0.02
        with pytest.raises(ContractError, match="shaper_relative_dynamics"):
            resolve_shaper(data)

    def test_auto_franka_with_all_params(self):
        data = make_data()
        data["shaper_type"] = "franka_fr3_ema_ruckig"
        data["shaper_ema_alpha"] = 0.02
        data["shaper_relative_dynamics"] = 0.3
        data["shaper_rate_hz"] = 1000.0
        spec = resolve_shaper(data)
        assert spec.type == "franka_fr3" and spec.rate_hz == 1000.0 and spec.approximate

    def test_cli_force_allows_defaults(self):
        spec = resolve_shaper(make_data(), cli_shaping="franka_fr3")
        assert spec.ema_alpha == 0.02 and spec.relative_dynamics == 0.3

    def test_gains_provenance_fallback(self):
        # The landed collector nests command_shaping in gains_provenance JSON.
        data = make_data()
        data["gains_provenance"] = json.dumps({"command_shaping": "none"})
        assert resolve_shaper(data).type == "none"

    def test_safety_clamp_rejected(self):
        data = make_data()
        data["safety_controller"] = {"active": True, "clamped": True}
        with pytest.raises(ContractError, match="safety controller"):
            validate_contract(data)


class TestOrderingAndBurnIn:
    def test_canonical_indices_permutation(self):
        names = ["fr3_joint3", "fr3_joint1", "fr3_joint7", "fr3_joint2", "fr3_joint5", "fr3_joint4", "fr3_joint6"]
        assert canonical_indices(names) == [2, 0, 6, 1, 4, 3, 5]

    def test_canonical_indices_unknown_fails(self):
        with pytest.raises(ContractError, match="canonical"):
            canonical_indices(["fr3_joint1", "mystery_joint"])

    def test_normalize(self):
        assert normalize_joint_name("panda_joint4") == "fr3_joint4"
        assert normalize_joint_name("fr3_joint4") == "fr3_joint4"
        assert normalize_joint_name("left_knee_joint") == "left_knee_joint"

    def test_burn_in_zero_for_no_shaper(self):
        assert compute_burn_in_s(ShaperSpec("none"), torch.zeros(7), torch.zeros(7)) == 0.0

    def test_burn_in_floor(self):
        spec = ShaperSpec("franka_fr3", 0.02, 0.3, 1000.0)
        q0 = torch.zeros(7)
        des0 = torch.full((7,), 1e-5)  # below target error -> floor
        assert compute_burn_in_s(spec, q0, des0) == pytest.approx(0.25)

    def test_burn_in_scales_with_mismatch(self):
        spec = ShaperSpec("franka_fr3", 0.02, 0.3, 1000.0)
        q0 = torch.zeros(7)
        small = compute_burn_in_s(spec, q0, torch.full((7,), 0.01))
        large = compute_burn_in_s(spec, q0, torch.full((7,), 0.2))
        assert large > small >= 0.25
        # 0.2 rad -> 0.05 * ln(0.2/1e-4) ~ 0.38 s
        assert large == pytest.approx(0.05 * np.log(0.2 / 1e-4), rel=1e-6)

    def test_burn_in_override(self):
        spec = ShaperSpec("franka_fr3", 0.02, 0.3, 1000.0)
        assert compute_burn_in_s(spec, torch.zeros(7), torch.ones(7), override=0.1) == 0.1

    def test_burn_in_override_negative_fails(self):
        spec = ShaperSpec("franka_fr3", 0.02, 0.3, 1000.0)
        with pytest.raises(ContractError, match="out of range"):
            compute_burn_in_s(spec, torch.zeros(7), torch.ones(7), override=-1.0)

    def test_measured_burn_in_covers_slow_convergence(self):
        # Deviation decays slowly (Ruckig-limited catch-up): the measured mask
        # must cover it fully, where the EMA-only formula would under-trim.
        ticks = 1500
        dev = 0.5 * np.exp(-np.arange(ticks) / 100.0)  # >1e-4 until tick ~852
        primary = np.zeros((300, 5, 2))
        alternate = primary + dev.reshape(300, 5, 1)
        burn = measure_convergence_burn_in(primary, alternate, tick_dt=0.001)
        last_exceed = int(np.nonzero(dev > 1e-4)[0][-1])
        assert burn == pytest.approx(1.5 * (last_exceed + 1) * 0.001)  # settle_factor 1.5
        assert burn > 1.2  # covers the 1.205 s plant-memory counterexample for m0=0.5

    def test_measured_burn_in_floor(self):
        primary = np.zeros((10, 5, 2))
        assert measure_convergence_burn_in(primary, primary.copy(), tick_dt=0.001) == 0.25


class TestNumericValidation:
    def test_nan_sample_rate_fails(self):
        data = make_data()
        data["sample_rate"] = float("nan")
        with pytest.raises(ContractError, match="finite"):
            validate_contract(data)

    def test_lone_kp_fails(self):
        data = make_data()
        del data["kd_used"]
        with pytest.raises(ContractError, match="kd_used"):
            validate_contract(data)

    def test_nan_gains_fail(self):
        data = make_data()
        data["kp_used"][3] = float("nan")
        with pytest.raises(ContractError, match="non-finite"):
            validate_contract(data)

    def test_nonpositive_kp_fails(self):
        data = make_data()
        data["kp_used"][0] = 0.0
        with pytest.raises(ContractError, match="kp_used must be > 0"):
            validate_contract(data)

    def test_missing_dof_vel_fails(self):
        data = make_data()
        del data["dof_vel"]
        with pytest.raises(ContractError, match="dof_vel"):
            validate_contract(data)

    def test_missing_active_names_fails(self):
        data = make_data()
        del data["active_joint_names"]
        with pytest.raises(ContractError, match="active_joint_names"):
            validate_contract(data)

    def test_duplicate_active_names_fail(self):
        data = make_data()
        data["active_joint_names"] = ["fr3_joint1", "panda_joint1"]
        with pytest.raises(ContractError, match="duplicate active"):
            validate_contract(data)

    def test_shaper_alpha_out_of_range_fails(self):
        data = make_data()
        data["shaper_type"] = "franka_fr3_ema_ruckig"
        data["shaper_ema_alpha"] = 2.0
        data["shaper_relative_dynamics"] = 0.3
        data["shaper_rate_hz"] = 1000.0
        with pytest.raises(ContractError, match="shaper_ema_alpha"):
            resolve_shaper(data)

    def test_negative_relative_dynamics_fails(self):
        data = make_data()
        data["shaper_type"] = "franka_fr3_ema_ruckig"
        data["shaper_ema_alpha"] = 0.02
        data["shaper_relative_dynamics"] = -0.3
        data["shaper_rate_hz"] = 1000.0
        with pytest.raises(ContractError, match="shaper_relative_dynamics"):
            resolve_shaper(data)

    def test_nan_shaper_rate_fails(self):
        data = make_data()
        data["shaper_type"] = "franka_fr3_ema_ruckig"
        data["shaper_ema_alpha"] = 0.02
        data["shaper_relative_dynamics"] = 0.3
        data["shaper_rate_hz"] = float("nan")
        with pytest.raises(ContractError, match="shaper_rate_hz"):
            resolve_shaper(data)

    def test_malformed_safety_json_fails(self):
        data = make_data()
        data["safety_controller"] = "{not json"
        with pytest.raises(ContractError, match="safety_controller"):
            validate_contract(data)

    def test_clamped_rejected_even_with_applied_key(self):
        # The des_dof_pos_applied path is not wired — key presence is NOT consumption.
        data = make_data()
        data["safety_controller"] = {"active": True, "clamped": True}
        data["des_dof_pos_applied"] = torch.zeros(1, 1)
        with pytest.raises(ContractError, match="clamped"):
            validate_contract(data)

    def test_malformed_gains_provenance_fails(self):
        data = make_data()
        data["gains_provenance"] = "{broken"
        with pytest.raises(ContractError, match="gains_provenance"):
            resolve_shaper(data)


def make_stale_data(stale_rows: slice) -> dict:
    """Dataset with a consistent stale block: repeated stamps + zeroed mask."""
    data = make_data()
    data["state_stamps"] = data["state_stamps"].clone()
    data["state_fresh"] = data["state_fresh"].clone()
    start, stop = stale_rows.start, stale_rows.stop
    for i in range(start, stop):
        data["state_stamps"][i] = data["state_stamps"][start - 1]
        data["state_fresh"][i] = 0
    return data


class TestFreshness:
    def test_stale_rows_rejected_by_default(self):
        with pytest.raises(ContractError, match="stale"):
            validate_contract(make_stale_data(slice(10, 28)))  # 18%

    def test_stale_rows_accepted_with_override(self):
        ds = validate_contract(make_stale_data(slice(10, 28)), allow_stale_fraction=0.2)
        assert ds.stale_fraction == pytest.approx(0.18)
        assert ds.state_fresh is not None

    def test_all_fresh_passes(self):
        ds = validate_contract(make_data())
        assert ds.stale_fraction == 0.0 and ds.state_fresh is not None

    def test_invalid_state_fresh_values_fail(self):
        data = make_data()
        data["state_fresh"] = torch.full((100,), 2, dtype=torch.uint8)
        with pytest.raises(ContractError, match="0/1"):
            validate_contract(data)

    def test_nonmonotonic_state_stamps_fail(self):
        data = make_data()
        data["state_stamps"][50] = data["state_stamps"][49] - 0.01
        with pytest.raises(ContractError, match="monotonically"):
            validate_contract(data)


class TestFreshnessBypasses:
    """Round-3b bypass closures: absence, consistency, override range, mask regression."""

    def test_absent_freshness_fails_by_default(self):
        data = make_data()
        del data["state_fresh"], data["state_stamps"]
        with pytest.raises(ContractError, match="state_fresh"):
            validate_contract(data)

    def test_absent_freshness_allowed_with_flag(self):
        data = make_data()
        del data["state_fresh"], data["state_stamps"]
        ds = validate_contract(data, allow_missing_freshness=True)
        assert ds.state_fresh is None

    def test_first_row_stale_fails(self):
        data = make_data()
        data["state_fresh"][0] = 0
        with pytest.raises(ContractError, match=r"state_fresh\[0\]"):
            validate_contract(data, allow_stale_fraction=0.2)

    def test_mask_stamp_inconsistency_fails(self):
        # All-ones mask + a repeated stamp must not pass.
        data = make_data()
        data["state_stamps"][50] = data["state_stamps"][49]
        with pytest.raises(ContractError, match="inconsistent"):
            validate_contract(data, allow_stale_fraction=0.2)

    def test_consistent_stale_rows_respected(self):
        data = make_data()
        data["state_stamps"][50] = data["state_stamps"][49]
        data["state_fresh"][50] = 0
        ds = validate_contract(data, allow_stale_fraction=0.05)
        assert ds.stale_fraction == pytest.approx(0.01)

    def test_nan_allow_stale_fraction_fails(self):
        with pytest.raises(ContractError, match="allow_stale_fraction"):
            validate_contract(make_data(), allow_stale_fraction=float("nan"))

    def test_out_of_range_allow_stale_fraction_fails(self):
        with pytest.raises(ContractError, match="allow_stale_fraction"):
            validate_contract(make_data(), allow_stale_fraction=1.5)


class TestLossMask:
    def test_stale_rows_excluded_from_objective(self):
        fresh = torch.ones(100)
        fresh[20:40] = 0
        mask = build_loss_mask(100, 0.005, burn_in_s=0.0, state_fresh=fresh)
        assert not mask[20:40].any() and mask[40:].all()
        assert int(mask.sum()) == 80

    def test_burn_in_and_stale_combine(self):
        fresh = torch.ones(100)
        fresh[60] = 0
        mask = build_loss_mask(100, 0.005, burn_in_s=0.25, state_fresh=fresh)
        assert not mask[:50].any()  # burn-in covers t < 0.25 s
        assert not mask[60]
        assert int(mask.sum()) == 49

    def test_all_masked_raises(self):
        with pytest.raises(ContractError, match="zero scored samples"):
            build_loss_mask(100, 0.005, burn_in_s=10.0, state_fresh=None)

    def test_all_stale_raises(self):
        with pytest.raises(ContractError, match="zero scored samples"):
            build_loss_mask(100, 0.005, burn_in_s=0.0, state_fresh=torch.zeros(100))


class TestRound4Hardening:
    def test_epoch_stamps_keep_float64_precision(self):
        # 1.75e9 s epoch stamps with 5 ms increments collapse in fp32 —
        # a valid rosbag-style dataset must PASS, not raise 99 inconsistencies.
        data = make_data()
        data["state_stamps"] = 1.75e9 + torch.arange(100, dtype=torch.float64) / 200.0
        ds = validate_contract(data)
        assert ds.stale_fraction == 0.0

    def test_missing_tau_fails(self):
        data = make_data()
        del data["dof_tau_est"]
        with pytest.raises(ContractError, match="dof_tau_est"):
            validate_contract(data)

    def test_nonfinite_tau_fails(self):
        data = make_data()
        data["dof_tau_est"][5, 3] = float("inf")
        with pytest.raises(ContractError, match="dof_tau_est"):
            validate_contract(data)

    def test_provenance_conflict_fails(self):
        data = make_data()
        data["shaper_type"] = "none"
        data["gains_provenance"] = json.dumps({"command_shaping": "franka_fr3"})
        with pytest.raises(ContractError, match="conflict"):
            resolve_shaper(data)

    def test_provenance_agreement_passes(self):
        data = make_data()
        data["shaper_type"] = "none"
        data["gains_provenance"] = json.dumps({"command_shaping": "zoh"})
        assert resolve_shaper(data).type == "none"

    def test_malformed_nested_provenance_fails_even_with_top_level(self):
        data = make_data()
        data["shaper_type"] = "none"
        data["gains_provenance"] = "{broken"
        with pytest.raises(ContractError, match="gains_provenance"):
            resolve_shaper(data)

    def test_aborted_run_rejected(self):
        data = make_data()
        data["safety_controller"] = {"active": True, "clamped": False, "aborted": True}
        with pytest.raises(ContractError, match="aborted"):
            validate_contract(data)

    def test_aborted_run_allowed_with_flag(self):
        data = make_data()
        data["safety_controller"] = {"active": True, "clamped": False, "aborted": True}
        validate_contract(data, allow_truncated=True)

    def test_truncated_duration_rejected(self):
        data = make_data()  # 100 rows @ 200 Hz = 0.5 s
        data["intended_duration_s"] = 20.0
        with pytest.raises(ContractError, match="truncated"):
            validate_contract(data)

    def test_complete_duration_passes(self):
        data = make_data()
        data["intended_duration_s"] = 0.5
        validate_contract(data)

    def test_stale_override_capped_at_pm_ceiling(self):
        with pytest.raises(ContractError, match="allow_stale_fraction"):
            validate_contract(make_data(), allow_stale_fraction=0.5)

    def test_envelope_takes_worst_alternate(self):
        primary = np.zeros((100, 5, 2))
        short = primary.copy()
        short[:10] += 1.0  # diverges 50 ticks
        long = primary.copy()
        long[:40] += 1.0  # diverges 200 ticks
        burn = measure_convergence_burn_in(primary, [short, long], tick_dt=0.001)
        assert burn == pytest.approx(1.5 * 200 * 0.001)

    def test_identical_streams_hit_floor_only(self):
        # des0 == dof0 counterexample: position-identical seeds alone give the
        # floor — the ± offset alternates in fit.py exist precisely for this.
        primary = np.zeros((100, 5, 2))
        assert measure_convergence_burn_in(primary, [primary.copy()], tick_dt=0.001) == 0.25

    def test_nonfinite_stream_fails(self):
        primary = np.zeros((10, 5, 2))
        bad = primary.copy()
        bad[3, 2, 1] = float("nan")
        with pytest.raises(ContractError, match="non-finite"):
            measure_convergence_burn_in(primary, [bad], tick_dt=0.001)


class TestRound5Hardening:
    def _collected(self) -> dict:
        data = make_data()
        data["mode"] = "independent"  # collector-produced
        data["safety_controller"] = {"active": True, "clamped": False, "aborted": False}
        data["intended_duration_s"] = 99.0 / 200.0  # exact span of 100 rows
        return data

    def test_collected_with_completion_passes(self):
        validate_contract(self._collected())

    def test_stripped_completion_fails_closed(self):
        data = self._collected()
        del data["safety_controller"], data["intended_duration_s"]
        with pytest.raises(ContractError, match="completion metadata"):
            validate_contract(data)

    def test_stripped_completion_allowed_as_diagnostics(self):
        data = self._collected()
        del data["safety_controller"], data["intended_duration_s"]
        validate_contract(data, allow_truncated=True)

    def test_one_missing_row_passes(self):
        data = self._collected()
        data["intended_duration_s"] = 100.0 / 200.0  # one command sample beyond span
        validate_contract(data)

    def test_eighty_missing_rows_fail(self):
        data = self._collected()
        data["intended_duration_s"] = (99.0 + 80.0) / 200.0
        with pytest.raises(ContractError, match="truncated"):
            validate_contract(data)

    def test_non_dict_gains_provenance_fails(self):
        data = make_data()
        data["shaper_type"] = "none"
        for bad in (json.dumps([]), json.dumps(3)):
            data["gains_provenance"] = bad
            with pytest.raises(ContractError, match="JSON object"):
                resolve_shaper(data)

    def test_burn_in_parameter_validation(self):
        primary = np.zeros((10, 5, 2))
        with pytest.raises(ContractError, match="at least one alternate"):
            measure_convergence_burn_in(primary, [], tick_dt=0.001)
        with pytest.raises(ContractError, match="tick_dt"):
            measure_convergence_burn_in(primary, [primary.copy()], tick_dt=float("nan"))
        with pytest.raises(ContractError, match="settle_factor"):
            measure_convergence_burn_in(primary, [primary.copy()], tick_dt=0.001, settle_factor=0.5)

    def test_operator_stop_rejected(self):
        data = self._collected()
        data["safety_controller"]["operator_stop"] = True
        with pytest.raises(ContractError, match="operator_stop"):
            validate_contract(data)

    def test_operator_stop_allowed_as_diagnostics(self):
        data = self._collected()
        data["safety_controller"]["operator_stop"] = True
        validate_contract(data, allow_truncated=True)
