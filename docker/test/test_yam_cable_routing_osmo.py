# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the checksum-gated YAM cable-routing OSMO workflow."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import tomllib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CLUSTER_DIR = REPO_ROOT / "docker" / "cluster"
WORKFLOW = CLUSTER_DIR / "yam_cable_routing_osmo_workflow.yaml"
MULTINODE_WORKFLOW = CLUSTER_DIR / "yam_cable_routing_multinode_osmo_workflow.yaml"
RUNNER = CLUSTER_DIR / "run_yam_cable_routing_osmo.sh"
MULTINODE_RUNNER = CLUSTER_DIR / "run_yam_cable_routing_multinode_node.sh"
PACKAGER = CLUSTER_DIR / "package_yam_cable_routing_source.sh"
SUBMITTER = CLUSTER_DIR / "submit_yam_cable_routing_osmo.sh"
PREPARER = CLUSTER_DIR / "prepare_yam_cable_routing_source.sh"
RANK_ENTRYPOINT = CLUSTER_DIR / "yam_cable_routing_rank_entrypoint.py"
PREFLIGHT = CLUSTER_DIR / "yam_cable_routing_preflight.py"
DDP_BARRIER = CLUSTER_DIR / "yam_cable_routing_ddp_barrier.py"
SOURCE_FETCHER = CLUSTER_DIR / "yam_cable_routing_source_fetch.py"
COMMANDS = (
    REPO_ROOT / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "contrib" / "cable_routing" / "mdp" / "commands.py"
)


def _load_rank_module():
    spec = importlib.util.spec_from_file_location("yam_cable_routing_rank_entrypoint", RANK_ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_source_fetch_module():
    spec = importlib.util.spec_from_file_location("yam_cable_routing_source_fetch", SOURCE_FETCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_source_package(directory: Path) -> str:
    directory.mkdir()
    archive = b"checksum-gated source archive\n"
    git_status = b" M docker/cluster/run_yam_cable_routing_osmo.sh\n"
    archive_sha256 = hashlib.sha256(archive).hexdigest()
    git_status_sha256 = hashlib.sha256(git_status).hexdigest()
    (directory / "source.tar.gz").write_bytes(archive)
    (directory / "source.metadata").write_text(
        f"source_sha256={archive_sha256}\ngit_status_sha256={git_status_sha256}\n",
        encoding="utf-8",
    )
    (directory / "git-status.txt").write_bytes(git_status)
    (directory / "source.sha256").write_text(f"{archive_sha256}\n", encoding="utf-8")
    return archive_sha256


def _start_source_server(directory: Path) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source_url = f"http://127.0.0.1:{server.server_address[1]}"
    return server, thread, source_url


def _checkpoint_verifier() -> str:
    runner = RUNNER.read_text(encoding="utf-8")
    return runner.split("# BEGIN verify_training_checkpoints", 1)[1].split("# END verify_training_checkpoints", 1)[0]


def _load_worker_template(num_workers: int) -> dict:
    """Render the worker loop without requiring an OSMO Jinja runtime."""
    source = WORKFLOW.read_text(encoding="utf-8")
    loop_start = "# {% for worker_index in range(num_workers) %}\n"
    loop_end = "# {% endfor %}\n"
    prefix, loop_and_suffix = source.split(loop_start, 1)
    worker, suffix = loop_and_suffix.split(loop_end, 1)
    source = prefix + "".join(worker.replace("{{ worker_index }}", str(index)) for index in range(num_workers)) + suffix
    return yaml.safe_load(source)


def _load_single_worker_template() -> dict:
    """Render one worker for static workflow assertions."""
    return _load_worker_template(1)


def _load_multinode_template(num_nodes: int = 8) -> dict:
    """Render the multi-node loop and lead conditional without OSMO Jinja."""
    source = MULTINODE_WORKFLOW.read_text(encoding="utf-8")
    loop_start = "# {% for node_index in range(num_nodes) %}\n"
    loop_end = "# {% endfor %}\n"
    conditional_start = "# {% if node_index == 0 %}\n"
    conditional_end = "# {% endif %}\n"
    prefix, loop_and_suffix = source.split(loop_start, 1)
    node_template, suffix = loop_and_suffix.split(loop_end, 1)
    rendered_nodes = []
    for node_index in range(num_nodes):
        before_conditional, conditional_and_after = node_template.split(conditional_start, 1)
        conditional, after_conditional = conditional_and_after.split(conditional_end, 1)
        if "# {% else %}\n" in conditional:
            lead_conditional, peer_conditional = conditional.split("# {% else %}\n", 1)
        else:
            lead_conditional, peer_conditional = conditional, ""
        rendered_node = before_conditional
        if node_index == 0:
            rendered_node += lead_conditional
        else:
            rendered_node += peer_conditional
        rendered_node += after_conditional
        rendered_nodes.append(rendered_node.replace("{{ node_index }}", str(node_index)))
    return yaml.safe_load(prefix + "".join(rendered_nodes) + suffix)


def test_workflow_uses_official_image_secret_mapping_and_newton_preflight() -> None:
    workflow = _load_single_worker_template()
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    defaults = workflow["default-values"]
    source_prep, trainer = workflow["workflow"]["tasks"]

    assert defaults["image"].startswith("nvcr.io/nvidia/isaac-lab:")
    assert defaults["platform"] == "ovx-l40s"
    assert defaults["num_gpu"] == 1
    assert defaults["num_workers"] == 1
    assert defaults["gpus_per_worker"] == 1
    assert defaults["num_cpu"] == 4
    assert defaults["memory_gi"] == 24
    assert defaults["storage_gi"] == 32
    assert defaults["newton_revision"] == project["tool"]["isaaclab"]["versions"]["newton"]
    assert defaults["warp_version"] == project["tool"]["isaaclab"]["versions"]["warp"]
    assert source_prep["name"] == "source-prep"
    assert source_prep["resource"] == "source"
    assert source_prep["files"] == [
        {
            "path": "/tmp/prepare-yam-cable-routing-source.sh",
            "localpath": "prepare_yam_cable_routing_source.sh",
        }
    ]
    assert workflow["workflow"]["resources"]["trainer"]["gpu"] == "{{ gpus_per_worker }}"
    assert trainer["credentials"] == {"wandb": {"WANDB_API_KEY": "wandb_api_key"}}
    assert trainer["name"] == "trainer-0"
    assert trainer["inputs"] == [{"task": "source-prep"}]
    assert trainer["environment"]["OSMO_RUN_MODE"] == "{{ run_mode }}"
    assert trainer["environment"]["OSMO_CAMPAIGN_TASK_MODE"] == "{{ campaign_task_mode }}"
    assert trainer["environment"]["OSMO_SOURCE_SHA256"] == "{{ source_sha256 }}"
    assert trainer["environment"]["OSMO_SOURCE_SYNC"] == "{{input:0}}/source-package"
    localpaths = {entry["localpath"] for entry in trainer["files"]}
    assert localpaths == {
        "run_yam_cable_routing_osmo.sh",
        "yam_cable_routing_preflight.py",
        "yam_cable_routing_rank_entrypoint.py",
    }

    preflight = PREFLIGHT.read_text(encoding="utf-8")
    assert "cable_capsule_clearance_mask" in preflight
    assert "cable_capsule_self_clearance_mask" in preflight
    live_clearance = preflight.split("clear = cable_capsule_clearance_mask(", 1)[1].split(")", 1)[0]
    assert "fixture_clearance=0.0" in live_clearance
    assert "board_bounds_b=command.cfg.settled_cable_bounds_b" in live_clearance
    assert "board_clearance=0.0" in live_clearance
    assert "cable_unrouted_mask" not in preflight
    assert "SuccessMonitor" in preflight
    assert "def _assert_reward_cfg" in preflight
    assert '"success": 20.0' in preflight
    assert '"stretch": -0.25' in preflight
    assert '"action_rate": -0.002' in preflight
    assert '"left_joint_velocity": -0.0001' in preflight
    assert '"right_joint_velocity": -0.0001' in preflight
    assert '"reward_cfg": reward_cfg' in preflight
    assert "CableResetCurveXPBDCfg" in preflight
    assert "relax_open_cable_curve_xpbd" in preflight
    assert "replay_cfg.curve_projection_iterations != 50" in preflight
    assert "replay_cfg.max_curve_attempts < 512" in preflight
    assert '"curve_projection_iterations": replay_cfg.curve_projection_iterations' in preflight
    assert "def _assert_xpbd_reset_projection" in preflight
    assert "projection_cfg.taubin_smoothing_passes != 5" in preflight
    assert '"maximum_turning_angle_rad": maximum_turning_angle' in preflight
    assert "XPBD reset projection did not repair cable spacing" in preflight
    assert '"xpbd_reset_projection": xpbd_reset_projection' in preflight
    assert "replay.state_buffer.capacity" in preflight
    assert "sampled_progress" in preflight
    assert "sampled_requested_progress" in preflight
    assert "Reset replay bank contains no rows for route" in preflight
    assert "does not cover all 16 progress bins" in preflight
    assert "active_step_progress_from_route_progress" in preflight
    assert "live_active_progress" in preflight
    assert "valid_live_progress" in preflight
    assert "live_requested_error < replay.cfg.post_settle_progress_tolerance" in preflight
    assert "Live restored route progress violates the replay acceptance bounds" in preflight
    assert "Live route progress does not match restored replay metadata" not in preflight
    assert '"maximum_live_progress_metadata_drift"' in preflight
    assert "Reset replay sampled progress outside the authored near-goal interval" not in preflight
    assert "distinct_sampled_sources" in preflight
    assert "command.succeeded" in preflight
    assert "max_zero_action_transition_success_fraction" in preflight
    assert "zero_action_policy_steps" in preflight
    assert "zero_action_transitions" in preflight
    assert "zero_action_transition_success_fraction" in preflight
    assert "source_before_step[success]" in preflight
    assert "All-zero/open-gripper actions completed" in preflight
    assert "math.floor(" in preflight
    assert "math.ceil(" not in preflight
    assert "assert_zero_action_did_not_route" not in preflight
    assert '"zero_action_route_successes": 0' not in preflight
    assert "sampled_active_progress < replay.cfg.maximum_settled_active_progress" in preflight
    commands = COMMANDS.read_text(encoding="utf-8")
    assert "maximum_active_progress=cfg.maximum_settled_active_progress" in commands
    assert "roundtrip_state = self._env.scene.get_state(is_relative=True)" in commands
    assert "self._env.scene.reset_to(roundtrip_state, env_ids=all_ids, is_relative=True)" in commands
    assert "_assert_contact_stress" in preflight
    assert "rigid_body_contact_buffer_size" in preflight
    assert "overflow_contacts_per_body" in preflight
    assert "actions[:, 6] = 1.0" in preflight
    assert "actions[:, 13] = 1.0" in preflight
    assert ".clone()" in preflight
    assert "opened_left - closed_left" in preflight
    assert "opened_left + opened_right" in preflight
    assert "closed_left + closed_right" in preflight
    assert "float(left.amin().item())" not in preflight
    assert "Non-finite YAM finger position" in preflight
    assert "YAM_OPEN_MIRROR_TOLERANCE_M = 2.0e-3" in preflight
    assert "YAM_CABLE_LOADED_MIRROR_TOLERANCE_M = 5.0e-3" in preflight
    assert "maximum_open_finger_mirror_error_m" in preflight
    assert "maximum_cable_loaded_finger_mirror_error_m" in preflight

    runner = RUNNER.read_text(encoding="utf-8")
    assert 'export WANDB_USERNAME="${OSMO_WANDB_ENTITY}"' in runner
    assert 'export WANDB_PROJECT="${OSMO_WANDB_PROJECT}"' in runner
    assert 'print("wandb_project_access=verified")' in runner
    assert '-f "${source_sync}/source.metadata"' in runner
    assert '-f "${source_sync}/git-status.txt"' in runner
    assert 'source_sync="${OSMO_SOURCE_SYNC:-/osmo/run/workspace/source-sync}"' in runner
    assert "yam-source-accepted.sha256" in runner
    newton_requirement = (
        '"newton[sim,importers] @ git+https://github.com/newton-physics/newton.git@${OSMO_NEWTON_REVISION}"'
    )
    assert newton_requirement in runner
    assert '"warp-lang==${OSMO_WARP_VERSION}"' in runner
    assert 'if not hasattr(newton, "ModelFlags")' in runner
    assert "export NO_NUCLEUS=1" in runner
    assert "unset NO_NUCLEUS" not in runner
    assert "install_newton_runtime" in runner
    assert "--retries 10" in runner
    assert "--timeout 120" in runner
    assert "runtime_install_attempt >= 3" in runner
    assert "runtime-install-attempts.txt" in runner
    assert "run_status=12" in runner
    assert 'cp -R --no-preserve=all "${live_results}/." "${final_results}/"' in runner
    assert 'cp -a "${live_results}/." "${final_results}/"' not in runner
    assert 'training_artifacts="${live_results}/training-artifacts"' in runner
    assert 'preflight_results="${training_artifacts}/preflight"' in runner
    assert 'provenance="${training_artifacts}/run-info"' in runner
    assert 'exec > >(tee -a "${console_logs}/osmo-run.log") 2>&1' in runner
    assert 'ln -s "${slot_training_root}" "${rank_root}/logs"' in runner
    assert 'ln -s "${training_runs}/ddp" "${ddp_root}/logs"' in runner
    assert 'sha256sum --check "training-artifacts/run-info/checkpoint-manifest.sha256"' in runner
    assert "Per-body (rigid|particle) contact buffer overflowed" in runner
    assert "Maximum logging rate exceeded" in runner
    assert "checkpoints are invalid" in runner
    assert '"env.commands.route.reset_replay.seed=${seed}"' in runner
    assert "run_newton_preflight" in runner
    assert "preflight_status == 134 || preflight_status == 139" in runner
    assert "retrying once in a fresh process" in runner
    assert 'exit "${preflight_status}"' in runner
    assert "preflight-attempts.txt" in runner

    preparer = PREPARER.read_text(encoding="utf-8")
    assert "yam-source-release.sha256" in preparer
    assert "reset_curve_xpbd.py" in preparer
    assert "reset_curves.py" in preparer
    assert "reset_replay.py" in preparer
    assert "reset_robot_targets.py" in preparer
    # OSMO renders injected files through Jinja; Bash array-length syntax starts
    # with the Jinja comment opener and therefore cannot appear in this script.
    assert "{#" not in runner
    assert '"${live_results}/logs"' not in runner
    assert '"${live_results}/metadata"' not in runner


@pytest.mark.parametrize(
    ("run_mode", "slots"),
    [
        ("smoke", ["smoke"]),
        ("campaign", ["tier1-s47"]),
        ("ddp", ["ddp"]),
        ("multinode", ["ddp"]),
    ],
)
def test_checkpoint_verifier_hashes_each_successful_training_slot(
    tmp_path: Path, run_mode: str, slots: list[str]
) -> None:
    live_results = tmp_path / "results"
    training_runs = live_results / "training-artifacts" / "runs"
    provenance = live_results / "training-artifacts" / "run-info"
    provenance.mkdir(parents=True)
    for index, slot in enumerate(slots):
        checkpoint = training_runs / slot / "rsl_rl" / "run" / "model_1.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"checkpoint-{index}".encode())

    harness = "\n".join(
        [
            "set -euo pipefail",
            f"live_results={shlex.quote(str(live_results))}",
            f"training_runs={shlex.quote(str(training_runs))}",
            f"provenance={shlex.quote(str(provenance))}",
            f"OSMO_RUN_MODE={shlex.quote(run_mode)}",
            "OSMO_MAX_ITERATIONS=2",
            *(["campaign_slot_label=tier1-s47"] if run_mode == "campaign" else []),
            _checkpoint_verifier(),
            "verify_training_checkpoints",
        ]
    )
    subprocess.run(["bash", "-c", harness], check=True)

    manifest = provenance / "checkpoint-manifest.sha256"
    inventory = (provenance / "checkpoint-inventory.tsv").read_text(encoding="utf-8")
    validation = (provenance / "checkpoint-validation.txt").read_text(encoding="utf-8")
    assert manifest.is_file()
    assert inventory.count("\tfinal\t") == len(slots)
    assert f"expected_slots={len(slots)}\n" in validation
    assert f"checkpoint_count={len(slots)}\n" in validation
    subprocess.run(
        ["sha256sum", "--check", str(manifest.relative_to(live_results))],
        cwd=live_results,
        check=True,
    )


def test_checkpoint_verifier_fails_closed_when_a_final_checkpoint_is_missing(tmp_path: Path) -> None:
    live_results = tmp_path / "results"
    training_runs = live_results / "training-artifacts" / "runs"
    provenance = live_results / "training-artifacts" / "run-info"
    (training_runs / "smoke").mkdir(parents=True)
    provenance.mkdir(parents=True)
    harness = "\n".join(
        [
            "set -euo pipefail",
            f"live_results={shlex.quote(str(live_results))}",
            f"training_runs={shlex.quote(str(training_runs))}",
            f"provenance={shlex.quote(str(provenance))}",
            "OSMO_RUN_MODE=smoke",
            "OSMO_MAX_ITERATIONS=2",
            _checkpoint_verifier(),
            "verify_training_checkpoints",
        ]
    )

    result = subprocess.run(["bash", "-c", harness], text=True, capture_output=True, check=False)

    assert result.returncode == 14
    assert "Missing final model_1.pt checkpoint for training slot smoke." in result.stderr
    assert not (provenance / "checkpoint-manifest.sha256").exists()


def test_campaign_round_robins_four_goals_or_focuses_one_task_across_up_to_32_workers() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    campaign = runner.split("campaign_tasks=(", 1)[1].split(")", 1)[0]
    assert campaign.count("IsaacContrib-CableRouting-YAM-Peg0-CCW") == 1
    assert campaign.count("IsaacContrib-CableRouting-YAM-Peg1-CW") == 1
    assert campaign.count("IsaacContrib-CableRouting-YAM-Tier1-Pegs") == 1
    assert campaign.count("IsaacContrib-CableRouting-YAM\n") == 1
    assert 'goal_index="$(( slot % 4 ))"' in runner
    assert 'campaign_seed="$(( OSMO_SEED + slot ))"' in runner
    assert "OSMO_CAMPAIGN_SIZE > 32" in runner
    assert 'device_token="${device_tokens[0]//[[:space:]]/}"' in runner
    assert 'case "${OSMO_CAMPAIGN_TASK_MODE}" in' in runner
    assert 'campaign_task="${OSMO_TASK}"' in runner
    assert "campaign_goal_label=tier1" in runner
    assert "campaign_goal_label=seven-goals" in runner
    assert 'campaign_slot_label="${campaign_goal_label}-s${campaign_seed}"' in runner
    assert "campaign-status.tsv" in runner


@pytest.mark.parametrize("worker_count", [8, 32])
def test_campaign_template_renders_independent_one_gpu_workers(worker_count: int) -> None:
    workflow = _load_worker_template(worker_count)
    source_prep, *trainers = workflow["workflow"]["tasks"]

    assert source_prep["name"] == "source-prep"
    assert [trainer["name"] for trainer in trainers] == [f"trainer-{index}" for index in range(worker_count)]
    assert all(trainer["resource"] == "trainer" for trainer in trainers)
    assert all(trainer["inputs"] == [{"task": "source-prep"}] for trainer in trainers)
    assert [trainer["environment"]["OSMO_CAMPAIGN_INDEX"] for trainer in trainers] == [
        str(index) for index in range(worker_count)
    ]
    assert all(trainer["environment"]["OSMO_NUM_GPU"] == "{{ gpus_per_worker }}" for trainer in trainers)
    assert workflow["workflow"]["resources"]["trainer"]["gpu"] == "{{ gpus_per_worker }}"

    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert 'num_workers="${num_gpu}"' in submitter
    assert "gpus_per_worker=1" in submitter
    assert "campaign_task_mode=round_robin" in submitter
    assert "campaign_task_mode=focused" in submitter
    assert '"campaign_task_mode=${campaign_task_mode}"' in submitter
    assert '"preflight_task=${task}"' in submitter
    assert '"${run_mode}" == campaign && "${campaign_task_mode}" == focused' in submitter
    assert "Unsupported focused cable-routing task" in submitter


def test_multinode_template_renders_one_synchronized_32_rank_world() -> None:
    workflow = _load_multinode_template()
    defaults = workflow["default-values"]
    groups = workflow["workflow"]["groups"]
    assert len(groups) == 1
    training_group = groups[0]
    trainers = training_group["tasks"]

    assert defaults["run_mode"] == "multinode"
    assert defaults["num_nodes"] == 8
    assert defaults["gpus_per_node"] == 4
    assert defaults["num_nodes"] * defaults["gpus_per_node"] == defaults["num_gpu"] == 32
    assert defaults["num_envs"] * defaults["num_gpu"] == 8192
    assert defaults["num_mini_batches"] == 4
    assert defaults["task"] == "IsaacContrib-CableRouting-YAM-SevenGoals"
    assert defaults["preflight_task"] == defaults["task"]
    assert defaults["source_port"] == 29402
    assert defaults["num_envs"] * 32 * defaults["num_gpu"] // defaults["num_mini_batches"] == 65536
    assert "tasks" not in workflow["workflow"]
    assert training_group["name"] == "training"
    assert "source" not in workflow["workflow"]["resources"]
    assert all(task["name"] != "source-prep" for task in trainers)
    assert workflow["workflow"]["resources"]["trainer"]["gpu"] == "{{ gpus_per_node }}"
    assert [trainer["name"] for trainer in trainers] == [f"trainer-node-{index}" for index in range(8)]
    assert all(trainer["resource"] == "trainer" for trainer in trainers)
    assert all("inputs" not in trainer for trainer in trainers)
    lead, *peers = trainers
    assert all(
        trainer["environment"]["OSMO_SOURCE_URL"] == "http://{{host:trainer-node-0}}:{{ source_port }}"
        for trainer in trainers
    )
    assert all(trainer["environment"]["OSMO_SOURCE_SERVE_PORT"] == "{{ source_port }}" for trainer in trainers)
    assert all("OSMO_SOURCE_SYNC" not in trainer["environment"] for trainer in trainers)
    assert "prepare_yam_cable_routing_source.sh" in {file["localpath"] for file in lead["files"]}
    assert all(
        "yam_cable_routing_source_fetch.py" in {file["localpath"] for file in trainer["files"]} for trainer in peers
    )
    assert [trainer["environment"]["OSMO_NODE_RANK"] for trainer in trainers] == [str(index) for index in range(8)]
    assert all(trainer["environment"]["OSMO_NUM_NODES"] == "{{ num_nodes }}" for trainer in trainers)
    assert all(trainer["environment"]["OSMO_NUM_GPU"] == "{{ gpus_per_node }}" for trainer in trainers)
    assert all(trainer["environment"]["OSMO_NUM_MINI_BATCHES"] == "{{ num_mini_batches }}" for trainer in trainers)
    assert all(trainer["environment"]["OSMO_TOTAL_GPU"] == "{{ num_gpu }}" for trainer in trainers)
    assert all(trainer["environment"]["OSMO_NUM_ENVS"] == "{{ num_envs }}" for trainer in trainers)
    assert all(trainer["environment"]["OSMO_MASTER_ADDR"] == "{{host:trainer-node-0}}" for trainer in trainers)
    assert all(trainer["environment"]["OSMO_MASTER_PORT"] == "{{ master_port }}" for trainer in trainers)
    assert all(trainer["environment"]["OSMO_RUN_MODE"] == "{{ run_mode }}" for trainer in trainers)
    assert all(trainer["environment"]["OSMO_TASK"] == "{{ task }}" for trainer in trainers)

    assert sum(task.get("lead") is True for task in training_group["tasks"]) == 1
    assert lead["lead"] is True
    assert all("lead" not in trainer for trainer in peers)

    submitter = SUBMITTER.read_text(encoding="utf-8")
    multinode_defaults = submitter.split("    multinode)\n", 1)[1].split("    *)\n", 1)[0]
    assert 'workflow_file="${script_dir}/yam_cable_routing_multinode_osmo_workflow.yaml"' in multinode_defaults
    assert "num_gpu=32" in multinode_defaults
    assert "gpus_per_node=4" in multinode_defaults
    assert "num_envs=256" in multinode_defaults
    assert "num_mini_batches=4" in multinode_defaults
    assert "max_iterations=3000" in multinode_defaults
    assert "seed=47" in multinode_defaults
    assert "task=IsaacContrib-CableRouting-YAM-SevenGoals" in multinode_defaults
    assert 'num_nodes="$(( num_gpu / gpus_per_node ))"' in submitter
    assert '"num_nodes=${num_nodes}"' in submitter
    assert '"gpus_per_node=${gpus_per_node}"' in submitter
    assert "IsaacContrib-CableRouting-YAM-SevenGoals" in submitter
    assert "multinode_goal_label=seven-goals" in RUNNER.read_text(encoding="utf-8")


def test_multinode_lead_source_preparer_serves_validated_source_after_submitter_release() -> None:
    preparer = PREPARER.read_text(encoding="utf-8")

    assert 'source_serve_port="${OSMO_SOURCE_SERVE_PORT:-}"' in preparer
    assert "OSMO_SOURCE_SERVE_PORT must be an integer from 1 through 65535." in preparer
    assert 'if [[ -z "${source_serve_port}" ]]; then' in preparer
    assert "docker/cluster/yam_cable_routing_source_fetch.py" in preparer
    assert 'exec "${image_python}" -m http.server' in preparer
    assert "--bind 0.0.0.0" in preparer
    assert '--directory "${source_sync}"' in preparer

    checksum_validation = preparer.index('actual_sha="$(sha256sum "${source_sync}/source.tar.gz"')
    acceptance = preparer.index('mv "${source_acceptance_tmp}" "${source_acceptance}"')
    release = preparer.index("if (( source_released == 0 )); then")
    server = preparer.index('exec "${image_python}" -m http.server')
    assert checksum_validation < acceptance < release < server

    wrapper = MULTINODE_RUNNER.read_text(encoding="utf-8")
    source_start = wrapper.index('bash "${source_preparer}" &')
    source_probe = wrapper.index("source_server_ready")
    local_source = wrapper.index("export OSMO_SOURCE_SYNC=")
    training = wrapper.index('bash "${runner}"')
    assert source_start < source_probe < local_source < training
    assert "unset OSMO_SOURCE_URL" in wrapper
    assert 'kill "${source_server_pid}"' in wrapper


def test_multinode_lead_runtime_gates_runner_on_released_matching_source_and_cleans_server(tmp_path: Path) -> None:
    expected_sha256 = hashlib.sha256(b"released source").hexdigest()
    wrong_sha256 = hashlib.sha256(b"unreleased source").hexdigest()
    served = tmp_path / "served"
    served.mkdir()
    (served / "source.sha256").write_text(f"{wrong_sha256}\n", encoding="utf-8")
    preparer_started = tmp_path / "preparer-started"
    release = tmp_path / "source-release.sha256"
    runner_invoked = tmp_path / "runner-invoked"
    server_pid_file = tmp_path / "source-server.pid"
    probe_log = tmp_path / "source-probes.tsv"
    barrier_complete = tmp_path / "barrier-complete"

    def write_script(path: Path, contents: str) -> None:
        path.write_text(contents, encoding="utf-8")
        path.chmod(0o755)

    bootstrap = tmp_path / "bootstrap-python.sh"
    write_script(
        bootstrap,
        "\n".join(
            [
                "#!/usr/bin/env bash",
                '"${HARNESS_REAL_PYTHON}" "$@"',
                "status=$?",
                'if [[ "${1-}" == - ]]; then',
                '    printf \'probe_status=%s\\n\' "${status}" >> "${HARNESS_PROBE_LOG}"',
                "fi",
                'exit "${status}"',
            ]
        ),
    )
    preparer = tmp_path / "prepare-source.sh"
    write_script(
        preparer,
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'printf \'%s\\n\' "$$" > "${HARNESS_SERVER_PID_FILE}"',
                'touch "${HARNESS_PREPARER_STARTED}"',
                "while true; do",
                '    if [[ -f "${HARNESS_RELEASE}" ]] && '
                '[[ "$(tr -d \'[:space:]\' < "${HARNESS_RELEASE}")" == "${OSMO_SOURCE_SHA256}" ]]; then',
                "        break",
                "    fi",
                "    sleep 0.01",
                "done",
                'exec "${HARNESS_REAL_PYTHON}" -m http.server "${OSMO_SOURCE_SERVE_PORT}" '
                '--bind 127.0.0.1 --directory "${HARNESS_SERVED_DIR}"',
            ]
        ),
    )
    runner = tmp_path / "runner.sh"
    write_script(
        runner,
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'touch "${HARNESS_RUNNER_INVOKED}"',
                'served_sha="$("${HARNESS_REAL_PYTHON}" -c \'import sys, urllib.request; '
                "print(urllib.request.urlopen(sys.argv[1], timeout=1).read().decode().strip())' "
                '"http://127.0.0.1:${OSMO_SOURCE_SERVE_PORT}/source.sha256")"',
                '[[ "${served_sha}" == "${OSMO_SOURCE_SHA256}" ]]',
                '[[ "$(tr -d \'[:space:]\' < "${HARNESS_RELEASE}")" == "${OSMO_SOURCE_SHA256}" ]]',
            ]
        ),
    )
    barrier = tmp_path / "barrier.py"
    barrier.write_text(
        "\n".join(
            [
                "import os",
                "import pathlib",
                "import sys",
                "import time",
                "complete = pathlib.Path(os.environ['HARNESS_BARRIER_COMPLETE'])",
                "if sys.argv[1] == 'serve':",
                "    deadline = time.monotonic() + 5",
                "    while not complete.exists() and time.monotonic() < deadline:",
                "        time.sleep(0.01)",
                "    raise SystemExit(0 if complete.exists() else 1)",
                "if sys.argv[1] == 'report' and sys.argv[sys.argv.index('--phase') + 1] == 'complete':",
                "    complete.touch()",
                "raise SystemExit(0)",
            ]
        ),
        encoding="utf-8",
    )

    wrapper = MULTINODE_RUNNER.read_text(encoding="utf-8")
    wrapper = wrapper.replace("runner=/tmp/run-yam-cable-routing-osmo.sh", f"runner={shlex.quote(str(runner))}")
    wrapper = wrapper.replace("barrier=/tmp/yam-cable-routing-ddp-barrier.py", f"barrier={shlex.quote(str(barrier))}")
    wrapper = wrapper.replace(
        "source_preparer=/tmp/prepare-yam-cable-routing-source.sh",
        f"source_preparer={shlex.quote(str(preparer))}",
    )
    wrapper_path = tmp_path / "multinode-wrapper.sh"
    wrapper_path.write_text(wrapper, encoding="utf-8")

    ports: list[int] = []
    while len(ports) < 3:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            candidate = reservation.getsockname()[1]
        if candidate not in ports:
            ports.append(candidate)
    master_port, completion_port, source_port = ports
    environment = {
        **os.environ,
        "HARNESS_BARRIER_COMPLETE": str(barrier_complete),
        "HARNESS_PREPARER_STARTED": str(preparer_started),
        "HARNESS_PROBE_LOG": str(probe_log),
        "HARNESS_REAL_PYTHON": sys.executable,
        "HARNESS_RELEASE": str(release),
        "HARNESS_RUNNER_INVOKED": str(runner_invoked),
        "HARNESS_SERVED_DIR": str(served),
        "HARNESS_SERVER_PID_FILE": str(server_pid_file),
        "OSMO_BARRIER_STARTUP_TIMEOUT_SECONDS": "311",
        "OSMO_COMPLETION_PORT": str(completion_port),
        "OSMO_IMAGE_PYTHON": str(bootstrap),
        "OSMO_MASTER_ADDR": "127.0.0.1",
        "OSMO_MASTER_PORT": str(master_port),
        "OSMO_NODE_RANK": "0",
        "OSMO_NUM_GPU": "1",
        "OSMO_NUM_NODES": "1",
        "OSMO_OUTPUT_DIR": str(tmp_path / "output"),
        "OSMO_RUN_MODE": "multinode",
        "OSMO_SOURCE_FETCH_WAIT_SECONDS": "310",
        "OSMO_SOURCE_SERVE_PORT": str(source_port),
        "OSMO_SOURCE_SHA256": expected_sha256,
        "OSMO_SOURCE_STARTUP_WAIT_SECONDS": "10",
        "OSMO_TOTAL_GPU": "1",
        "OSMO_VCS_REF": "0" * 40,
        "OSMO_WORKFLOW_ID": "functional-wrapper-test",
    }

    process = subprocess.Popen(
        ["bash", str(wrapper_path)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def wait_until(predicate, description: str, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout is not None else ""
                pytest.fail(f"Wrapper exited before {description} (status {process.returncode}):\n{output}")
            time.sleep(0.01)
        pytest.fail(f"Timed out waiting for {description}.")

    try:
        wait_until(preparer_started.exists, "the source preparer")
        assert not runner_invoked.exists()

        release.write_text(f"{expected_sha256}\n", encoding="utf-8")

        def observed_rejected_probe() -> bool:
            return probe_log.exists() and "probe_status=1" in probe_log.read_text(encoding="utf-8")

        wait_until(observed_rejected_probe, "a rejected mismatching source probe")
        assert not runner_invoked.exists()

        matching_source = served / ".source.sha256.tmp"
        matching_source.write_text(f"{expected_sha256}\n", encoding="utf-8")
        os.replace(matching_source, served / "source.sha256")
        stdout, _ = process.communicate(timeout=10)
        assert process.returncode == 0, stdout
        assert runner_invoked.exists()
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=5)

    server_pid = int(server_pid_file.read_text(encoding="utf-8").strip())
    with pytest.raises(ProcessLookupError):
        os.kill(server_pid, 0)
    with pytest.raises((OSError, urllib.error.URLError)):
        urllib.request.urlopen(f"http://127.0.0.1:{source_port}/source.sha256", timeout=0.2)


def test_multinode_runner_fetches_peer_source_with_the_rendered_checksum() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    fetch = runner.split('if [[ -n "${OSMO_SOURCE_URL:-}" ]]; then', 1)[1].split(
        "# OSMO's rsync daemon starts with the workflow.", 1
    )[0]

    assert "OSMO_SOURCE_URL and OSMO_SOURCE_SYNC are mutually exclusive." in fetch
    assert "source_fetcher=/tmp/yam-cable-routing-source-fetch.py" in fetch
    assert '"${image_python}" "${source_fetcher}"' in fetch
    assert '--source-url "${OSMO_SOURCE_URL}"' in fetch
    assert '--destination "${source_sync}"' in fetch
    assert '--expected-sha256 "${OSMO_SOURCE_SHA256}"' in fetch


def test_source_fetcher_downloads_http_package_and_publishes_readiness_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_fetch = _load_source_fetch_module()
    served_package = tmp_path / "served"
    expected_sha256 = _write_source_package(served_package)
    destination = tmp_path / "downloaded"
    published_names: list[str] = []
    replace = source_fetch.os.replace

    def record_replace(source: Path, target: Path) -> None:
        published_names.append(Path(target).name)
        replace(source, target)

    monkeypatch.setattr(source_fetch.os, "replace", record_replace)
    server, server_thread, source_url = _start_source_server(served_package)
    try:
        source_fetch.fetch_source_package(
            source_url,
            destination,
            expected_sha256,
            wait_seconds=2.0,
            retry_seconds=0.01,
            request_timeout_seconds=1.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    assert published_names == ["source.tar.gz", "source.metadata", "git-status.txt", "source.sha256"]
    for file_name in published_names:
        assert (destination / file_name).read_bytes() == (served_package / file_name).read_bytes()
    assert not list(tmp_path.glob(".downloaded-fetch-*"))


@pytest.mark.parametrize(
    ("tampered_file", "error"),
    [
        ("source.tar.gz", "source.tar.gz digest"),
        ("git-status.txt", "git-status.txt digest"),
    ],
)
def test_source_fetcher_rejects_checksum_mismatch_without_replacing_ready_package(
    tmp_path: Path, tampered_file: str, error: str
) -> None:
    source_fetch = _load_source_fetch_module()
    served_package = tmp_path / "served"
    expected_sha256 = _write_source_package(served_package)
    (served_package / tampered_file).write_bytes((served_package / tampered_file).read_bytes() + b"tampered")
    destination = tmp_path / "downloaded"
    destination.mkdir()
    previous_package = {
        "source.tar.gz": b"previous archive",
        "source.metadata": b"previous metadata",
        "git-status.txt": b"previous status",
        "source.sha256": b"previous readiness marker",
    }
    for file_name, contents in previous_package.items():
        (destination / file_name).write_bytes(contents)

    server, server_thread, source_url = _start_source_server(served_package)
    try:
        with pytest.raises(ValueError, match=error):
            source_fetch._fetch_once(source_url, destination, expected_sha256, request_timeout_seconds=1.0)
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    assert {path.name: path.read_bytes() for path in destination.iterdir()} == previous_package
    assert not list(tmp_path.glob(".downloaded-fetch-*"))


def test_source_fetcher_retries_while_peer_server_starts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_fetch = _load_source_fetch_module()
    attempts = 0

    def fetch_once(*_args, **_kwargs) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("peer server is not listening yet")

    monkeypatch.setattr(source_fetch, "_fetch_once", fetch_once)
    source_fetch.fetch_source_package(
        "http://source-prep:29402",
        tmp_path / "downloaded",
        "0" * 64,
        wait_seconds=1.0,
        retry_seconds=0.001,
        request_timeout_seconds=0.1,
    )

    assert attempts == 3


def test_multinode_lead_waits_for_every_node_terminal_status() -> None:
    wrapper = MULTINODE_RUNNER.read_text(encoding="utf-8")

    assert "OSMO_NUM_NODES * OSMO_NUM_GPU != OSMO_TOTAL_GPU" in wrapper
    assert 'if (( OSMO_NODE_RANK == 0 )); then\n    "${image_python}" "${barrier}" serve' in wrapper
    assert "--phase probe" in wrapper
    assert '--expected-nodes "${OSMO_NUM_NODES}"' in wrapper
    assert '"${image_python}" "${barrier}" report' in wrapper
    assert '--node-rank "${OSMO_NODE_RANK}"' in wrapper
    assert "--phase complete" in wrapper
    assert '--status "${node_status}"' in wrapper
    assert 'wait "${barrier_pid}"' in wrapper
    assert "barrier_status != 0 && node_status == 0" in wrapper
    barrier = DDP_BARRIER.read_text(encoding="utf-8")
    assert "completion_grace_seconds" in barrier
    assert "ddp_barrier_completion_timeout" in barrier


def test_multinode_lead_supervises_runner_with_barrier_and_uses_rendered_startup_timeout() -> None:
    wrapper = MULTINODE_RUNNER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert "os.setsid()" in wrapper
    assert 'wait -n -p completed_pid "${runner_pid}" "${barrier_pid}"' in wrapper
    assert 'kill -TERM -- "-${runner_pid}"' in wrapper
    assert 'kill -KILL -- "-${runner_pid}"' in wrapper
    assert '--timeout-seconds "${OSMO_BARRIER_STARTUP_TIMEOUT_SECONDS}"' in runner
    assert "--timeout-seconds 3900" not in runner


@pytest.mark.parametrize("script", [PREPARER, MULTINODE_RUNNER, RUNNER])
def test_remote_runtime_scripts_use_official_image_python_without_bare_python3(script: Path) -> None:
    contents = script.read_text(encoding="utf-8")
    official_image_python = "/workspace/isaaclab/_isaac_sim/python.sh"
    resolved_contents = contents.replace("${image_root}", "/workspace/isaaclab")

    assert official_image_python in resolved_contents
    assert "python3" not in contents
    if "tar --extract" in contents:
        assert resolved_contents.index(official_image_python) < resolved_contents.index("tar --extract")


def test_multinode_wrapper_accepts_injected_runner_without_executable_mode() -> None:
    wrapper = MULTINODE_RUNNER.read_text(encoding="utf-8")

    assert 'if [[ ! -f "${runner}" || ! -f "${barrier}" ]]; then' in wrapper
    assert '! -x "${runner}"' not in wrapper


def test_multinode_runner_uses_static_torchrun_and_reports_global_environment_count() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    multinode = runner.split("    multinode)\n", 1)[1].split("    *)\n", 1)[0]

    assert '"${python}" -m torch.distributed.run' in multinode
    assert '--nnodes "${OSMO_NUM_NODES}"' in multinode
    assert '--nproc_per_node "${OSMO_NUM_GPU}"' in multinode
    assert '--node_rank "${OSMO_NODE_RANK}"' in multinode
    assert '--master_addr "${OSMO_MASTER_ADDR}"' in multinode
    assert '--master_port "${OSMO_MASTER_PORT}"' in multinode
    assert "--local_ranks_filter 0" in multinode
    assert '--log_dir "${console_logs}/torchrun-ranks"' in multinode
    assert '"${rank_entrypoint}"' in multinode
    assert "--distributed" in multinode
    assert "--standalone" not in multinode
    assert "--rdzv_endpoint" not in multinode
    assert "OSMO_NUM_NODES * OSMO_NUM_GPU != OSMO_TOTAL_GPU" in multinode
    assert "export OSMO_RESET_REPLAY_SEED_PER_RANK=1" in multinode
    assert '"env.commands.route.reset_replay.seed=${OSMO_SEED}"' in multinode
    assert '"agent.algorithm.num_mini_batches=${OSMO_NUM_MINI_BATCHES}"' in multinode
    assert '"$(( OSMO_NUM_ENVS * OSMO_TOTAL_GPU ))"' in runner
    assert "--phase ready" in runner
    assert '"${OSMO_RUN_MODE}" == multinode && "${OSMO_NODE_RANK:-0}" != 0' in runner
    assert "wandb_auth=verified_by_global_rank_zero" in runner


def test_multinode_checkpoints_are_verified_and_published_only_by_global_rank_zero() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    publish_results = runner.split("publish_results() {", 1)[1].split("trap publish_results EXIT", 1)[0]
    checkpoint_completion = runner.split("# END verify_training_checkpoints", 1)[1]

    assert 'if [[ "${OSMO_RUN_MODE}" == multinode && "${OSMO_NODE_RANK:-0}" != 0 ]]; then' in publish_results
    assert "manifest_required=0" in publish_results
    assert "run_status == 0 && manifest_required != 0" in publish_results
    assert 'if [[ "${OSMO_RUN_MODE}" != multinode || "${OSMO_NODE_RANK:-0}" == 0 ]]; then' in checkpoint_completion
    assert "verify_training_checkpoints" in checkpoint_completion
    assert "checkpoint_owner_node_rank=0" in checkpoint_completion

    rank_entrypoint = RANK_ENTRYPOINT.read_text(encoding="utf-8")
    assert 'if global_rank != 0:\n        os.environ["WANDB_MODE"] = "disabled"' in rank_entrypoint


def test_rank_entrypoint_offsets_reset_replay_seed_by_global_rank() -> None:
    module = _load_rank_module()
    arguments = ["train.py", "env.commands.route.reset_replay.seed=47", "--distributed"]

    rank_zero_arguments, rank_zero_seed = module._rewrite_reset_replay_seed(arguments, 0)
    rank_31_arguments, rank_31_seed = module._rewrite_reset_replay_seed(arguments, 31)

    assert rank_zero_seed == 47
    assert rank_zero_arguments == arguments
    assert rank_31_seed == 78
    assert rank_31_arguments == ["train.py", "env.commands.route.reset_replay.seed=78", "--distributed"]
    assert arguments == ["train.py", "env.commands.route.reset_replay.seed=47", "--distributed"]

    unchanged, missing_seed = module._rewrite_reset_replay_seed(["train.py", "--distributed"], 31)
    assert unchanged == ["train.py", "--distributed"]
    assert missing_seed is None
    with pytest.raises(SystemExit, match="exactly one"):
        module._rewrite_reset_replay_seed(
            ["env.commands.route.reset_replay.seed=47", "env.commands.route.reset_replay.seed=48"], 1
        )
    with pytest.raises(SystemExit, match="must be an integer"):
        module._rewrite_reset_replay_seed(["env.commands.route.reset_replay.seed=invalid"], 1)


@pytest.mark.parametrize(("statuses", "expected_exit_code"), [([0, 0], 0), ([0, 9], 1)])
def test_multinode_barrier_aggregates_node_success_and_failure(
    tmp_path: Path, statuses: list[int], expected_exit_code: int
) -> None:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    output = tmp_path / "ddp-node-status.tsv"
    server = subprocess.Popen(
        [
            sys.executable,
            str(DDP_BARRIER),
            "serve",
            "--bind",
            "127.0.0.1",
            "--port",
            str(port),
            "--expected-nodes",
            str(len(statuses)),
            "--workflow-id",
            "test-workflow",
            "--startup-timeout-seconds",
            "5",
            "--output",
            str(output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for node_rank, status in enumerate(statuses):
            report = subprocess.run(
                [
                    sys.executable,
                    DDP_BARRIER,
                    "report",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--node-rank",
                    str(node_rank),
                    "--phase",
                    "complete",
                    "--status",
                    str(status),
                    "--workflow-id",
                    "test-workflow",
                    "--timeout-seconds",
                    "3",
                    "--retry-seconds",
                    "0.01",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            assert report.returncode == 0, report.stderr
        stdout, stderr = server.communicate(timeout=10)
    finally:
        if server.poll() is None:
            server.kill()
            server.wait()

    assert server.returncode == expected_exit_code, stdout + stderr
    assert output.read_text(encoding="utf-8") == "node_rank\texit_code\n" + "".join(
        f"{node_rank}\t{status}\n" for node_rank, status in enumerate(statuses)
    )
    if expected_exit_code == 0:
        assert "ddp_barrier_complete all_nodes_succeeded=true" in stdout
    else:
        assert "ddp_barrier_failed" in stdout


def test_multinode_barrier_fails_immediately_after_one_nonzero_terminal_status(tmp_path: Path) -> None:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    output = tmp_path / "ddp-node-status.tsv"
    server = subprocess.Popen(
        [
            sys.executable,
            str(DDP_BARRIER),
            "serve",
            "--bind",
            "127.0.0.1",
            "--port",
            str(port),
            "--expected-nodes",
            "2",
            "--workflow-id",
            "test-workflow",
            "--startup-timeout-seconds",
            "30",
            "--completion-grace-seconds",
            "30",
            "--output",
            str(output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        report = subprocess.run(
            [
                sys.executable,
                str(DDP_BARRIER),
                "report",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--node-rank",
                "1",
                "--phase",
                "complete",
                "--status",
                "9",
                "--workflow-id",
                "test-workflow",
                "--timeout-seconds",
                "3",
                "--retry-seconds",
                "0.01",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert report.returncode == 0, report.stdout + report.stderr
        try:
            stdout, stderr = server.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            server.kill()
            stdout, stderr = server.communicate(timeout=5)
            pytest.fail(f"Barrier did not fail immediately after a peer failure:\n{stdout}{stderr}")
    finally:
        if server.poll() is None:
            server.kill()
            server.wait()

    assert server.returncode == 1, stdout + stderr
    assert output.read_text(encoding="utf-8") == "node_rank\texit_code\n1\t9\n"
    assert "ddp_barrier_failed statuses={1: 9}" in stdout


def test_multinode_lead_terminates_long_runner_process_group_after_peer_failure(tmp_path: Path) -> None:
    expected_sha256 = hashlib.sha256(b"supervised source").hexdigest()
    served = tmp_path / "served"
    served.mkdir()
    runner_pid_file = tmp_path / "runner.pid"
    child_pid_file = tmp_path / "runner-child.pid"
    source_server_pid_file = tmp_path / "source-server.pid"

    def write_script(path: Path, contents: str) -> None:
        path.write_text(contents, encoding="utf-8")
        path.chmod(0o755)

    source_preparer = tmp_path / "prepare-source.sh"
    write_script(
        source_preparer,
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'printf \'%s\\n\' "$$" > "${HARNESS_SOURCE_SERVER_PID_FILE}"',
                'printf \'%s\\n\' "${OSMO_SOURCE_SHA256}" > "${HARNESS_SERVED_DIR}/source.sha256"',
                'exec "${HARNESS_REAL_PYTHON}" -m http.server "${OSMO_SOURCE_SERVE_PORT}" '
                '--bind 127.0.0.1 --directory "${HARNESS_SERVED_DIR}"',
            ]
        ),
    )
    runner = tmp_path / "runner.sh"
    write_script(
        runner,
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'printf \'%s\\n\' "$$" > "${HARNESS_RUNNER_PID_FILE}"',
                "sleep 300 &",
                'child_pid="$!"',
                'printf \'%s\\n\' "${child_pid}" > "${HARNESS_CHILD_PID_FILE}"',
                'wait "${child_pid}"',
            ]
        ),
    )

    wrapper = MULTINODE_RUNNER.read_text(encoding="utf-8")
    wrapper = wrapper.replace("runner=/tmp/run-yam-cable-routing-osmo.sh", f"runner={shlex.quote(str(runner))}")
    wrapper = wrapper.replace(
        "barrier=/tmp/yam-cable-routing-ddp-barrier.py", f"barrier={shlex.quote(str(DDP_BARRIER))}"
    )
    wrapper = wrapper.replace(
        "source_preparer=/tmp/prepare-yam-cable-routing-source.sh",
        f"source_preparer={shlex.quote(str(source_preparer))}",
    )
    wrapper_path = tmp_path / "multinode-wrapper.sh"
    wrapper_path.write_text(wrapper, encoding="utf-8")

    ports: list[int] = []
    while len(ports) < 3:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            candidate = reservation.getsockname()[1]
        if candidate not in ports:
            ports.append(candidate)
    master_port, completion_port, source_port = ports
    environment = {
        **os.environ,
        "HARNESS_CHILD_PID_FILE": str(child_pid_file),
        "HARNESS_REAL_PYTHON": sys.executable,
        "HARNESS_RUNNER_PID_FILE": str(runner_pid_file),
        "HARNESS_SERVED_DIR": str(served),
        "HARNESS_SOURCE_SERVER_PID_FILE": str(source_server_pid_file),
        "OSMO_BARRIER_STARTUP_TIMEOUT_SECONDS": "311",
        "OSMO_COMPLETION_PORT": str(completion_port),
        "OSMO_IMAGE_PYTHON": sys.executable,
        "OSMO_MASTER_ADDR": "127.0.0.1",
        "OSMO_MASTER_PORT": str(master_port),
        "OSMO_NODE_RANK": "0",
        "OSMO_NUM_GPU": "1",
        "OSMO_NUM_NODES": "2",
        "OSMO_OUTPUT_DIR": str(tmp_path / "output"),
        "OSMO_RUN_MODE": "multinode",
        "OSMO_SOURCE_FETCH_WAIT_SECONDS": "310",
        "OSMO_SOURCE_SERVE_PORT": str(source_port),
        "OSMO_SOURCE_SHA256": expected_sha256,
        "OSMO_SOURCE_STARTUP_WAIT_SECONDS": "10",
        "OSMO_TOTAL_GPU": "2",
        "OSMO_VCS_REF": "0" * 40,
        "OSMO_WORKFLOW_ID": "functional-supervision-test",
    }
    process = subprocess.Popen(
        ["bash", str(wrapper_path)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def wait_for_file(path: Path, description: str, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                return
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout is not None else ""
                pytest.fail(f"Wrapper exited before {description} (status {process.returncode}):\n{output}")
            time.sleep(0.01)
        pytest.fail(f"Timed out waiting for {description}.")

    runner_pid = 0
    child_pid = 0
    source_server_pid = 0
    try:
        wait_for_file(runner_pid_file, "the supervised runner")
        wait_for_file(child_pid_file, "the runner child")
        wait_for_file(source_server_pid_file, "the source server")
        runner_pid = int(runner_pid_file.read_text(encoding="utf-8").strip())
        child_pid = int(child_pid_file.read_text(encoding="utf-8").strip())
        source_server_pid = int(source_server_pid_file.read_text(encoding="utf-8").strip())

        report = subprocess.run(
            [
                sys.executable,
                str(DDP_BARRIER),
                "report",
                "--host",
                "127.0.0.1",
                "--port",
                str(completion_port),
                "--node-rank",
                "1",
                "--phase",
                "complete",
                "--status",
                "9",
                "--workflow-id",
                "functional-supervision-test",
                "--timeout-seconds",
                "3",
                "--retry-seconds",
                "0.01",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert report.returncode == 0, report.stdout + report.stderr
        try:
            stdout, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pytest.fail("Lead wrapper did not terminate its runner after the peer failure.")
        assert process.returncode == 17, stdout

        termination_deadline = time.monotonic() + 5
        lingering_pids = (runner_pid, child_pid, source_server_pid)
        while time.monotonic() < termination_deadline:
            for pid in lingering_pids:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    continue
                break
            else:
                break
            time.sleep(0.01)
        for pid in lingering_pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if runner_pid:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(runner_pid, signal.SIGKILL)
        for pid in (runner_pid, child_pid, source_server_pid):
            if pid:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)


def test_dry_run_validation_is_not_blocked_by_transient_pool_capacity() -> None:
    submitter = SUBMITTER.read_text(encoding="utf-8")

    capacity_gate = submitter.split("if (( capacity_free < num_gpu )); then", 1)[1].split(
        'if [[ ! "${pool_platform}"', 1
    )[0]
    assert "if (( effective_free < 0 )); then" in submitter
    assert "effective_free=0" in submitter
    assert 'if [[ "${priority}" == LOW ]]; then' in submitter
    assert 'capacity_free="${total_free}"' in submitter
    assert "capacity_kind=physical" in submitter
    assert "if (( submit != 0 )); then" in capacity_gate
    assert "validating ${num_gpu} anyway" in capacity_gate
    assert "the workflow may be preempted" in submitter
    assert '"platform=${pool_platform}"' in submitter
    assert "ovx-l40s|ovx-l40|dgx-h100" in submitter
    assert 'num_workers="${num_gpu}"' in submitter
    assert "gpus_per_worker=1" in submitter
    assert "supports at most 32 GPUs" in submitter
    assert "Single-node DDP supports at most eight GPUs" in submitter


@pytest.mark.parametrize("script", [PACKAGER, PREPARER, RUNNER, MULTINODE_RUNNER, SUBMITTER])
def test_shell_scripts_parse(script: Path) -> None:
    subprocess.run(["bash", "-n", script], check=True)


@pytest.mark.parametrize("script", [DDP_BARRIER, RANK_ENTRYPOINT, SOURCE_FETCHER, PREFLIGHT])
def test_python_scripts_compile(script: Path) -> None:
    subprocess.run(["python3", "-m", "py_compile", script], check=True)


def test_rank_entrypoint_isolates_scheduler_device_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_rank_module()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c")

    assert module._select_physical_device(1) == "GPU-b"
    with pytest.raises(SystemExit, match="exceeds CUDA_VISIBLE_DEVICES"):
        module._select_physical_device(3)


def test_packager_captures_dirty_files_ignores_logs_and_publishes_checksum(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / ".gitignore").write_text("logs/\n**/*.usd\n**/*.usda\n**/*.usdc\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0.1'\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore", "pyproject.toml", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=OSMO Test",
            "-c",
            "user.email=osmo@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    yam_assets = repo / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "contrib" / "cable_routing" / "assets" / "yam"
    (yam_assets / "payloads").mkdir(parents=True)
    (yam_assets / "yam.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (yam_assets / "payloads" / "geometries.usd").write_bytes(b"ignored only outside the asset allowlist")
    (yam_assets / "payloads" / "geometry_library.usdc").write_bytes(b"binary USD asset")
    manipulationnet_assets = yam_assets.parent / "manipulationnet"
    manipulationnet_assets.mkdir()
    (manipulationnet_assets / "board.usdc").write_bytes(b"derived board USD asset")
    (manipulationnet_assets / "round_peg.usdc").write_bytes(b"derived peg USD asset")
    (repo / "unrelated.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (repo / "logs").mkdir()
    (repo / "logs" / "model.pt").write_bytes(b"ignored")
    fixture_packager = repo / "docker" / "cluster" / PACKAGER.name
    fixture_packager.parent.mkdir(parents=True)
    shutil.copy2(PACKAGER, fixture_packager)
    sync_root = tmp_path / "sync"

    # Invoke outside the fixture repository to prove source-root discovery is
    # anchored to the helper itself rather than the caller's working directory.
    subprocess.run(["bash", fixture_packager, sync_root], cwd=tmp_path, check=True)

    archive = sync_root / "source.tar.gz"
    expected_sha = (sync_root / "source.sha256").read_text(encoding="utf-8").strip()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected_sha
    with tarfile.open(archive, "r:gz") as source_tar:
        names = set(source_tar.getnames())
    assert {"pyproject.toml", "tracked.txt", "dirty.txt"} <= names
    assert {
        "source/isaaclab_tasks/isaaclab_tasks/contrib/cable_routing/assets/yam/yam.usda",
        "source/isaaclab_tasks/isaaclab_tasks/contrib/cable_routing/assets/yam/payloads/geometries.usd",
        "source/isaaclab_tasks/isaaclab_tasks/contrib/cable_routing/assets/yam/payloads/geometry_library.usdc",
        "source/isaaclab_tasks/isaaclab_tasks/contrib/cable_routing/assets/manipulationnet/board.usdc",
        "source/isaaclab_tasks/isaaclab_tasks/contrib/cable_routing/assets/manipulationnet/round_peg.usdc",
    } <= names
    assert "unrelated.usda" not in names
    assert "logs/model.pt" not in names
    metadata = (sync_root / "source.metadata").read_text(encoding="utf-8")
    assert f"source_sha256={expected_sha}" in metadata
    assert "dirty_file_count=2" in metadata
    status_sha = hashlib.sha256((sync_root / "git-status.txt").read_bytes()).hexdigest()
    assert metadata.endswith(f"git_status_sha256={status_sha}\n")


def test_packager_rejects_untracked_secret_names(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0.1'\n", encoding="utf-8")
    (repo / ".env").write_text("SECRET=do-not-package\n", encoding="utf-8")
    fixture_packager = repo / "docker" / "cluster" / PACKAGER.name
    fixture_packager.parent.mkdir(parents=True)
    shutil.copy2(PACKAGER, fixture_packager)

    result = subprocess.run(
        ["bash", fixture_packager, tmp_path / "sync"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "LC_ALL": "C"},
    )

    assert result.returncode == 4
    assert "Refusing to package possible secret file: .env" in result.stderr


def test_submitter_bounds_rsync_without_forwarding_unsupported_timeout_options() -> None:
    submitter = SUBMITTER.read_text(encoding="utf-8")

    assert "rsync_transfer_options=(--archive)" in submitter
    assert 'timeout 300 "${system_rsync}"' in submitter
    assert 'timeout 60 "${system_rsync}"' in submitter
    assert 'timeout 10 "${system_rsync}"' in submitter
    assert "--contimeout=" not in submitter
    assert "--timeout=120" not in submitter


@pytest.mark.parametrize(
    ("run_mode", "source_task", "priority", "quota_free"),
    [
        ("smoke", "source-prep", "NORMAL", 32),
        ("multinode", "trainer-node-0", "NORMAL", 32),
        ("multinode", "trainer-node-0", "LOW", 16),
    ],
)
def test_submitter_uses_verified_system_rsync_tunnel(
    tmp_path: Path, run_mode: str, source_task: str, priority: str, quota_free: int
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    osmo_log = tmp_path / "osmo-commands.log"
    system_rsync_log = tmp_path / "system-rsync-commands.log"
    fake_osmo = fake_bin / "osmo"
    fake_osmo.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\t' "$@" >> "${FAKE_OSMO_LOG}"
printf '\\n' >> "${FAKE_OSMO_LOG}"
case "$1 ${2:-}" in
    "version ")
        echo "OSMO client version: test"
        ;;
    "profile list")
        echo "test-profile"
        ;;
    "credential list")
        printf 'wandb GENERIC\\n'
        ;;
    "pool list")
        printf '%s%s%s\\n' \
            '{"node_sets":[{"pools":[{"status":"ONLINE","default_platform":"ovx-l40s","resource_usage":{"quota_free":' \
            "${FAKE_QUOTA_FREE:-32}" \
            ',"total_free":32}}]}]}'
        ;;
    "workflow validate")
        echo "valid"
        ;;
    "workflow submit")
        if printf '%s\\n' "$@" | grep -Fxq -- '--dry-run'; then
            echo "rendered: true"
        else
            printf '%s\\n' '{"name":"fake-yam-workflow-1"}'
        fi
        ;;
    "workflow query")
        if [[ -n "${FAKE_OSMO_QUERY_STATE:-}" && ! -e "${FAKE_OSMO_QUERY_STATE}" ]]; then
            touch "${FAKE_OSMO_QUERY_STATE}"
            exit 10
        fi
        printf '{"tasks":[{"name":"%s","status":"RUNNING"}]}\\n' "${FAKE_SOURCE_TASK}"
        ;;
    "workflow rsync")
        if [[ "$3" == upload ]]; then
            printf '%s\n' "$@" | grep -Fxq -- '--daemon'
        elif [[ "$3" == stop ]]; then
            :
        else
            echo "Unexpected fake rsync command: $*" >&2
            exit 98
        fi
        ;;
    *)
        echo "Unexpected fake OSMO command: $*" >&2
        exit 99
        ;;
esac
""",
        encoding="utf-8",
    )
    fake_osmo.chmod(0o755)
    fake_system_rsync = fake_bin / "system-rsync"
    fake_system_rsync.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\t' "$@" >> "${FAKE_SYSTEM_RSYNC_LOG}"
printf '\\n' >> "${FAKE_SYSTEM_RSYNC_LOG}"
previous=""
current=""
for argument in "$@"; do
    previous="${current}"
    current="${argument}"
done
if [[ "${current}" == "rsync://127.0.0.1:16000/" ]]; then
    printf 'osmo\\tOSMO task workspace\\n'
elif [[ "${previous}" == "rsync://127.0.0.1:16000/osmo/yam-source-accepted.sha256" ]]; then
    cp "${FAKE_SYNC_ROOT}/source.sha256" "${current}/yam-source-accepted.sha256"
elif [[ "${current}" == "rsync://127.0.0.1:16000/osmo/source-sync/" ]]; then
    :
elif [[ "${current}" == "rsync://127.0.0.1:16000/osmo/" ]]; then
    :
else
    echo "Unexpected fake system rsync command: $*" >&2
    exit 97
fi
""",
        encoding="utf-8",
    )
    fake_system_rsync.chmod(0o755)
    sync_root = tmp_path / "sync"

    result = subprocess.run(
        [
            "bash",
            SUBMITTER,
            run_mode,
            "--submit",
            "--priority",
            priority,
            "--sync-root",
            sync_root,
            "--source-upload-wait-seconds",
            "5",
            "--source-upload-retry-seconds",
            "5",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "FAKE_OSMO_LOG": str(osmo_log),
            "FAKE_QUOTA_FREE": str(quota_free),
            "FAKE_SYSTEM_RSYNC_LOG": str(system_rsync_log),
            "FAKE_SYNC_ROOT": str(sync_root),
            "FAKE_SOURCE_TASK": source_task,
            "FAKE_OSMO_QUERY_STATE": str(tmp_path / "query-failed-once"),
            "OSMO_RSYNC_PORT_OVERRIDE": "16000",
            "OSMO_SYSTEM_RSYNC_BIN": str(fake_system_rsync),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
    )

    assert result.returncode == 0, result.stderr
    commands = [line.rstrip("\t").split("\t") for line in osmo_log.read_text(encoding="utf-8").splitlines()]
    submissions = [command for command in commands if command[:2] == ["workflow", "submit"]]
    assert len(submissions) == 2
    assert submissions[-1][submissions[-1].index("--priority") + 1] == priority
    assert [command[:2] for command in commands].count(["workflow", "query"]) >= 2
    assert "--rsync" not in submissions[-1]
    uploads = [command for command in commands if command[:3] == ["workflow", "rsync", "upload"]]
    assert len(uploads) == 1
    assert uploads[0][3:5] == ["fake-yam-workflow-1", source_task]
    assert uploads[0][5].startswith("/tmp/yam-osmo-rsync-hold.")
    assert uploads[0][5].endswith(":/osmo/run/workspace/.yam-rsync-tunnel")
    assert "--daemon" in uploads[0]
    assert [command[:3] for command in commands].count(["workflow", "rsync", "stop"]) == 1

    rsync_commands = [
        line.rstrip("\t").split("\t") for line in system_rsync_log.read_text(encoding="utf-8").splitlines()
    ]
    remote_source = "rsync://127.0.0.1:16000/osmo/source-sync/"
    remote_writes = [command for command in rsync_commands if command[-1] == remote_source]
    assert len(remote_writes) == 2
    assert [str(sync_root / name) for name in ("source.tar.gz", "source.metadata", "git-status.txt")] == [
        argument for argument in remote_writes[0] if argument.startswith(str(sync_root))
    ]
    assert remote_writes[1][-2] == str(sync_root / "source.sha256")
    acceptance_downloads = [
        command
        for command in rsync_commands
        if command[-2] == "rsync://127.0.0.1:16000/osmo/yam-source-accepted.sha256"
    ]
    assert len(acceptance_downloads) == 1
    assert Path(acceptance_downloads[0][-1]).name.startswith("yam-osmo-marker-verify.")
    release_uploads = [command for command in rsync_commands if command[-1] == "rsync://127.0.0.1:16000/osmo/"]
    assert len(release_uploads) == 1
    assert release_uploads[0][-2].endswith("/yam-source-release.sha256")
    assert "source_upload=complete workflow=fake-yam-workflow-1" in result.stdout
