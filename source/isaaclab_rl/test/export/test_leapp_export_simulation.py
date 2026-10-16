# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for LEAPP export commands."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from leapp_direct_env import DIRECT_TASK
from leapp_initialized_checkpoints import resolved_path_file, task_checkpoint_dir
from leapp_test_tasks import MANAGER_TASK

from isaaclab.test.utils import resolve_test_sim_device

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CHECKPOINT_SCRIPT = Path(__file__).with_name("leapp_initialized_checkpoints.py")
_OUTPUT_TAIL_SIZE = 5000


def _run_checked(command: list[str], timeout: int = 600) -> str:
    """Run a command and return its combined output."""
    try:
        result = subprocess.run(
            command,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        message = "\n".join(
            [
                f"Command timed out after {timeout}s: {command}",
                stdout[-_OUTPUT_TAIL_SIZE:],
                stderr[-_OUTPUT_TAIL_SIZE:],
            ]
        )
        pytest.fail(message)
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"{command} exited with {result.returncode}:\n{output[-_OUTPUT_TAIL_SIZE:]}"
    assert "Traceback (most recent call last):" not in output, output[-_OUTPUT_TAIL_SIZE:]
    return output


def _run_export(backend: str, task: str, checkpoint: str, tmp_path: Path) -> Path:
    if task == DIRECT_TASK:
        command = [
            sys.executable,
            str(Path(__file__).with_name("leapp_direct_env.py")),
            "--device",
            resolve_test_sim_device(),
        ]
    else:
        command = [sys.executable, str(Path(__file__).with_name("leapp_test_tasks.py")), backend]
    command.extend(
        [
            "--task",
            task,
            "--checkpoint",
            checkpoint,
            "--export_save_path",
            str(tmp_path / "export"),
            "--disable_graph_visualization",
        ]
    )
    if backend == "sb3" and task == MANAGER_TASK:
        command.append("env.observations.policy.concatenate_terms=False")
    output = _run_checked(command)
    if checkpoint == "pretrained" and "pre-trained checkpoint is currently unavailable" in output:
        pytest.skip(f"No published checkpoint for {task}")
    artifact_dir = tmp_path / "export" / task
    for name in (f"{task}.onnx", f"{task}.yaml", "log.txt"):
        assert (artifact_dir / name).is_file(), f"Missing {name}:\n{output[-_OUTPUT_TAIL_SIZE:]}"
    return artifact_dir / f"{task}.yaml"


def _initialized_export(backend: str, task: str, tmp_path: Path) -> Path:
    command = [
        sys.executable,
        str(_CHECKPOINT_SCRIPT),
        "--checkpoint_root",
        str(tmp_path),
        "--spec",
        backend,
        task,
    ]
    _run_checked(command, timeout=1200)
    checkpoint = resolved_path_file(task_checkpoint_dir(tmp_path, backend, task)).read_text(encoding="utf-8").strip()
    assert Path(checkpoint).is_file()
    return _run_export(backend, task, checkpoint, tmp_path)


def _assert_live_inputs(pipeline: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    """Replay the artifact with independently varied inputs and verify deterministic reset."""
    from leapp import InferenceManager

    description = yaml.safe_load(pipeline.read_text())
    manager = InferenceManager(str(pipeline))

    def infer(values):
        # InferenceManager.reset() clears borrowed input/output tensors in place.
        outputs = manager.run_policy({name: value.clone() for name, value in values.items()})
        return {name: value.clone() for name, value in outputs.items()}

    inputs = {}
    for node_name, names in description["pipeline"]["inputs"].items():
        for desc in description["models"][node_name]["inputs"]:
            if desc["name"] not in names:
                continue
            value = torch.zeros(desc["shape"], dtype=getattr(torch, desc["dtype"]))
            if desc.get("kind") == "state/body/rotation":
                value[..., 3] = 1.0
            if desc.get("isaaclab_connection") == "state:camera:intrinsic_matrices":
                value.copy_(torch.eye(3))
            inputs[f"{node_name}/{desc['name']}"] = value

    manager.reset()
    baseline = infer(inputs)
    action_outputs = baseline.keys() - description["pipeline"]["feedback_flow"].keys()
    for key, value in inputs.items():
        changed = value.clone()
        changed.reshape(-1)[::2] += 0.25 if value.is_floating_point() else (127 if value.dtype == torch.uint8 else 1)
        manager.reset()
        outputs = infer({**inputs, key: changed})
        assert all(torch.isfinite(output).all() for output in outputs.values()), key
        assert any(not torch.allclose(outputs[name], baseline[name], atol=1e-7) for name in action_outputs), (
            f"Exported input {key} has no effect on any action output"
        )

    # Repeated calls exercise feedback; reset must reproduce the whole sequence.
    sequences = []
    for _ in range(2):
        manager.reset()
        sequences.append([infer(inputs) for _ in range(4)])
    for first, second in zip(*sequences):
        for name in first:
            torch.testing.assert_close(first[name], second[name])
    depth_key = next(key for key in inputs if key.endswith("camera_output_depth"))
    invalid_depth = inputs[depth_key].clone()
    invalid_depth.reshape(-1)[:4] = torch.tensor([float("nan"), float("inf"), -1.0, 3.0])
    manager.reset()
    assert all(torch.isfinite(value).all() for value in infer({**inputs, depth_key: invalid_depth}).values())
    return description, baseline


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["rsl_rl", "rl_games", "skrl", "sb3"])
def test_manager_export(backend: str, tmp_path: Path) -> None:
    """Each framework exports live scene/camera inputs, action metadata, and feedback."""
    pipeline = _initialized_export(backend, MANAGER_TASK, tmp_path)
    description, baseline = _assert_live_inputs(pipeline)
    model = description["models"][MANAGER_TASK]
    inputs = {desc.get("isaaclab_connection"): desc for desc in model["inputs"]}
    assert {
        "state:camera:output.rgb",
        "state:camera:output.depth",
        "state:camera:output.distance_to_camera",
        "state:camera:output.normals",
        "state:camera:output.semantic_segmentation",
        "state:camera:output.motion_vectors",
        "state:collection:object_state_w",
        "state:relative:joint_pos",
        "state:object:root_pos_w",
        "state:object:root_quat_w",
        "state:height_scanner:ray_hits_w",
        "state:contact:net_normal_forces_w_history",
        "state:contact:force_matrix_w",
        "state:contact:friction_force_matrix_w",
        "state:contact:contact_pos_w",
        "state:contact:current_air_time",
        "state:deformable:nodal_pos_w",
        "state:deformable:nodal_vel_w",
        "state:deformable:nodal_state_w",
        "state:deformable:root_pos_w",
        "state:deformable:root_vel_w",
        "state:imu:ang_vel_b",
        "state:imu:lin_acc_b",
        "state:pva:pose_w",
        "state:pva:ang_acc_b",
        "state:frames:target_pose_source",
        "state:frames:target_pose_w",
        "state:frames:source_pose_w",
        "state:camera:pos_w",
        "state:camera:intrinsic_matrices",
        "state:camera:quat_w_world",
        "state:camera:quat_w_ros",
        "state:camera:quat_w_opengl",
        "state:wrench:force",
        "state:wrench:torque",
        "state:cable:segment_pose_w",
        "state:cable:segment_velocity_w",
        "state:robot:body_joint_wrench",
        "state:robot:body_link_jacobian_w",
        "state:robot:body_com_jacobian_w",
        "state:robot:mass_matrix",
        "state:robot:gravity_compensation_forces",
        "command:velocity",
        "command:pose",
    } <= inputs.keys()
    assert "state:relative:joint_vel" not in inputs  # Only the critic reads this property.
    assert inputs["state:camera:output.rgb"]["shape"] == [1, 4, 4, 3]
    assert inputs["state:camera:output.rgb"]["dtype"] == "uint8"
    feedback = description["pipeline"]["feedback_flow"]
    if backend == "rl_games":
        assert f"{MANAGER_TASK}/actor_state_0_out" in feedback
        assert f"{MANAGER_TASK}/actor_state_1_out" in feedback
    if backend == "rsl_rl":
        import onnx

        graph = onnx.load(pipeline.with_suffix(".onnx")).graph
        assert any(node.op_type == "Conv" for node in graph.node)
    for state in (
        "last_action",
        "last_action_position",
        "last_action_relative",
        "h_policy_joint_pos",
        "frames_policy_rgb",
        "ema_previous_applied_actions",
    ):
        assert f"{MANAGER_TASK}/{state}_out" in feedback
    outputs = {desc["name"]: desc for desc in model["outputs"]}
    for name, entity, joint in (
        ("relative", "relative", "slider_to_cart"),
        ("binary", "gripper", "cart_to_pole"),
        ("ema", "gripper", "slider_to_cart"),
    ):
        assert outputs[name]["isaaclab_connection"] == f"write:{entity}:set_joint_position_target_index"
        assert outputs[name]["element_names"] == [[joint]]
    assert outputs["position"]["element_names"] == [["cart_to_pole", "slider_to_cart"]]
    assert outputs["position"]["isaaclab_connection"] == "write:robot:set_joint_position_target_index"
    assert outputs["effort"]["isaaclab_connection"] == "write:relative:set_joint_effort_target_index"
    assert outputs["velocity_out"]["isaaclab_connection"] == "write:robot:set_joint_velocity_target_index"
    assert "command" in outputs
    assert "isaaclab_connection" not in outputs["command"]
    assert "command_kp_gains" not in outputs and "command_kd_gains" not in outputs
    # Configured implicit gains and the requested joint order must survive export.
    torch.testing.assert_close(baseline[f"{MANAGER_TASK}/position_kp_gains"], torch.tensor([[0.0, 20.0]]))
    torch.testing.assert_close(baseline[f"{MANAGER_TASK}/velocity_kd_gains"], torch.tensor([[3.0]]))


@pytest.mark.integration
def test_direct_export(tmp_path: Path) -> None:
    """Export the ANYmal direct tutorial with its published RSL-RL checkpoint."""
    _run_export("rsl_rl", DIRECT_TASK, "pretrained", tmp_path)


@pytest.mark.integration
def test_ur10_reach_export(tmp_path: Path) -> None:
    """Smoke-test the maintained UR10 reach task with RSL-RL and its default preset."""
    _initialized_export("rsl_rl", "Isaac-Reach-UR10", tmp_path)
