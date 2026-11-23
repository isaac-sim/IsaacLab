# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral regressions for deprecated Ray entry points (requires the ``ray`` extra)."""

import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")
mlflow = pytest.importorskip("mlflow")

from ray import job_submission
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from isaaclab_contrib.rl.ray import util

pytestmark = [pytest.mark.unit, pytest.mark.kitless]

ROOT = Path(__file__).resolve().parents[4]
LEGACY = ROOT / "scripts/reinforcement_learning/ray"


def test_legacy_mlflow_converter_preserves_worker_pickling(tmp_path):
    """The legacy CLI must convert metrics in its real process pool."""
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment("compat", artifact_location=(tmp_path / "artifacts").as_uri())
    run = client.create_run(experiment)
    client.log_metric(run.info.run_id, "reward", 12.5, step=3)
    client.set_terminated(run.info.run_id)
    output = tmp_path / "converted"

    result = subprocess.run(
        [
            sys.executable,
            str(LEGACY / "mlflow_to_local_tensorboard.py"),
            "--uri",
            uri,
            "--experiment-name",
            "compat",
            "--download-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "tensorboard" / f"run_{run.info.run_id}").exists(), result.stdout + result.stderr
    events = EventAccumulator(str(output / "tensorboard" / f"run_{run.info.run_id}")).Reload()
    assert [(event.step, event.value) for event in events.Scalars("metrics/reward")] == [(3, 12.5)]


def test_legacy_submit_uploads_files_beside_script(tmp_path, monkeypatch):
    """A legacy invocation must keep uploading its adjacent user configuration and modules."""
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for name in ("submit_job.py", "_compat.py"):
        shutil.copy(LEGACY / name, legacy / name)
    (legacy / "tasks.yaml").write_text("tasks: []\n")
    (legacy / "custom_module.py").write_text("value = 42\n")
    cluster_spec = tmp_path / "clusters"
    cluster_spec.write_text("name: local address: http://127.0.0.1:8265\n")
    submitted = []
    status = job_submission.JobStatus.SUCCEEDED

    class Client:
        def __init__(self, address):
            pass

        def submit_job(self, *, entrypoint, runtime_env):
            submitted.append(runtime_env)
            return "job"

        def get_job_status(self, job_id):
            return status

        def get_job_logs(self, job_id):
            return "completed"

    monkeypatch.setattr(job_submission, "JobSubmissionClient", Client)
    monkeypatch.syspath_prepend(str(legacy))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(legacy / "submit_job.py"),
            "--config_file",
            str(cluster_spec),
            "--aggregate_jobs",
            "task_runner.py",
            "--task_cfg",
            "tasks.yaml",
        ],
    )
    with pytest.warns(FutureWarning):
        runpy.run_path(str(legacy / "submit_job.py"), run_name="__main__")
    assert len(submitted) == 1
    assert Path(submitted[0]["working_dir"]) == legacy
    status = job_submission.JobStatus.FAILED
    with pytest.warns(FutureWarning), pytest.raises(RuntimeError, match="completed"):
        runpy.run_path(str(legacy / "submit_job.py"), run_name="__main__")


def test_failed_worker_command_is_reported(monkeypatch):
    """A real failed subprocess must fail its Ray job rather than report completion."""
    monkeypatch.setattr(util.ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "local"))
    with pytest.raises(subprocess.CalledProcessError) as error:
        util.execute_job(f"{sys.executable} -c \"import sys; sys.stderr.write('worker failed'); sys.exit(7)\"")
    assert error.value.returncode == 7
    assert "worker failed" in error.value.stderr


def test_resource_wrapper_honors_manual_cpu_only_allocation(monkeypatch):
    """Explicit resource options must override node totals without requiring --num_workers."""
    monkeypatch.setattr(util, "ray_init", lambda **kwargs: None)
    monkeypatch.setattr(
        util,
        "get_gpu_node_resources",
        lambda **kwargs: [{"CPU": 8, "GPU": 2, "ram_gb": 16, "id": "node"}],
    )
    submitted = []
    monkeypatch.setattr(util, "submit_wrapped_jobs", lambda jobs, **kwargs: submitted.extend(jobs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wrap_resources",
            "--cpu_per_worker",
            "2",
            "--gpu_per_worker",
            "0",
            "--ram_gb_per_worker",
            "1",
            "--sub_jobs",
            "python train.py",
        ],
    )
    runpy.run_module("isaaclab_contrib.rl.ray.wrap_resources", run_name="__main__")
    assert len(submitted) == 1
    assert submitted[0].resources.to_opt() == {"num_cpus": 2, "num_gpus": 0, "memory": 1024**3}
