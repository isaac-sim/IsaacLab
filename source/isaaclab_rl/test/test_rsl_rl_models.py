# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the RSL-RL neural models customized for Isaac Lab.

These tests run on CPU and do not require Isaac Sim. They cover the image-only observation
support of :class:`~isaaclab_rl.rsl_rl.models.CNNModel` for training (``get_latent``) and for
deployment (``as_jit`` / ``as_onnx`` export wrappers), as well as parity of the exported
models with the original model.
"""

import io

import pytest
import torch

pytest.importorskip("rsl_rl", reason="rsl-rl-lib is not installed")
pytest.importorskip("tensordict", reason="tensordict is not installed")

from tensordict import TensorDict  # noqa: E402

from isaaclab_rl.rsl_rl.models import CNNModel  # noqa: E402

CNN_CFG = {"output_channels": [16, 32], "kernel_size": [8, 4], "stride": [4, 2], "activation": "relu"}
DIST_CFG = {"class_name": "GaussianDistribution", "init_std": 1.0}


def _make_image_only_model() -> tuple[CNNModel, TensorDict]:
    """Create a CNN model with a single image observation group (e.g. cartpole camera task)."""
    obs = TensorDict({"policy": torch.rand(1, 3, 64, 64)}, batch_size=[1])
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    model = CNNModel(
        obs,
        obs_groups,
        "actor",
        2,
        hidden_dims=[64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=dict(DIST_CFG),
        cnn_cfg=dict(CNN_CFG),
    )
    model.eval()
    return model, obs


def _make_mixed_model() -> tuple[CNNModel, TensorDict]:
    """Create a CNN model with image and proprioceptive observation groups."""
    obs = TensorDict({"proprio": torch.randn(1, 12), "camera": torch.rand(1, 3, 64, 64)}, batch_size=[1])
    obs_groups = {"actor": ["proprio", "camera"], "critic": ["proprio", "camera"]}
    model = CNNModel(
        obs,
        obs_groups,
        "actor",
        2,
        hidden_dims=[64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=dict(DIST_CFG),
        cnn_cfg=dict(CNN_CFG),
    )
    model.eval()
    return model, obs


def _script_roundtrip(jit_model: torch.nn.Module) -> torch.jit.ScriptModule:
    """Script, save and reload a module the way the runner export + deployment does."""
    scripted = torch.jit.script(jit_model)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    loaded = torch.jit.load(buffer)
    loaded.eval()
    return loaded


def test_cnn_model_image_only_forward():
    """Image-only models compute a latent purely from the CNN encoders."""
    model, obs = _make_image_only_model()
    with torch.inference_mode():
        out = model(obs)
    assert out.shape == (1, 2)


def test_cnn_model_image_only_jit_export_takes_only_images():
    """The JIT export of an image-only model takes only the 2D observations as input."""
    model, obs = _make_image_only_model()
    loaded = _script_roundtrip(model.as_jit())
    with torch.inference_mode():
        out = loaded([obs["policy"]])
        ref = model(obs)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


def test_cnn_model_image_only_onnx_export_takes_only_images(tmp_path):
    """The ONNX export of an image-only model declares only the 2D observations as inputs."""
    onnx = pytest.importorskip("onnx", reason="onnx is not installed")

    model, obs = _make_image_only_model()
    onnx_model = model.as_onnx(verbose=False)
    onnx_model.eval()
    assert onnx_model.input_names == ["policy"]

    path = str(tmp_path / "policy.onnx")
    torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),
        path,
        export_params=True,
        opset_version=18,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
    )
    graph_inputs = [graph_input.name for graph_input in onnx.load(path).graph.input]
    assert graph_inputs == ["policy"]

    ort = pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")
    session = ort.InferenceSession(path)
    out = torch.from_numpy(session.run(None, {"policy": obs["policy"].numpy()})[0])
    with torch.inference_mode():
        ref = model(obs)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_cnn_model_mixed_obs_jit_export_parity():
    """Models with 1D and 2D observation groups keep the upstream export interface."""
    model, obs = _make_mixed_model()
    loaded = _script_roundtrip(model.as_jit())
    with torch.inference_mode():
        out = loaded(obs["proprio"], [obs["camera"]])
        ref = model(obs)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


def test_cnn_model_mixed_obs_onnx_export_parity(tmp_path):
    """Models with 1D and 2D observation groups keep the upstream ONNX interface."""
    pytest.importorskip("onnx", reason="onnx is not installed")
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")

    model, obs = _make_mixed_model()
    onnx_model = model.as_onnx(verbose=False)
    onnx_model.eval()
    assert onnx_model.input_names == ["obs", "camera"]

    path = str(tmp_path / "policy.onnx")
    torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),
        path,
        export_params=True,
        opset_version=18,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
    )
    session = ort.InferenceSession(path)
    out = torch.from_numpy(session.run(None, {"obs": obs["proprio"].numpy(), "camera": obs["camera"].numpy()})[0])
    with torch.inference_mode():
        ref = model(obs)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
