# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX I/O utilities."""

import io

import torch

from ..assets import read_file


def load_onnx_model(
    model_path: str,
    sample_inputs: tuple,
    *,
    device: str,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> io.BytesIO:
    """Load a network file as an ONNX model.

    An ``.onnx`` file is read as it is. Any other file is loaded as a TorchScript module and exported to ONNX in
    memory, with the TorchScript exporter since the dynamo exporter does not accept TorchScript modules.

    Args:
        model_path: Path to the network file, local or on the Nucleus server.
        sample_inputs: Inputs that trace the TorchScript module for the export, on ``device``.
        device: Device to load the TorchScript module on.
        input_names: Names of the exported model's inputs.
        output_names: Names of the exported model's outputs.
        dynamic_axes: Dynamic axes of the exported model's inputs and outputs, by name. Defaults to None.

    Returns:
        The ONNX model. An ``.onnx`` file keeps its own input and output names.
    """
    file_bytes = read_file(model_path)
    if model_path.endswith(".onnx"):
        return file_bytes
    module = torch.jit.load(file_bytes, map_location=device).eval()
    model = io.BytesIO()
    torch.onnx.export(
        module,
        sample_inputs,
        model,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    model.seek(0)
    return model
