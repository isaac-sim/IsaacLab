# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""TorchScript I/O utilities."""

import os

import torch


def load_torchscript_model(model_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load a TorchScript model from the specified path.

    This function only loads TorchScript models (.pt or .pth files created with torch.jit.save).
    It will not work with raw PyTorch checkpoints (.pth files created with torch.save).

    Args:
        model_path (str): Path to the TorchScript model file (.pt or .pth)
        device (str, optional): Device to load the model on. Defaults to 'cpu'.

    Returns:
        torch.nn.Module: The loaded TorchScript model in evaluation mode

    Raises:
        FileNotFoundError: If the model file does not exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TorchScript model file not found: {model_path}")

    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        print(f"Successfully loaded TorchScript model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading TorchScript model: {e}")
        return None
