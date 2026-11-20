# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos3 world model behind the Cosmos service: action chunks in, generated frames out.

The service runs in the Cosmos Framework environment, so Isaac Lab does not install the Framework. A client opens one
session for its environments, resets environments from start images and prompts, and steps every environment with a
chunk of normalized ``unified_v1`` controls.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import socket
import sys
import tempfile
import threading
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .._protocol import ProtocolError, receive_message, send_message
from .service import _IDLE_TIMEOUT_S, _await_disconnect, _check_fields, _run_inference, _ServiceState, serve

if TYPE_CHECKING:
    import torch

_LOGGER = logging.getLogger(__name__)

DEFAULT_WORLD_MODEL_ENDPOINT = (
    f"unix:///tmp/isaaclab-cosmos-world-model-{os.getuid()}.sock"
    if hasattr(socket, "AF_UNIX") and hasattr(os, "getuid")
    else "tcp://127.0.0.1:5556"
)
"""Default world-model endpoint; it differs from the camera service's so both can run on one machine."""

NUM_CONTROLS = 59
"""Number of ``unified_v1`` controls; the model pads them to 64."""

_SEED_LIMIT = 2**32
"""Seeds are integers in ``[0, 2**32)``, the range of the Framework's sampler."""


@dataclass(frozen=True)
class Cosmos3Settings:
    """Inference settings of a ``unified_v1`` checkpoint, fixed for the service's lifetime."""

    checkpoint_path: str
    """Directory of a local HF export, or of a DCP checkpoint together with :attr:`config_file`."""
    output_dir: str
    """Directory the Cosmos Framework writes inference artifacts to."""
    config_file: str | None = None
    """Saved YAML configuration of a DCP checkpoint."""
    mode: Literal["window", "session"] = "window"
    """``window`` conditions each prediction on the last generated image (Stage 0 checkpoints) and serves any number
    of environments. ``session`` keeps the native causal history of distilled checkpoints and serves one."""
    image_size: tuple[int, int] = (480, 640)
    """Height and width of the frames [pixels]."""
    chunk_size: int = 16
    """Frames predicted, and controls consumed, per step. Divisible by 4."""
    fps: int = 30
    domain_name: str = "hand_pose"
    resolution: str = "480"
    """Resolution tier of the Cosmos Framework."""
    num_inference_steps: int = 30
    guidance: float = 1.0
    shift: float = 10.0
    use_ema_weights: bool = True


class Cosmos3WorldModel:
    """Resident Cosmos3 weights; per-environment state belongs to the session.

    Construction loads the weights, so the service builds it only after it reserved its endpoint.
    """

    def __init__(self, settings: Cosmos3Settings, device: str = "cuda:0"):
        import torch

        torch.cuda.set_device(torch.device(device))
        self.settings = settings
        self.capabilities = {
            "kind": "world_model",
            "device": device,
            "chunk_length": settings.chunk_size,
            "fps": settings.fps,
            "image_size": list(settings.image_size),
            "max_envs": 1 if settings.mode == "session" else None,
        }
        # The service samples frozen dynamics; the Framework default enables its separate training stack on import.
        os.environ.setdefault("COSMOS_TRAINING", "0")
        from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
        from cosmos_framework.inference.args import OmniSetupOverrides
        from cosmos_framework.inference.bimanual import BimanualActionContract

        _allow_saved_config_rewrites()
        kwargs = {"config_file": settings.config_file} if settings.config_file is not None else {}
        setup = OmniSetupOverrides(
            checkpoint_path=settings.checkpoint_path,
            output_dir=settings.output_dir,
            use_ema_weights=settings.use_ema_weights,
            use_torch_compile=False,
            use_cuda_graphs=False,
            diffusion_cache=False,
            # The service calls the model directly, which skips the guardrails of the Framework's own pipeline.
            guardrails=False,
            dp_replicate_size=1,
            dp_shard_size=1,
            cp_size=1,
            cfgp_size=1,
            **kwargs,
        ).build_setup()
        self.pipe = setup.get_inference_cls().create(setup)
        self.contract = getattr(self.pipe.model, "_bimanual_action_contract", None) or BimanualActionContract(
            NUM_CONTROLS, 64, {settings.domain_name: get_domain_id(settings.domain_name)}
        )

    def warmup(self) -> None:
        """No-op: the model is not compiled."""

    def open_session(self, num_envs: int) -> Cosmos3Session:
        """Open a session for ``num_envs`` environments."""
        limit = self.capabilities["max_envs"]
        if limit is not None and num_envs > limit:
            raise ValueError(f"Cosmos3 {self.settings.mode} mode serves at most {limit} environments.")
        return Cosmos3Session(self, num_envs)

    def close(self) -> None:
        self.pipe = self.contract = None


class Cosmos3Session:
    """The environments of one client: each one's last generated image, prompt, seed and elapsed frames."""

    def __init__(self, model: Cosmos3WorldModel, num_envs: int):
        self.model = model
        self.num_envs = num_envs
        self.condition: list[torch.Tensor | None] = [None] * num_envs
        self.prompts = [""] * num_envs
        self.seeds = [0] * num_envs
        self.elapsed = [0] * num_envs
        self.native = None

    def reset(self, env_ids: list[int], images: np.ndarray, prompts: list[str], seeds: list[int]) -> None:
        """Start the selected environments from their start images ``(K, H, W, 3)``."""
        import torch

        settings = self.model.settings
        with torch.inference_mode():
            if settings.mode == "session":
                if self.native is not None:
                    self.native.close()
                self.native = self.model.pipe.open_bimanual_session(
                    vision_path=_write_png(images[0]),
                    prompt=prompts[0],
                    domain_name=settings.domain_name,
                    fps=settings.fps,
                    seed=seeds[0],
                    resolution=settings.resolution,
                )
                frames = _pixels(self.native.observation)[:, -1]
            else:
                frames = torch.from_numpy(np.ascontiguousarray(images))
        for env, frame, prompt, seed in zip(env_ids, frames, prompts, seeds, strict=True):
            self.condition[env] = _condition(frame)
            self.prompts[env] = prompt
            self.seeds[env] = seed
            self.elapsed[env] = 0

    def step(self, controls: np.ndarray) -> torch.Tensor:
        """Predict the next chunk of every environment from controls ``(N, T, 59)``; return frames
        ``(N, T, H, W, 3)``."""
        import torch

        settings = self.model.settings
        model = self.model.pipe.model
        with torch.inference_mode():
            if settings.mode == "session":
                rgb = self.native.step({"action_space": "normalized_unified_v1", "action": controls[0].tolist()})
                frames = _pixels(rgb)
            else:
                chunks = []
                for env in range(self.num_envs):
                    result = model.generate_samples_from_batch(
                        self._batch(self.condition[env], controls[env], self.prompts[env]),
                        guidance=settings.guidance,
                        shift=settings.shift,
                        seed=[(self.seeds[env] + self.elapsed[env]) % _SEED_LIMIT],
                        num_steps=settings.num_inference_steps,
                        n_sample=1,
                    )
                    chunks.append(_pixels((model.decode(result["vision"][0]) + 1) / 2)[:, 1:])
                frames = torch.cat(chunks)
        for env, frame in enumerate(frames):
            self.condition[env] = _condition(frame[-1])
            self.elapsed[env] += settings.chunk_size
        return frames

    def close(self) -> None:
        if self.native is not None:
            self.native.close()
            self.native = None

    def _batch(self, condition: torch.Tensor, controls: np.ndarray, prompt: str) -> dict:
        import torch
        from cosmos_framework.inference.action import build_action_batch
        from cosmos_framework.inference.args import ModelMode
        from cosmos_framework.inference.inference import _finalize_data_batch

        settings = self.model.settings
        model = self.model.pipe.model
        payload = {"action_space": "normalized_unified_v1", "action": controls.tolist()}
        values, domains = self.model.contract.validate(
            payload, action_count=settings.chunk_size, domain_name=settings.domain_name
        )
        padded = torch.zeros(settings.chunk_size, 64, dtype=torch.float32)
        padded[:, :NUM_CONTROLS] = torch.tensor(values, dtype=torch.float32)
        batch = build_action_batch(
            video=condition,
            action=padded,
            raw_action_dim=NUM_CONTROLS,
            prompt=prompt,
            view_point="ego_view",
            domain_name=settings.domain_name,
            model_mode=ModelMode.FORWARD_DYNAMICS,
            action_chunk_size=settings.chunk_size,
            fps=settings.fps,
            resolution=settings.resolution,
            input_video_key=model.input_video_key,
            device=model.tensor_kwargs["device"],
            action_domain_ids=torch.tensor(domains, dtype=torch.long),
            format_prompt_as_json=False,
        )
        return _finalize_data_batch(batch, 1, model)


def _pixels(rgb: torch.Tensor) -> torch.Tensor:
    """Convert decoded RGB in [0, 1], shape ``(1, 3, T, H, W)``, to ``uint8`` frames ``(1, T, H, W, 3)``."""
    import torch

    # Match the native public API before using pixels as the next condition.
    return rgb.detach().clamp(0, 1).mul(255).to(torch.uint8).permute(0, 2, 3, 4, 1)


def _condition(frame: torch.Tensor) -> torch.Tensor:
    """The next condition from a frame ``(H, W, 3)``: a host ``uint8`` clip ``(3, 1, H, W)``, as the Framework reads
    it from a file."""
    return frame.permute(2, 0, 1)[:, None].cpu()


def _write_png(image: np.ndarray) -> Path:
    """Write a start image for the Framework's native session, which opens it from a file."""
    import torch
    from torchvision.io import write_png

    path = Path(tempfile.mkdtemp(prefix="cosmos3-start-")) / "start.png"
    write_png(torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1), str(path))
    return path


def _allow_saved_config_rewrites() -> None:
    """Let the Framework update its saved config under OmegaConf 2.4.

    While building the network, the Framework normalizes and upgrades the saved ``multiview_attention`` config in
    place. OmegaConf 2.4 marks that config read-only, so the writes raise; with OmegaConf 2.3 they succeeded. The
    writes only fill defaults and normalize values, so each call runs with the read-only flag lifted.
    """
    from omegaconf import DictConfig, flag_override

    module = importlib.import_module("cosmos_framework.configs.base.defaults.multiview_attention")
    for name in ("normalize_multiview_temporal_windows", "upgrade_saved_multiview_attention"):
        original = getattr(module, name)

        def writable(config, *args, _original=original, **kwargs):
            if isinstance(config, DictConfig):
                with flag_override(config, "readonly", False):
                    return _original(config, *args, **kwargs)
            return _original(config, *args, **kwargs)

        # Modules that imported the function by name keep their own reference, so replace those too.
        for other in list(sys.modules.values()):
            if getattr(other, name, None) is original:
                setattr(other, name, writable)


class _FrameOutput:
    """The GPU buffer and event the service writes each chunk's frames to, for a client on this machine (CUDA IPC).

    The service owns both, and the client maps them. The client releases its mapping before it
    closes the session or disconnects, and the service frees them only after that.
    """

    def __init__(self, device: torch.device, shape: tuple[int, ...]):
        from .. import _cuda_ipc

        with ExitStack() as created:
            self.buffer = _cuda_ipc.SharedBuffer(device, shape)
            created.callback(self.buffer.close)
            self.ready = _cuda_ipc.SharedEvent(device)
            created.callback(self.ready.close)
            self._created = created.pop_all()
        self.handles = {"output": self.buffer.handle, "output_ready": self.ready.handle}

    def write(self, frames: torch.Tensor) -> None:
        """Copy a chunk into the buffer and record the event for the client's stream (inference thread)."""
        import torch

        self.buffer.tensor.copy_(frames.to(self.buffer.device))
        self.ready.record()
        # A GPU fault must reach the client as an error reply; its stream would otherwise wait on the event forever.
        torch.cuda.current_stream(self.buffer.device).synchronize()

    def close(self, *, peer_released: bool) -> None:
        """Free the buffer and event, or keep them until exit if the client may still have them mapped."""
        from .. import _cuda_ipc

        if peer_released:
            self._created.close()
        else:
            _cuda_ipc._KEPT.append(self._created.pop_all())


def handle_world_model_connection(
    connection: socket.socket,
    state: _ServiceState,
    connections: set[socket.socket],
    connections_lock: threading.Lock,
) -> None:
    """Serve one client's world-model session: ``status``, ``open``, ``reset``, ``step``, ``close``, ``shutdown``.

    Chunks return through the socket or, when the client opened the session with CUDA IPC, through a GPU buffer. A
    failed reset or step ends the session, since the model may already have advanced its state.
    """
    token = object()
    session = None
    output: _FrameOutput | None = None
    client_released = False
    num_envs = 0
    try:
        while not state.stopping.is_set():
            connection.settimeout(_IDLE_TIMEOUT_S)
            metadata, arrays = receive_message(connection)
            operation = metadata.get("op")
            if operation == "status":
                _check_fields(metadata, {"op", "version"}, arrays)
                with state.ownership_lock:
                    active = state.owner is not None
                send_message(
                    connection,
                    {
                        "ok": True,
                        "ready": not state.stopping.is_set(),
                        "capabilities": {
                            **state.model.capabilities,
                            "transports": state.transports,
                            "host_id": state.host_id,
                            "pci_bus_id": state.pci_bus_id,
                        },
                        "session_active": active,
                    },
                )
            elif operation == "open":
                _check_fields(metadata, {"op", "version", "num_envs"} | ({"transport"} & metadata.keys()), arrays)
                if session is not None:
                    raise RuntimeError("This connection already has a world-model session.")
                num_envs = metadata["num_envs"]
                if type(num_envs) is not int or num_envs < 1:
                    raise ValueError("A world-model session needs a positive num_envs.")
                transport = metadata.get("transport", "socket")
                if transport not in state.transports:
                    raise ValueError(f"This Cosmos service does not offer the {transport} transport.")
                with state.ownership_lock:
                    if state.owner is not None:
                        raise RuntimeError("The world model already serves a session; close it before connecting.")
                    state.owner = token
                session = _run_inference(state, state.model.open_session, num_envs)
                reply = {"ok": True}
                if transport == "cuda_ipc":
                    import torch

                    capabilities = state.model.capabilities
                    shape = (num_envs, capabilities["chunk_length"], *capabilities["image_size"], 3)
                    output = _run_inference(state, _FrameOutput, torch.device(capabilities["device"]), shape)
                    reply["ipc"] = output.handles
                send_message(connection, reply)
            elif operation == "reset":
                _check_fields(metadata, {"op", "version", "env_ids", "prompts", "seeds"}, arrays, allow_arrays=True)
                if session is None:
                    raise RuntimeError("Open a world-model session before resetting environments.")
                env_ids, prompts, seeds = _reset_arguments(metadata, num_envs)
                height, width = state.model.capabilities["image_size"]
                if (
                    len(arrays) != 1
                    or arrays[0].dtype != np.uint8
                    or arrays[0].shape != (len(env_ids), height, width, 3)
                ):
                    raise ValueError(f"A reset sends one uint8 start image ({height}, {width}, 3) per environment.")
                _run_inference(state, session.reset, env_ids, arrays[0], prompts, seeds)
                send_message(connection, {"ok": True})
            elif operation == "step":
                _check_fields(metadata, {"op", "version"}, arrays, allow_arrays=True)
                if session is None:
                    raise RuntimeError("Open a world-model session before stepping it.")
                chunk = state.model.capabilities["chunk_length"]
                if (
                    len(arrays) != 1
                    or arrays[0].dtype != np.float32
                    or arrays[0].shape != (num_envs, chunk, NUM_CONTROLS)
                ):
                    raise ValueError(f"A step sends float32 controls ({num_envs}, {chunk}, {NUM_CONTROLS}).")
                if output is None:
                    frames = _run_inference(state, session.step, arrays[0])
                    send_message(connection, {"ok": True}, [frames.cpu().numpy()])
                else:
                    _run_inference(state, lambda controls: output.write(session.step(controls)), arrays[0])
                    send_message(connection, {"ok": True})
            elif operation == "close":
                _check_fields(metadata, {"op", "version"}, arrays)
                if session is not None:
                    _run_inference(state, session.close)
                    session = None
                if output is not None:
                    # The client released its mapping before it asked to close.
                    _run_inference(state, output.close, peer_released=True)
                    output = None
                with state.ownership_lock:
                    if state.owner is token:
                        state.owner = None
                send_message(connection, {"ok": True})
            elif operation == "shutdown":
                _check_fields(metadata, {"op", "version"}, arrays)
                send_message(connection, {"ok": True})
                state.stopping.set()
                break
            else:
                raise ProtocolError(
                    "Unknown world-model operation; expected status, open, reset, step, close or shutdown."
                )
    except ConnectionError:
        # The client disconnected. It releases the output buffer first, and a process's mappings end with it. The
        # service's own shutdown also ends the connection, which confirms nothing.
        client_released = not state.stopping.is_set()
    except OSError:
        pass
    except Exception as exc:
        if session is not None:
            _LOGGER.exception("World-model session failed.")
        with suppress(OSError):
            send_message(connection, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            if session is not None:
                state.executor.submit(session.close).result()
        except Exception:
            _LOGGER.exception("Failed to close a world-model session after its client disconnected.")
        finally:
            with state.ownership_lock:
                if state.owner is token:
                    state.owner = None
            try:
                if output is not None:
                    # Without the client's confirmation that it released the buffer, it stays allocated until exit.
                    with suppress(OSError):
                        connection.settimeout(_IDLE_TIMEOUT_S)
                    released = client_released or _await_disconnect(connection, state.stopping)
                    state.executor.submit(output.close, peer_released=released).result()
            finally:
                with connections_lock:
                    connections.discard(connection)
                connection.close()


def _reset_arguments(metadata: dict, num_envs: int) -> tuple[list[int], list[str], list[int]]:
    env_ids, prompts, seeds = metadata["env_ids"], metadata["prompts"], metadata["seeds"]
    if (
        not isinstance(env_ids, list)
        or not env_ids
        or any(type(env) is not int or not 0 <= env < num_envs for env in env_ids)
        or len(set(env_ids)) != len(env_ids)
    ):
        raise ValueError("A world-model reset names distinct environments of the session.")
    if not isinstance(prompts, list) or len(prompts) != len(env_ids) or any(not isinstance(p, str) for p in prompts):
        raise ValueError("A world-model reset gives one prompt per environment.")
    if (
        not isinstance(seeds, list)
        or len(seeds) != len(env_ids)
        or any(type(seed) is not int or not 0 <= seed < _SEED_LIMIT for seed in seeds)
    ):
        raise ValueError("A world-model reset gives one integer seed in [0, 2**32) per environment.")
    return env_ids, prompts, seeds


def main(args: list[str] | None = None) -> None:
    """Run the world-model service in the Cosmos Framework environment."""
    parser = argparse.ArgumentParser(description="Load Cosmos3 once, then serve world-model sessions to Isaac Lab.")
    parser.add_argument("--checkpoint", required=True, help="Local HF export, or DCP root with --config-file.")
    parser.add_argument("--config-file", help="Saved YAML configuration of a DCP checkpoint.")
    parser.add_argument("--output-dir", default="outputs/cosmos3", help="Directory for Framework artifacts.")
    parser.add_argument("--mode", choices=("window", "session"), default="window", help="Inference mode.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_WORLD_MODEL_ENDPOINT,
        help=f"unix:///path (same machine, owner-only) or tcp://host:port (default {DEFAULT_WORLD_MODEL_ENDPOINT}).",
    )
    options = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Cosmos3Settings(
        checkpoint_path=options.checkpoint,
        config_file=options.config_file,
        output_dir=options.output_dir,
        mode=options.mode,
    )
    serve(
        lambda: Cosmos3WorldModel(settings, options.device),
        options.endpoint,
        handler=handle_world_model_connection,
    )


if __name__ == "__main__":
    main()
