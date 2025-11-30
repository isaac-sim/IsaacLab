from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import torch
import asyncio
import threading

from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.utils.datasets import EpisodeData

from isaaclab_mimic.async_writer import AsyncWriter


def _flatten_episode_dict(episode_data: Dict[str, Any], prefix: str = "") -> Dict[str, torch.Tensor]:
    """Flattens nested dict tensors in EpisodeData.data into a flat dict with '/'-joined keys.

    Returns a mapping key -> tensor shaped (T, ...). All tensors remain as torch tensors (on CPU).
    """
    flat: Dict[str, torch.Tensor] = {}
    for key, value in episode_data.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_episode_dict(value, prefix=full_key))
        elif isinstance(value, torch.Tensor):
            # Ensure on CPU for downstream numpy conversion
            flat[full_key] = value.detach().to("cpu")
        else:
            # Ignore unsupported types silently
            continue
    return flat


def _episode_to_dataframe_from_dict(data_dict: Dict[str, Any]) -> pd.DataFrame:
    """Converts a nested episode data dict to a pandas DataFrame (one row per timestep)."""
    if not data_dict:
        return pd.DataFrame()

    flat = _flatten_episode_dict(data_dict)

    # Determine timesteps (prefer 'actions' length, else max length across tensors)
    def _tensor_len(t: torch.Tensor) -> int:
        return int(t.shape[0]) if t.dim() > 0 else 1

    timesteps = 0
    if "actions" in flat:
        timesteps = _tensor_len(flat["actions"])
    else:
        for t in flat.values():
            timesteps = max(timesteps, _tensor_len(t))

    if timesteps == 0:
        return pd.DataFrame()

    columns: Dict[str, List[Any]] = {}
    for k, t in flat.items():
        if _tensor_len(t) == timesteps:
            # Per-step series
            columns[k] = [t[i] for i in range(timesteps)]
        else:
            # Do not broadcast single-step values like initial_state; keep single entry
            single = t[0] if t.dim() > 0 else t
            if k.endswith("initial_state") or k.split("/")[-1] == "initial_state":
                columns[k] = [single]
            else:
                # For non-initial_state, retain previous behavior (broadcast) to keep writer compatibility
                columns[k] = [single for _ in range(timesteps)]

    return pd.DataFrame(columns)


# take snapshot of episode data, onto cpu & cloned so live buffers arent used 
def _snapshot_episode_data(episode: EpisodeData) -> Dict[str, Any]:

    def _clone_tree(node: Any) -> Any:
        if isinstance(node, torch.Tensor):
            return node.detach().to("cpu").clone()
        if isinstance(node, dict):
            return {k: _clone_tree(v) for k, v in node.items()}
        return node

    return _clone_tree(episode.data)




class AsyncWriterRecorder(RecorderTerm):

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if AsyncWriter is None:
            raise RuntimeError("AsyncWriter could not be imported; cannot initialize AsyncWriterRecorder.")


        rm_cfg = getattr(env.cfg, "recorders", None)
        if rm_cfg is None:

            self._output_dir = "/tmp/isaaclab/logs"
            self._out_file = "dataset.hdf5"
        else:
            self._output_dir = rm_cfg.dataset_export_dir_path
            self._out_file = f"{rm_cfg.dataset_filename}.hdf5"

        self._writer = AsyncWriter(output_dir=self._output_dir)
        # Mirror HDF5DatasetFileHandler defaults: set env_name and type=2
        env_name = getattr(env.cfg, "env_name", "")
        try:
            self._writer.set_env_args({"env_name": env_name, "type": 2})
        except Exception:
            pass

        # Track pending async write tasks to allow explicit draining/close
        self._pending_tasks: set[asyncio.Task] = set()

    def record_pre_reset(self, env_ids: Sequence[int] | None) -> Tuple[str | None, torch.Tensor | dict | None]:
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))

      
        for env_id in env_ids:
            episode = self._env.recorder_manager.get_episode(env_id)
            if episode is None or episode.is_empty():
                continue
            self.schedule_async_write_for_episode(episode)
            # Clear episode buffer synchronously
            from isaaclab.utils.datasets import EpisodeData as _EpisodeData
            self._env.recorder_manager._episodes[env_id] = _EpisodeData()


        return None, None


    async def schedule_async_write_for_episode(self, episode: EpisodeData) -> None:
        snapshot = episode.data
        if not snapshot:
            return

        async def _do_write(data_snapshot: Dict[str, Any]):
            df = await asyncio.to_thread(_episode_to_dataframe_from_dict, data_snapshot)
            if not df.empty:
                await self._writer.write_trajectory_data_async(df, self._out_file, debug=False)

        task = asyncio.create_task(_do_write(snapshot))
        self._pending_tasks.add(task)
        task.add_done_callback(lambda t: self._pending_tasks.discard(t))
        # Return immediately; manager may await this coroutine, but work continues in background
        return

    # Synchronous writer for single-writer-thread consumer
    def schedule_sync_write_for_episode(self, episode: EpisodeData) -> None:
        snapshot = _snapshot_episode_data(episode)
        if not snapshot:
            return
        df = _episode_to_dataframe_from_dict(snapshot)
        if not df.empty:
            # direct, blocking write on the writer thread
            self._writer._write_trajectory_data_hdf5(df, self._out_file, debug=False)

    def close(self) -> None:
        try:
            if hasattr(self, "_writer") and self._writer is not None:
                self._writer.close()
        except Exception:
            pass

    async def flush_async(self) -> None:
        if not self._pending_tasks:
            return
        await asyncio.gather(*list(self._pending_tasks), return_exceptions=True)



