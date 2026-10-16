# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for resolving RSL-RL checkpoints stored on Weights & Biases (wandb)."""

from __future__ import annotations

import contextlib
import logging
import os
import re
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

# Suppress import error if wandb is not installed; only needed when a wandb checkpoint is requested.
with contextlib.suppress(ImportError):
    import wandb

# Matches a run URL copied from the browser, e.g.
# https://wandb.ai/<entity>/<project>/runs/<run_id>[?<query>][#<fragment>]
_WANDB_URL_PATTERN = re.compile(r"^https?://wandb\.ai/(?P<entity>[^/]+)/(?P<project>[^/]+)/runs/(?P<run_id>[^/?#]+)")
# Matches the shorthand URI accepted anywhere a run URL is, e.g. wandb:<entity>/<project>/<run_id>
_WANDB_URI_PATTERN = re.compile(r"^wandb:(?P<entity>[^/]+)/(?P<project>[^/]+)/(?P<run_id>[^/?#]+)")


def is_wandb_checkpoint(path: str | None) -> bool:
    """Check whether ``path`` identifies a Weights & Biases run rather than a local or Nucleus path.

    Args:
        path: The ``--checkpoint`` value passed on the command line.

    Returns:
        Whether ``path`` matches a ``https://wandb.ai/<entity>/<project>/runs/<run_id>`` URL or a
        ``wandb:<entity>/<project>/<run_id>`` shorthand.
    """
    return bool(path) and (_WANDB_URL_PATTERN.match(path) is not None or _WANDB_URI_PATTERN.match(path) is not None)


def resolve_wandb_checkpoint(path: str, download_dir: str = "logs/wandb_checkpoints") -> str:
    """Download the checkpoint referenced by a wandb run URL or shorthand URI.

    The checkpoint iteration defaults to the latest one available on the run. A specific iteration
    can be requested by appending a ``checkpoint`` query parameter, e.g.
    ``https://wandb.ai/<entity>/<project>/runs/<run_id>?checkpoint=100``.

    Args:
        path: A ``https://wandb.ai/<entity>/<project>/runs/<run_id>`` URL or a
            ``wandb:<entity>/<project>/<run_id>`` shorthand, identifying the run to download the
            model checkpoint from.
        download_dir: Directory the checkpoint is downloaded to.

    Returns:
        The local path to the downloaded model checkpoint.

    Raises:
        ValueError: If ``path`` does not match a recognized wandb reference.
    """
    match = _WANDB_URL_PATTERN.match(path) or _WANDB_URI_PATTERN.match(path)
    if match is None:
        raise ValueError(f"'{path}' is not a recognized Weights & Biases checkpoint reference.")
    entity, project, run_id = match.group("entity", "project", "run_id")

    checkpoint = None
    query = urlparse(path).query
    if query:
        checkpoint_values = parse_qs(query).get("checkpoint")
        if checkpoint_values:
            checkpoint = int(checkpoint_values[0])

    return get_model_checkpoint(
        run_id=run_id, project=project, checkpoint=checkpoint, wandb_entity=entity, download_dir=download_dir
    )


def resolve_wandb_entity(explicit_entity: str | None = None) -> str | None:
    """Resolve the wandb entity (user or team) to use for a run or checkpoint lookup.

    Args:
        explicit_entity: An entity parsed from an explicit checkpoint reference. Takes precedence
            when given.

    Returns:
        ``explicit_entity`` if given, otherwise the ``WANDB_ENTITY`` or ``WANDB_USERNAME``
        environment variable (in that order), or None if neither is set.
    """
    if explicit_entity is not None:
        return explicit_entity
    return os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_USERNAME")


def announce_new_run(project: str, entity: str | None = None) -> None:
    """Pin a deterministic run id for an upcoming wandb run and print how to find it later.

    Sets the ``WANDB_RUN_ID`` environment variable (unless already set, e.g. by the caller
    resuming a specific run) so the run id is known before the logger lazily calls
    ``wandb.init()``, and prints the reference to pass as ``--checkpoint`` to resume or play it.

    Args:
        project: The wandb project the run will be logged to.
        entity: The wandb entity (user or team) the run will be logged under. If None, the run
            will log under the account's default entity and only the project and run id are
            printed.
    """
    if "wandb" not in globals():
        return
    if "WANDB_RUN_ID" not in os.environ:
        os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
    run_id = os.environ["WANDB_RUN_ID"]

    if entity is None:
        logger.info(f"Logging to wandb project '{project}', run id '{run_id}'.")
        logger.info(
            f"Once your entity is known, resume or play it with: --checkpoint wandb:<entity>/{project}/{run_id}"
        )
    else:
        logger.info(f"Logging to wandb run: wandb:{entity}/{project}/{run_id}")
        logger.info(f"Resume or play it later with: --checkpoint wandb:{entity}/{project}/{run_id}")


def get_model_checkpoint(
    run_id: str,
    project: str,
    checkpoint: int | None = None,
    wandb_entity: str | None = None,
    download_dir: str = "logs/wandb_checkpoints",
    *,
    checkpoint_pattern: str = r"model_\d+\.pt",
) -> str:
    """Download a model checkpoint logged to Weights & Biases (wandb).

    Args:
        run_id: The ID of the wandb run.
        project: The name of the wandb project the run belongs to.
        checkpoint: The specific checkpoint iteration to download. If None, downloads the latest.
        wandb_entity: The wandb entity (user or team) that owns the project. If None, uses the
            ``WANDB_ENTITY`` or ``WANDB_USERNAME`` environment variable, then the local W&B login.
        download_dir: Directory the checkpoint is downloaded to.
        checkpoint_pattern: Regular expression matching file names when ``checkpoint`` is None.
            Multiple matches are sorted naturally to select the highest numeric iteration.

    Returns:
        The local path to the downloaded model checkpoint.

    Raises:
        ValueError: If no matching checkpoint is found on the run.
    """
    import wandb

    wandb_entity = resolve_wandb_entity(wandb_entity)
    run_path = "/".join(filter(None, (wandb_entity, project, run_id)))
    logger.info(f"Downloading model checkpoint from wandb run: {run_path}")
    api = wandb.Api()
    wdb_run = api.run(run_path)

    pattern = rf"model_{checkpoint}\.pt" if checkpoint is not None else checkpoint_pattern
    models = [file for file in wdb_run.files() if re.fullmatch(pattern, os.path.basename(file.name))]
    if not models:
        if checkpoint is not None:
            raise ValueError(f"Model checkpoint iteration {checkpoint} not found in run '{run_id}'.")
        raise ValueError(f"No model checkpoints found matching '{pattern}' in wandb run '{run_path}'.")
    model = max(models, key=lambda file: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", file.name)])

    target_folder = os.path.join(download_dir, project, run_id)
    os.makedirs(target_folder, exist_ok=True)
    model.download(root=target_folder, replace=True)
    return os.path.join(target_folder, model.name)
