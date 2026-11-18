# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare the model files of the medical-pad example.

Checks that the world model, VAE, tokenizer and reward classifier listed in ``artifacts.json`` are in
``.pretrained_checkpoints``, downloads missing ones with ``--download``, and points the world model's saved config
at them.

Usage:

.. code-block:: bash

    uv run --extra cosmos3 python scripts/tools/cosmos/prepare_medical_pad.py --download
"""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "source/isaaclab_tasks/isaaclab_tasks/contrib/medical_pad_unfold"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--download", action="store_true", help="Download missing files from Hugging Face.")
args = parser.parse_args()


def main():
    manifest = json.loads((TASK / "assets/artifacts.json").read_text())
    models = ROOT / manifest["model_directory"]
    missing = []
    for name, artifact in manifest["artifacts"].items():
        for filename in artifact["files"]:
            target = models / name / filename
            if not target.is_file() and args.download:
                source = hf_hub_download(
                    repo_id=artifact["repo_id"],
                    revision=artifact["revision"],
                    filename="/".join(filter(None, (artifact["prefix"], filename))),
                    cache_dir=models.parent / "huggingface/hub",
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            if not target.is_file():
                missing.append(target.relative_to(ROOT))
    if missing:
        raise SystemExit("Missing model files (run with --download):\n" + "\n".join(map(str, missing)))

    # The saved world model config refers to the VAE, the tokenizer and the Qwen3-VL config of the Cosmos Framework.
    reasoner = (
        Path(importlib.util.find_spec("cosmos_framework").origin).parent
        / "model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
    )
    qwen_config = models / "world_model/resources" / reasoner.name
    qwen_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(reasoner, qwen_config)
    config_file = models / "world_model/config.json"
    config = json.loads(config_file.read_text())

    def point_to_local_files(value):
        if isinstance(value, dict):
            if "vae_path" in value:
                value["vae_path"] = str(models / "vae/Wan2.2_VAE.pth")
            if value.get("_target") == "qwen3_vl_mot_config_from_json_file":
                value["json_file"] = str(qwen_config)
            if value.get("_target") == "create_qwen2_tokenizer_with_download":
                value["pretrained_model_name"] = str(models / "tokenizer")
            for item in value.values():
                point_to_local_files(item)
        elif isinstance(value, list):
            for item in value:
                point_to_local_files(item)

    point_to_local_files(config)
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Model files are ready in {models.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
