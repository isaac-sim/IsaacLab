# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos prompts read from files: text as written, JSON as-is, one prompt per file."""

import json

import pytest
from isaaclab_experimental.cosmos import read_prompt_files

_SCENE = {
    "subjects": [{"description": "A small apple with a yellow upper half and a red lower half."}],
    "context": "Preserve the geometry, camera pose and movements of the depth conditioning.",
}


def test_text_and_json_files_become_prompts_as_written_in_the_order_given(tmp_path):
    """Text loses only its surrounding whitespace; JSON reaches Cosmos as the file's own text."""
    (tmp_path / "b.txt").write_text("  A grey bedsheet covers the table.\nSoft light.\n", encoding="utf-8")
    scene = json.dumps(_SCENE, indent=2)
    (tmp_path / "a.json").write_text(scene + "\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("A white plate.", encoding="utf-8")
    sheet = "A grey bedsheet covers the table.\nSoft light."

    assert read_prompt_files([str(tmp_path / "b.txt"), str(tmp_path / "a.json")]) == [sheet, scene]
    # A pattern expands in sorted order; a bracketed list keeps its own order.
    assert read_prompt_files([str(tmp_path / "*.txt")]) == [sheet, "A white plate."]
    assert read_prompt_files([f"[{tmp_path / 'c.txt'}, {tmp_path / 'a.json'}]"]) == ["A white plate.", scene]


@pytest.mark.parametrize(
    "name,content,message",
    [
        ("missing.txt", None, "does not exist"),
        ("empty.txt", " \n", "is empty"),
        ("broken.json", '{"subjects": [', "not valid JSON"),
        ("prompt.md", "A lab.", "must be a .txt or .json file"),
        ("none-*.json", None, "No Cosmos prompt file matches"),
    ],
)
def test_unusable_prompt_files_are_rejected_with_their_path(tmp_path, name, content, message):
    if content is not None:
        (tmp_path / name).write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_prompt_files([str(tmp_path / name)])
