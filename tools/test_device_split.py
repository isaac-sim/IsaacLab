# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``tools/_device_split.py``."""

from __future__ import annotations

import textwrap
from pathlib import Path

from _device_split import is_device_split_file


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test_foo.py"
    p.write_text(textwrap.dedent(content))
    return p


def test_single_mark(tmp_path):
    f = _write(
        tmp_path,
        """
        import pytest

        pytestmark = pytest.mark.device_split

        def test_x():
            pass
        """,
    )
    assert is_device_split_file(f) is True


def test_list_form_single_line(tmp_path):
    f = _write(
        tmp_path,
        """
        import pytest

        pytestmark = [pytest.mark.device_split, pytest.mark.foo]

        def test_x():
            pass
        """,
    )
    assert is_device_split_file(f) is True


def test_preloaded_source(tmp_path):
    source = textwrap.dedent(
        """
        import pytest

        pytestmark = pytest.mark.device_split
        """
    )
    assert is_device_split_file(tmp_path / "does_not_exist.py", source=source) is True


def test_no_mark(tmp_path):
    f = _write(
        tmp_path,
        """
        import pytest

        def test_x():
            pass
        """,
    )
    assert is_device_split_file(f) is False


def test_word_in_comment_does_not_match(tmp_path):
    f = _write(
        tmp_path,
        """
        import pytest

        # This file mentions device_split in a comment but is not marked.

        def test_x():
            pass
        """,
    )
    assert is_device_split_file(f) is False


def test_unrelated_pytestmark_does_not_match(tmp_path):
    f = _write(
        tmp_path,
        """
        import pytest

        pytestmark = pytest.mark.skipif(False, reason="x")

        def test_x():
            pass
        """,
    )
    assert is_device_split_file(f) is False


def test_missing_file(tmp_path):
    # A path that does not exist must not raise; treat as not-marked.
    assert is_device_split_file(tmp_path / "does_not_exist.py") is False
