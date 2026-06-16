# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for scripts/benchmarks/_common.py helpers (Isaac-Sim-free)."""

from __future__ import annotations

import sys

import pytest

from scripts.benchmarks._common import (
    get_backend_type,
    get_backend_types,
    import_module_from_path,
    preset_tokens,
)

# ---------------------------------------------------------------------------
# preset_tokens
# ---------------------------------------------------------------------------


class TestPresetTokens:
    """Test cases for preset_tokens extraction from the unfolded Hydra argument remainder."""

    def test_comma_separated_tokens(self):
        """Test that multiple comma-separated tokens are split correctly."""
        assert preset_tokens(["presets=newton_mjwarp,rgb"]) == ["newton_mjwarp", "rgb"]

    def test_empty_presets_value_returns_empty_list(self):
        """Test that ``presets=`` with an empty value returns an empty list."""
        assert preset_tokens(["presets="]) == []

    def test_no_presets_arg_returns_empty_list(self):
        """Test that an argument list with no ``presets=`` entry returns an empty list."""
        assert preset_tokens(["--task", "X"]) == []

    def test_single_token(self):
        """Test that a single preset token is returned as a one-element list."""
        assert preset_tokens(["presets=physx"]) == ["physx"]

    def test_returns_empty_for_empty_list(self):
        """Test that an empty argument list returns an empty list."""
        assert preset_tokens([]) == []


# ---------------------------------------------------------------------------
# get_backend_type
# ---------------------------------------------------------------------------


class TestGetBackendType:
    """Test cases for the CLI backend name to canonical type mapping."""

    def test_omniperf_canonical(self):
        """Test that the canonical short name maps to itself."""
        assert get_backend_type("omniperf") == "omniperf"

    def test_json_file_metrics_legacy(self):
        """Test that the legacy long-form JSONFileMetrics maps to json."""
        assert get_backend_type("JSONFileMetrics") == "json"

    def test_osmo_kpi_file_legacy(self):
        """Test that the legacy long-form OsmoKPIFile maps to osmo."""
        assert get_backend_type("OsmoKPIFile") == "osmo"

    def test_summary_canonical(self):
        """Test that the summary backend maps to itself."""
        assert get_backend_type("summary") == "summary"

    def test_unknown_defaults_to_omniperf(self):
        """Test that an unknown backend name falls back to omniperf."""
        assert get_backend_type("unknown") == "omniperf"

    def test_schema_canonical(self):
        """Test that the schema backend maps to itself."""
        assert get_backend_type("schema") == "schema"


# ---------------------------------------------------------------------------
# get_backend_types
# ---------------------------------------------------------------------------


class TestGetBackendTypes:
    """Test cases for splitting a comma-separated --benchmark_backend value."""

    def test_single_token(self):
        """Test that a single token yields a one-element list."""
        assert get_backend_types("schema") == ["schema"]

    def test_comma_separated_tokens(self):
        """Test that comma-separated tokens map to an ordered list."""
        assert get_backend_types("schema,omniperf") == ["schema", "omniperf"]

    def test_duplicate_tokens_deduplicated(self):
        """Test that duplicate canonical types are removed while preserving order."""
        assert get_backend_types("schema,schema") == ["schema"]

    def test_empty_input_falls_back_to_omniperf(self):
        """Test that an empty input yields the default omniperf backend."""
        assert get_backend_types("") == ["omniperf"]

    def test_unknown_token_falls_back_to_omniperf(self):
        """Test that an unknown token falls back to omniperf."""
        assert get_backend_types("nonsense") == ["omniperf"]

    def test_legacy_alias_and_canonical(self):
        """Test that a legacy long-form alias and a canonical token both normalize."""
        assert get_backend_types("JSONFileMetrics,osmo") == ["json", "osmo"]

    def test_whitespace_tokens_ignored(self):
        """Test that surrounding whitespace around tokens is stripped."""
        assert get_backend_types(" schema , omniperf ") == ["schema", "omniperf"]


# ---------------------------------------------------------------------------
# import_module_from_path
# ---------------------------------------------------------------------------


class TestImportModuleFromPath:
    """Test cases for loading a Python file as a module by explicit path."""

    def test_loads_module_and_exposes_attributes(self, tmp_path):
        """Test that a module written to a temp file can be loaded and its VALUE attribute read."""
        mod_file = tmp_path / "tmp_mod_xyz.py"
        mod_file.write_text("VALUE = 42\n")
        mod_name = "tmp_mod_xyz_test_common"
        # Guard: remove any prior registration so the test is isolated.
        sys.modules.pop(mod_name, None)
        mod = import_module_from_path(mod_name, mod_file)
        assert mod.VALUE == 42
        # Clean up to avoid cross-test contamination.
        sys.modules.pop(mod_name, None)

    def test_cached_in_sys_modules(self, tmp_path):
        """Test that a second call returns the cached module without re-executing the file."""
        mod_file = tmp_path / "tmp_mod_cache.py"
        mod_file.write_text("VALUE = 99\n")
        mod_name = "tmp_mod_cache_test_common"
        sys.modules.pop(mod_name, None)
        mod1 = import_module_from_path(mod_name, mod_file)
        # Overwrite file contents — a re-import would change VALUE, but cached must stay 99.
        mod_file.write_text("VALUE = 0\n")
        mod2 = import_module_from_path(mod_name, mod_file)
        assert mod1 is mod2
        assert mod2.VALUE == 99
        sys.modules.pop(mod_name, None)

    def test_raises_for_nonexistent_path(self, tmp_path):
        """Test that a missing file raises an error (ImportError or FileNotFoundError)."""
        mod_name = "tmp_mod_missing_test_common"
        sys.modules.pop(mod_name, None)
        with pytest.raises((ImportError, FileNotFoundError)):
            import_module_from_path(mod_name, tmp_path / "does_not_exist.py")
        sys.modules.pop(mod_name, None)
