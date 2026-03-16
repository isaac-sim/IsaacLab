# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the test selection script."""

import ast
import json
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import select_tests


def _load_matches_include_files():
    """Extract _matches_include_files from conftest.py without importing the module.

    conftest.py has top-level imports (junitparser) that are unavailable in
    lightweight test environments, so we parse the function source with AST
    and compile it in isolation.
    """
    source = Path(__file__).parent.joinpath("conftest.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_matches_include_files":
            func_source = textwrap.dedent(ast.get_source_segment(source, node))
            namespace: dict = {}
            exec(compile(func_source, "conftest.py", "exec"), namespace)  # noqa: S102
            return namespace["_matches_include_files"]
    raise RuntimeError("_matches_include_files not found in conftest.py")


_matches_include_files = _load_matches_include_files()


class TestLoadMapping:
    """Tests for loading the dependency mapping file."""

    def test_load_valid_mapping(self, tmp_path):
        """A valid mapping file should be loaded and its contents returned."""
        mapping = {
            "metadata": {
                "generated_at": "2026-03-16T04:00:00Z",
                "commit": "abc123",
                "test_file_count": 2,
                "source_file_count": 1,
            },
            "source_to_tests": {
                "source/isaaclab/isaaclab/utils/math.py": [
                    "source/isaaclab/test/utils/test_math.py",
                ],
            },
        }
        mapping_path = tmp_path / "test-dependency-map.json"
        mapping_path.write_text(json.dumps(mapping))

        result = select_tests.load_mapping(str(mapping_path))
        assert result["source_to_tests"]["source/isaaclab/isaaclab/utils/math.py"] == [
            "source/isaaclab/test/utils/test_math.py"
        ]

    def test_load_missing_mapping(self, tmp_path):
        """A missing mapping file should return None."""
        result = select_tests.load_mapping(str(tmp_path / "nonexistent.json"))
        assert result is None

    def test_load_empty_mapping(self, tmp_path):
        """An empty mapping file should return None."""
        mapping_path = tmp_path / "test-dependency-map.json"
        mapping_path.write_text("{}")

        result = select_tests.load_mapping(str(mapping_path))
        assert result is None

    def test_load_invalid_json(self, tmp_path):
        """An invalid JSON file should return None."""
        mapping_path = tmp_path / "test-dependency-map.json"
        mapping_path.write_text("not json")

        result = select_tests.load_mapping(str(mapping_path))
        assert result is None


class TestClassifyFile:
    """Tests for classifying changed files."""

    def test_test_file_in_subdirectory(self):
        """A test file in a test subdirectory should be classified as 'test'."""
        result = select_tests.classify_file("source/isaaclab/test/utils/test_math.py", "M", mapping_keys=set())
        assert result == "test"

    def test_test_file_directly_in_test_dir(self):
        """A test file directly in test/ (no subdirectory) should be classified as 'test'."""
        result = select_tests.classify_file("source/isaaclab_tasks/test/test_environments.py", "M", mapping_keys=set())
        assert result == "test"

    def test_test_file_under_scripts(self):
        """A test file under scripts/ should be classified as 'test'."""
        result = select_tests.classify_file("scripts/tools/test/test_something.py", "M", mapping_keys=set())
        assert result == "test"

    def test_mapped_source_file(self):
        """A Python source file that exists in the mapping should be classified as 'mapped'."""
        keys = {"source/isaaclab/isaaclab/utils/math.py"}
        result = select_tests.classify_file("source/isaaclab/isaaclab/utils/math.py", "M", mapping_keys=keys)
        assert result == "mapped"

    def test_unmapped_source_file(self):
        """A Python source file NOT in the mapping should be classified as 'unmapped'."""
        result = select_tests.classify_file("source/isaaclab/isaaclab/new_module.py", "M", mapping_keys=set())
        assert result == "unmapped"

    def test_non_python_file_under_source(self):
        """A non-Python file under source/ should be classified as 'non_python_source'."""
        result = select_tests.classify_file("source/isaaclab/test/utils/test_config.yaml", "M", mapping_keys=set())
        assert result == "non_python_source"

    def test_infrastructure_conftest(self):
        """tools/conftest.py should be classified as 'infrastructure'."""
        result = select_tests.classify_file("tools/conftest.py", "M", mapping_keys=set())
        assert result == "infrastructure"

    def test_infrastructure_test_settings(self):
        """tools/test_settings.py should be classified as 'infrastructure'."""
        result = select_tests.classify_file("tools/test_settings.py", "M", mapping_keys=set())
        assert result == "infrastructure"

    def test_infrastructure_root_pyproject(self):
        """Root pyproject.toml should be classified as 'infrastructure'."""
        result = select_tests.classify_file("pyproject.toml", "M", mapping_keys=set())
        assert result == "infrastructure"

    def test_infrastructure_setup_py(self):
        """A setup.py should be classified as 'infrastructure'."""
        result = select_tests.classify_file("source/isaaclab/setup.py", "M", mapping_keys=set())
        assert result == "infrastructure"

    def test_infrastructure_extension_toml(self):
        """An extension.toml in config/ should be classified as 'infrastructure'."""
        result = select_tests.classify_file("source/isaaclab/config/extension.toml", "M", mapping_keys=set())
        assert result == "infrastructure"

    def test_ci_tooling_select_tests(self):
        """tools/select_tests.py should be classified as 'infrastructure'."""
        result = select_tests.classify_file("tools/select_tests.py", "M", mapping_keys=set())
        assert result == "infrastructure"

    def test_ci_tooling_collect_coverage(self):
        """tools/collect_coverage_map.py should be classified as 'infrastructure'."""
        result = select_tests.classify_file("tools/collect_coverage_map.py", "M", mapping_keys=set())
        assert result == "infrastructure"

    def test_apps_file(self):
        """A file under apps/ should be classified as 'apps'."""
        result = select_tests.classify_file("apps/something.py", "M", mapping_keys=set())
        assert result == "apps"

    def test_docs_file_ignored(self):
        """A docs file should be classified as 'ignore'."""
        result = select_tests.classify_file("docs/README.md", "M", mapping_keys=set())
        assert result == "ignore"

    def test_github_file_is_infrastructure(self):
        """A .github/ file should be classified as 'infrastructure'."""
        result = select_tests.classify_file(".github/workflows/build.yaml", "M", mapping_keys=set())
        assert result == "infrastructure"

    def test_docker_file_is_infrastructure(self):
        """A docker/ file should be classified as 'infrastructure'."""
        result = select_tests.classify_file("docker/Dockerfile.base", "M", mapping_keys=set())
        assert result == "infrastructure"

    def test_deleted_source_file(self):
        """A deleted Python source file should be classified as 'deleted_source'."""
        result = select_tests.classify_file("source/isaaclab/isaaclab/old.py", "D", mapping_keys=set())
        assert result == "deleted_source"

    def test_deleted_non_source_file_ignored(self):
        """A deleted docs file should be classified as 'ignore'."""
        result = select_tests.classify_file("docs/old.md", "D", mapping_keys=set())
        assert result == "ignore"

    def test_renamed_file(self):
        """A renamed file should be classified as 'renamed'."""
        result = select_tests.classify_file("source/isaaclab/isaaclab/new_name.py", "R", mapping_keys=set())
        assert result == "renamed"


class TestStalenessCheck:
    """Tests for mapping staleness detection."""

    def test_fresh_mapping_is_not_stale(self):
        """A mapping generated just now should not be stale."""
        now = datetime.now(timezone.utc).isoformat()
        metadata = {"generated_at": now}
        assert select_tests.is_mapping_stale(metadata, max_age_days=7) is False

    def test_old_mapping_is_stale(self):
        """A mapping older than max_age_days should be stale."""
        old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        metadata = {"generated_at": old}
        assert select_tests.is_mapping_stale(metadata, max_age_days=7) is True

    def test_missing_timestamp_is_stale(self):
        """A mapping without generated_at should be considered stale."""
        assert select_tests.is_mapping_stale({}, max_age_days=7) is True


class TestAssignTestToJob:
    """Tests for assigning test files to CI jobs."""

    def test_physx_test(self):
        """A test under isaaclab_physx should go to test-physx."""
        assert select_tests.assign_test_to_job("source/isaaclab_physx/test/assets/test_articulation.py") == "test-physx"

    def test_newton_test(self):
        """A test under isaaclab_newton should go to test-newton."""
        assert (
            select_tests.assign_test_to_job("source/isaaclab_newton/test/assets/test_articulation.py") == "test-newton"
        )

    def test_general_test(self):
        """A test under isaaclab (core) should go to test-general."""
        assert select_tests.assign_test_to_job("source/isaaclab/test/utils/test_math.py") == "test-general"

    def test_tasks_1_test(self):
        """A tasks test in the first split should go to test-isaaclab-tasks."""
        assert (
            select_tests.assign_test_to_job("source/isaaclab_tasks/test/test_environments.py") == "test-isaaclab-tasks"
        )

    def test_tasks_2_test(self):
        """A tasks test in the second split should go to test-isaaclab-tasks-2."""
        assert (
            select_tests.assign_test_to_job("source/isaaclab_tasks/test/test_environment_determinism.py")
            == "test-isaaclab-tasks-2"
        )

    def test_environments_training(self):
        """test_environments_training.py should go to test-environments-training."""
        assert (
            select_tests.assign_test_to_job("source/isaaclab_tasks/test/test_environments_training.py")
            == "test-environments-training"
        )

    def test_flaky_test(self):
        """A test in FLAKY_TESTS should go to test-flaky."""
        assert select_tests.assign_test_to_job("source/isaaclab/test/test_logger.py") == "test-flaky"

    def test_slightly_flaky_test(self):
        """A test in SLIGHTLY_FLAKY_TESTS should go to test-slightly-flaky."""
        assert (
            select_tests.assign_test_to_job("source/isaaclab_physx/test/assets/test_surface_gripper.py")
            == "test-slightly-flaky"
        )

    def test_curobo_test(self):
        """A test in CUROBO_TESTS should go to test-curobo."""
        assert (
            select_tests.assign_test_to_job("source/isaaclab_tasks/test/test_environments_skillgen.py") == "test-curobo"
        )

    def test_duplicate_basename_different_packages(self):
        """Duplicate basenames in different packages should go to different jobs."""
        physx_job = select_tests.assign_test_to_job("source/isaaclab_physx/test/assets/test_articulation.py")
        newton_job = select_tests.assign_test_to_job("source/isaaclab_newton/test/assets/test_articulation.py")
        assert physx_job == "test-physx"
        assert newton_job == "test-newton"


class TestSelectTests:
    """End-to-end tests for the select_tests function."""

    def _make_mapping(self, tmp_path, source_to_tests, age_days=0):
        """Helper to create a mapping file with given content and age."""
        generated = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
        mapping = {
            "metadata": {
                "generated_at": generated,
                "commit": "abc123",
                "test_file_count": sum(len(v) for v in source_to_tests.values()),
                "source_file_count": len(source_to_tests),
            },
            "source_to_tests": source_to_tests,
        }
        path = tmp_path / "map.json"
        path.write_text(json.dumps(mapping))
        return str(path)

    def test_mapped_source_selects_tests(self, tmp_path):
        """Changing a mapped source file should select its tests."""
        mapping_path = self._make_mapping(
            tmp_path,
            {
                "source/isaaclab/isaaclab/utils/math.py": [
                    "source/isaaclab/test/utils/test_math.py",
                    "source/isaaclab_physx/test/assets/test_articulation.py",
                ],
            },
        )
        changed = [("M", "source/isaaclab/isaaclab/utils/math.py")]

        result = select_tests.select_tests(mapping_path, changed, max_age_days=7)

        assert result["run_all"] is False
        assert "source/isaaclab/test/utils/test_math.py" in result["jobs"]["test-general"]
        assert "source/isaaclab_physx/test/assets/test_articulation.py" in result["jobs"]["test-physx"]

    def test_unmapped_source_triggers_fallback(self, tmp_path):
        """Changing an unmapped source file should trigger run_all."""
        mapping_path = self._make_mapping(
            tmp_path,
            {
                "source/isaaclab/isaaclab/utils/math.py": ["source/isaaclab/test/utils/test_math.py"],
            },
        )
        changed = [("M", "source/isaaclab/isaaclab/new_module.py")]

        result = select_tests.select_tests(mapping_path, changed, max_age_days=7)

        assert result["run_all"] is True

    def test_modified_test_file_is_always_included(self, tmp_path):
        """Modifying a test file directly should include it."""
        mapping_path = self._make_mapping(tmp_path, {})
        changed = [("M", "source/isaaclab/test/utils/test_math.py")]

        result = select_tests.select_tests(mapping_path, changed, max_age_days=7)

        assert result["run_all"] is False
        assert "source/isaaclab/test/utils/test_math.py" in result["jobs"]["test-general"]

    def test_non_python_file_triggers_fallback(self, tmp_path):
        """Changing a non-Python file under source/ should trigger run_all."""
        mapping_path = self._make_mapping(tmp_path, {})
        changed = [("M", "source/isaaclab/test/utils/test_config.yaml")]

        result = select_tests.select_tests(mapping_path, changed, max_age_days=7)

        assert result["run_all"] is True

    def test_docs_only_change(self, tmp_path):
        """Changing only docs files should select no tests and not trigger run_all."""
        mapping_path = self._make_mapping(tmp_path, {})
        changed = [("M", "docs/README.md")]

        result = select_tests.select_tests(mapping_path, changed, max_age_days=7)

        assert result["run_all"] is False
        assert all(v == "" for v in result["jobs"].values())

    def test_infrastructure_file_triggers_fallback(self, tmp_path):
        """Changing tools/conftest.py should trigger run_all."""
        mapping_path = self._make_mapping(tmp_path, {})
        changed = [("M", "tools/conftest.py")]

        result = select_tests.select_tests(mapping_path, changed, max_age_days=7)

        assert result["run_all"] is True

    def test_deleted_source_triggers_fallback(self, tmp_path):
        """Deleting a source file should trigger run_all."""
        mapping_path = self._make_mapping(tmp_path, {})
        changed = [("D", "source/isaaclab/isaaclab/old_module.py")]

        result = select_tests.select_tests(mapping_path, changed, max_age_days=7)

        assert result["run_all"] is True

    def test_renamed_file_triggers_fallback(self, tmp_path):
        """Renaming a source file should trigger run_all."""
        mapping_path = self._make_mapping(tmp_path, {})
        changed = [("R", "source/isaaclab/isaaclab/renamed.py")]

        result = select_tests.select_tests(mapping_path, changed, max_age_days=7)

        assert result["run_all"] is True

    def test_stale_mapping_triggers_fallback(self, tmp_path):
        """A mapping older than max_age_days should trigger run_all."""
        mapping_path = self._make_mapping(
            tmp_path,
            {"source/isaaclab/isaaclab/utils/math.py": ["source/isaaclab/test/utils/test_math.py"]},
            age_days=10,
        )
        changed = [("M", "source/isaaclab/isaaclab/utils/math.py")]

        result = select_tests.select_tests(mapping_path, changed, max_age_days=7)

        assert result["run_all"] is True

    def test_missing_mapping_triggers_fallback(self, tmp_path):
        """A missing mapping file should trigger run_all."""
        result = select_tests.select_tests(str(tmp_path / "nope.json"), [("M", "source/x.py")], max_age_days=7)
        assert result["run_all"] is True

    def test_apps_file_triggers_fallback(self, tmp_path):
        """Changing a file under apps/ should trigger run_all."""
        mapping_path = self._make_mapping(tmp_path, {})
        changed = [("M", "apps/something.py")]

        result = select_tests.select_tests(mapping_path, changed, max_age_days=7)

        assert result["run_all"] is True

    def test_mixed_mapped_and_ignored(self, tmp_path):
        """Mapped source + ignored docs should select only the mapped tests, no fallback."""
        mapping_path = self._make_mapping(
            tmp_path,
            {
                "source/isaaclab/isaaclab/utils/math.py": ["source/isaaclab/test/utils/test_math.py"],
            },
        )
        changed = [
            ("M", "source/isaaclab/isaaclab/utils/math.py"),
            ("M", "docs/README.md"),
        ]

        result = select_tests.select_tests(mapping_path, changed, max_age_days=7)

        assert result["run_all"] is False
        assert "source/isaaclab/test/utils/test_math.py" in result["jobs"]["test-general"]

    def test_ci_tooling_change_triggers_fallback(self, tmp_path):
        """Changing tools/select_tests.py should trigger run_all."""
        mapping_path = self._make_mapping(tmp_path, {})
        changed = [("M", "tools/select_tests.py")]

        result = select_tests.select_tests(mapping_path, changed, max_age_days=7)

        assert result["run_all"] is True

    def test_job_assignment_in_output(self, tmp_path):
        """Tests from different packages should appear under their respective jobs."""
        mapping_path = self._make_mapping(
            tmp_path,
            {
                "source/isaaclab/isaaclab/utils/math.py": [
                    "source/isaaclab/test/utils/test_math.py",
                    "source/isaaclab_physx/test/assets/test_rigid_object.py",
                    "source/isaaclab_newton/test/assets/test_rigid_object.py",
                ],
            },
        )
        changed = [("M", "source/isaaclab/isaaclab/utils/math.py")]

        result = select_tests.select_tests(mapping_path, changed, max_age_days=7)

        assert result["run_all"] is False
        assert "source/isaaclab/test/utils/test_math.py" in result["jobs"]["test-general"]
        assert "source/isaaclab_physx/test/assets/test_rigid_object.py" in result["jobs"]["test-physx"]
        assert "source/isaaclab_newton/test/assets/test_rigid_object.py" in result["jobs"]["test-newton"]

    def test_github_change_triggers_fallback(self, tmp_path):
        """Changing a .github/ file should trigger run_all."""
        mapping_path = self._make_mapping(tmp_path, {})
        changed = [("M", ".github/workflows/build.yaml")]

        result = select_tests.select_tests(mapping_path, changed, max_age_days=7)

        assert result["run_all"] is True

    def test_docker_change_triggers_fallback(self, tmp_path):
        """Changing a docker/ file should trigger run_all."""
        mapping_path = self._make_mapping(tmp_path, {})
        changed = [("M", "docker/Dockerfile.base")]

        result = select_tests.select_tests(mapping_path, changed, max_age_days=7)

        assert result["run_all"] is True


class TestMatchesIncludeFiles:
    """Tests for _matches_include_files in conftest.py."""

    def test_full_path_match(self):
        """A full-path entry should match only that exact path suffix."""
        include = {"source/isaaclab_physx/test/assets/test_articulation.py"}
        assert _matches_include_files(
            "source/isaaclab_physx/test/assets/test_articulation.py", "test_articulation.py", include
        )

    def test_full_path_no_cross_package_match(self):
        """A full-path entry for physx should NOT match newton."""
        include = {"source/isaaclab_physx/test/assets/test_articulation.py"}
        assert not _matches_include_files(
            "source/isaaclab_newton/test/assets/test_articulation.py", "test_articulation.py", include
        )

    def test_basename_match(self):
        """A basename-only entry should match any file with that basename."""
        include = {"test_articulation.py"}
        assert _matches_include_files(
            "source/isaaclab_physx/test/assets/test_articulation.py", "test_articulation.py", include
        )
        assert _matches_include_files(
            "source/isaaclab_newton/test/assets/test_articulation.py", "test_articulation.py", include
        )

    def test_no_match(self):
        """A file not in the include set should not match."""
        include = {"source/isaaclab_physx/test/assets/test_articulation.py"}
        assert not _matches_include_files("source/isaaclab/test/utils/test_math.py", "test_math.py", include)

    def test_empty_include_files(self):
        """An empty include set should match nothing."""
        assert not _matches_include_files("source/isaaclab/test/utils/test_math.py", "test_math.py", set())


class TestParseGitDiff:
    """Tests for parsing git diff --name-status output."""

    def test_parse_modified_file(self):
        """Standard modified file line should be parsed."""
        lines = ["M\tsource/isaaclab/isaaclab/utils/math.py"]
        result = select_tests.parse_git_diff_output(lines)
        assert result == [("M", "source/isaaclab/isaaclab/utils/math.py")]

    def test_parse_added_file(self):
        """Added file line should be parsed."""
        lines = ["A\tsource/isaaclab/isaaclab/new.py"]
        result = select_tests.parse_git_diff_output(lines)
        assert result == [("A", "source/isaaclab/isaaclab/new.py")]

    def test_parse_deleted_file(self):
        """Deleted file line should be parsed."""
        lines = ["D\tsource/isaaclab/isaaclab/old.py"]
        result = select_tests.parse_git_diff_output(lines)
        assert result == [("D", "source/isaaclab/isaaclab/old.py")]

    def test_parse_renamed_file(self):
        """Renamed file line (Rxx) should be parsed, using the new path."""
        lines = ["R100\tsource/old.py\tsource/new.py"]
        result = select_tests.parse_git_diff_output(lines)
        assert result == [("R", "source/new.py")]

    def test_parse_multiple_lines(self):
        """Multiple lines should all be parsed."""
        lines = [
            "M\tsource/a.py",
            "A\tsource/b.py",
            "D\tdocs/c.md",
        ]
        result = select_tests.parse_git_diff_output(lines)
        assert len(result) == 3

    def test_parse_empty_lines_ignored(self):
        """Empty lines should be skipped."""
        lines = ["M\tsource/a.py", "", "A\tsource/b.py"]
        result = select_tests.parse_git_diff_output(lines)
        assert len(result) == 2
