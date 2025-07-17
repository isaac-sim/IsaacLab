# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test cases for Cosmos prompt generation script."""

import json
import os
import tempfile

import pytest

from scripts.tools.cosmos.cosmos_prompt_gen import generate_prompt, main


@pytest.fixture(scope="class")
def temp_templates_file():
    """Create temporary templates file."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)

    # Create test templates
    test_templates = {
        "lighting": ["with bright lighting", "with dim lighting", "with natural lighting"],
        "color": ["in warm colors", "in cool colors", "in vibrant colors"],
        "style": ["in a realistic style", "in an artistic style", "in a minimalist style"],
        "empty_section": [],  # Test empty section
        "invalid_section": "not a list",  # Test invalid section
    }

    # Write templates to file
    with open(temp_file.name, "w") as f:
        json.dump(test_templates, f)

    yield temp_file.name
    # Cleanup
    os.remove(temp_file.name)


@pytest.fixture
def temp_output_file():
    """Create temporary output file."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    yield temp_file.name
    # Cleanup
    os.remove(temp_file.name)


class TestCosmosPromptGen:
    """Test cases for Cosmos prompt generation functionality."""

    def test_generate_prompt_valid_templates(self, temp_templates_file):
        """Test generating a prompt with valid templates."""
        prompt = generate_prompt(temp_templates_file)

        # Check that prompt is a string
        assert isinstance(prompt, str)

        # Check that prompt contains at least one word
        assert len(prompt.split()) > 0

        # Check that prompt contains valid sections
        valid_sections = ["lighting", "color", "style"]
        found_sections = [section for section in valid_sections if section in prompt.lower()]
        assert len(found_sections) > 0

    def test_generate_prompt_invalid_file(self):
        """Test generating a prompt with invalid file path."""
        with pytest.raises(FileNotFoundError):
            generate_prompt("nonexistent_file.json")

    def test_generate_prompt_invalid_json(self):
        """Test generating a prompt with invalid JSON file."""
        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_file.write(b"invalid json content")
            temp_file.flush()

            try:
                with pytest.raises(ValueError):
                    generate_prompt(temp_file.name)
            finally:
                os.remove(temp_file.name)

    def test_main_function_single_prompt(self, temp_templates_file, temp_output_file):
        """Test main function with single prompt generation."""
        # Mock command line arguments
        import sys

        original_argv = sys.argv
        sys.argv = [
            "cosmos_prompt_gen.py",
            "--templates_path",
            temp_templates_file,
            "--num_prompts",
            "1",
            "--output_path",
            temp_output_file,
        ]

        try:
            main()

            # Check if output file was created
            assert os.path.exists(temp_output_file)

            # Check content of output file
            with open(temp_output_file) as f:
                content = f.read().strip()
                assert len(content) > 0
                assert len(content.split("\n")) == 1
        finally:
            # Restore original argv
            sys.argv = original_argv

    def test_main_function_multiple_prompts(self, temp_templates_file, temp_output_file):
        """Test main function with multiple prompt generation."""
        # Mock command line arguments
        import sys

        original_argv = sys.argv
        sys.argv = [
            "cosmos_prompt_gen.py",
            "--templates_path",
            temp_templates_file,
            "--num_prompts",
            "3",
            "--output_path",
            temp_output_file,
        ]

        try:
            main()

            # Check if output file was created
            assert os.path.exists(temp_output_file)

            # Check content of output file
            with open(temp_output_file) as f:
                content = f.read().strip()
                assert len(content) > 0
                assert len(content.split("\n")) == 3

                # Check that each line is a valid prompt
                for line in content.split("\n"):
                    assert len(line) > 0
        finally:
            # Restore original argv
            sys.argv = original_argv

    def test_main_function_default_output(self, temp_templates_file):
        """Test main function with default output path."""
        # Mock command line arguments
        import sys

        original_argv = sys.argv
        sys.argv = ["cosmos_prompt_gen.py", "--templates_path", temp_templates_file, "--num_prompts", "1"]

        try:
            main()

            # Check if default output file was created
            assert os.path.exists("prompts.txt")

            # Clean up default output file
            os.remove("prompts.txt")
        finally:
            # Restore original argv
            sys.argv = original_argv
