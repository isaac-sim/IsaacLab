# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test cases for Cosmos prompt generation script."""

import json
import os
import tempfile
import unittest

from scripts.tools.cosmos.cosmos_prompt_gen import generate_prompt, main


class TestCosmosPromptGen(unittest.TestCase):
    """Test cases for Cosmos prompt generation functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all test methods."""
        # Create temporary templates file
        cls.temp_templates_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)

        # Create test templates
        test_templates = {
            "lighting": ["with bright lighting", "with dim lighting", "with natural lighting"],
            "color": ["in warm colors", "in cool colors", "in vibrant colors"],
            "style": ["in a realistic style", "in an artistic style", "in a minimalist style"],
            "empty_section": [],  # Test empty section
            "invalid_section": "not a list",  # Test invalid section
        }

        # Write templates to file
        with open(cls.temp_templates_file.name, "w") as f:
            json.dump(test_templates, f)

    def setUp(self):
        """Set up test fixtures that are created for each test method."""
        self.temp_output_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove the temporary output file
        os.remove(self.temp_output_file.name)

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures that are shared across all test methods."""
        # Remove the temporary templates file
        os.remove(cls.temp_templates_file.name)

    def test_generate_prompt_valid_templates(self):
        """Test generating a prompt with valid templates."""
        prompt = generate_prompt(self.temp_templates_file.name)

        # Check that prompt is a string
        self.assertIsInstance(prompt, str)

        # Check that prompt contains at least one word
        self.assertTrue(len(prompt.split()) > 0)

        # Check that prompt contains valid sections
        valid_sections = ["lighting", "color", "style"]
        found_sections = [section for section in valid_sections if section in prompt.lower()]
        self.assertTrue(len(found_sections) > 0)

    def test_generate_prompt_invalid_file(self):
        """Test generating a prompt with invalid file path."""
        with self.assertRaises(FileNotFoundError):
            generate_prompt("nonexistent_file.json")

    def test_generate_prompt_invalid_json(self):
        """Test generating a prompt with invalid JSON file."""
        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_file.write(b"invalid json content")
            temp_file.flush()

            try:
                with self.assertRaises(ValueError):
                    generate_prompt(temp_file.name)
            finally:
                os.remove(temp_file.name)

    def test_main_function_single_prompt(self):
        """Test main function with single prompt generation."""
        # Mock command line arguments
        import sys

        original_argv = sys.argv
        sys.argv = [
            "cosmos_prompt_gen.py",
            "--templates_path",
            self.temp_templates_file.name,
            "--num_prompts",
            "1",
            "--output_path",
            self.temp_output_file.name,
        ]

        try:
            main()

            # Check if output file was created
            self.assertTrue(os.path.exists(self.temp_output_file.name))

            # Check content of output file
            with open(self.temp_output_file.name) as f:
                content = f.read().strip()
                self.assertTrue(len(content) > 0)
                self.assertEqual(len(content.split("\n")), 1)
        finally:
            # Restore original argv
            sys.argv = original_argv

    def test_main_function_multiple_prompts(self):
        """Test main function with multiple prompt generation."""
        # Mock command line arguments
        import sys

        original_argv = sys.argv
        sys.argv = [
            "cosmos_prompt_gen.py",
            "--templates_path",
            self.temp_templates_file.name,
            "--num_prompts",
            "3",
            "--output_path",
            self.temp_output_file.name,
        ]

        try:
            main()

            # Check if output file was created
            self.assertTrue(os.path.exists(self.temp_output_file.name))

            # Check content of output file
            with open(self.temp_output_file.name) as f:
                content = f.read().strip()
                self.assertTrue(len(content) > 0)
                self.assertEqual(len(content.split("\n")), 3)

                # Check that each line is a valid prompt
                for line in content.split("\n"):
                    self.assertTrue(len(line) > 0)
        finally:
            # Restore original argv
            sys.argv = original_argv

    def test_main_function_default_output(self):
        """Test main function with default output path."""
        # Mock command line arguments
        import sys

        original_argv = sys.argv
        sys.argv = ["cosmos_prompt_gen.py", "--templates_path", self.temp_templates_file.name, "--num_prompts", "1"]

        try:
            main()

            # Check if default output file was created
            self.assertTrue(os.path.exists("prompts.txt"))

            # Clean up default output file
            os.remove("prompts.txt")
        finally:
            # Restore original argv
            sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()
