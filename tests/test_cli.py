"""Tests for the CliParser module."""

from __future__ import annotations

from pathlib import Path

import pytest

from visagism.cli import CliParser


class TestCliParser:
    """Test suite for CliParser.parse()."""

    def test_parse_basic_input(self) -> None:
        """Test parsing with only required --input argument."""
        config = CliParser.parse(["--input", "photo.jpg"])
        assert config.input_path == Path("photo.jpg")
        assert config.output_dir == Path("output")
        assert config.model_path is None
        assert config.visualize is False
        assert config.save_viz is False

    def test_parse_short_form(self) -> None:
        """Test parsing with short form -i."""
        config = CliParser.parse(["-i", "photo.jpg"])
        assert config.input_path == Path("photo.jpg")

    def test_parse_all_options(self) -> None:
        """Test parsing with all options specified."""
        config = CliParser.parse([
            "--input", "photo.jpg",
            "--output", "results",
            "--model", "/path/to/model.dat",
            "--visualize",
            "--save-viz",
        ])
        assert config.input_path == Path("photo.jpg")
        assert config.output_dir == Path("results")
        assert config.model_path == Path("/path/to/model.dat")
        assert config.visualize is True
        assert config.save_viz is True

    def test_parse_output_default(self) -> None:
        """Test that output defaults to 'output'."""
        config = CliParser.parse(["--input", "photo.jpg"])
        assert config.output_dir == Path("output")

    def test_parse_model_optional(self) -> None:
        """Test that --model is optional and defaults to None."""
        config = CliParser.parse(["--input", "photo.jpg"])
        assert config.model_path is None

        config = CliParser.parse(["--input", "photo.jpg", "--model", "custom.dat"])
        assert config.model_path == Path("custom.dat")

    def test_parse_visualize_flag(self) -> None:
        """Test --visualize flag."""
        config = CliParser.parse(["--input", "photo.jpg", "--visualize"])
        assert config.visualize is True

    def test_parse_save_viz_flag(self) -> None:
        """Test --save-viz flag."""
        config = CliParser.parse(["--input", "photo.jpg", "--save-viz"])
        assert config.save_viz is True

    def test_parse_missing_input(self) -> None:
        """Test that missing --input raises SystemExit."""
        with pytest.raises(SystemExit):
            CliParser.parse([])

    def test_parse_unknown_argument(self) -> None:
        """Test that unknown arguments raise SystemExit."""
        with pytest.raises(SystemExit):
            CliParser.parse(["--input", "photo.jpg", "--unknown-flag"])

    def test_parse_input_path_resolution(self) -> None:
        """Test that input_path preserves relative paths."""
        config = CliParser.parse(["--input", "../photos/test.png"])
        assert config.input_path == Path("../photos/test.png")
