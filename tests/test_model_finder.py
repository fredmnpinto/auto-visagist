"""Tests for the ModelFinder module."""

from __future__ import annotations

from pathlib import Path

import pytest

from visagism.errors import ModelNotFoundError
from visagism.model_finder import ModelFinder


class TestModelFinder:
    """Test suite for ModelFinder.find()."""

    def test_find_with_override(self, fake_model_path: Path) -> None:
        """Test finding model via explicit override path."""
        found = ModelFinder.find(model_override=fake_model_path)
        assert found == fake_model_path.resolve()

    def test_find_with_env_var(
        self, fake_model_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test finding model via DLIB_MODEL_PATH env var."""
        monkeypatch.setenv("DLIB_MODEL_PATH", str(fake_model_path))
        found = ModelFinder.find()
        assert found == fake_model_path.resolve()

    def test_find_override_precedes_env(
        self, fake_model_path: Path, tmp_path: Path
    ) -> None:
        """Test that --model override takes precedence over env var."""
        env_path = tmp_path / "env_model.dat"
        env_path.touch()

        found = ModelFinder.find(model_override=fake_model_path)
        assert found == fake_model_path.resolve()

    def test_find_no_model_raises_error(self) -> None:
        """Test that missing model raises ModelNotFoundError."""
        with pytest.raises(ModelNotFoundError):
            ModelFinder.find()

    def test_find_returns_absolute_path(self, fake_model_path: Path) -> None:
        """Test that the returned path is absolute."""
        found = ModelFinder.find(model_override=fake_model_path)
        assert found.is_absolute()

    def test_find_non_existent_override(self) -> None:
        """Test that a non-existent override still searches other locations."""
        with pytest.raises(ModelNotFoundError):
            ModelFinder.find(model_override=Path("/nonexistent/model.dat"))

    def test_find_env_var_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that unset env var is handled gracefully."""
        monkeypatch.delenv("DLIB_MODEL_PATH", raising=False)
        with pytest.raises(ModelNotFoundError):
            ModelFinder.find()
