"""Shared test fixtures for MatKit tests."""

import pytest
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_cif(test_data_dir):
    """Path to a sample CIF file for testing."""
    return str(test_data_dir / "test_structure.cif")


@pytest.fixture
def tmp_output(tmp_path):
    """Temporary directory for test output."""
    return tmp_path
