"""Pytest fixtures: imports bot.py with every heavy dependency stubbed out."""
from __future__ import annotations

import pytest

import stubs


@pytest.fixture(scope="session")
def bot():
    return stubs.import_bot()
