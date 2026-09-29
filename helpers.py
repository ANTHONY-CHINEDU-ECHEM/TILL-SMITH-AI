"""Shared fixtures for the test suite."""

from tillsmith_ai.engine import TillsmithEngine


def engine():
    return TillsmithEngine.shared(log=lambda *a: None)
