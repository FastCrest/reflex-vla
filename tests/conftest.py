"""Pytest fixtures shared across the test suite.

Sets a wide terminal width so Rich/Click don't elide option names in
``--help`` output. CI environments (GitHub Actions Linux runners)
default to a width where Rich abbreviates long flag names to ``--...``,
which breaks substring assertions like ``"--export-mode" in result.output``.
Setting COLUMNS=200 at session start makes the rendered help stable
across local (Mac terminal) and CI (no-TTY Linux runner).
"""
from __future__ import annotations

import os


def pytest_configure(config):
    os.environ["COLUMNS"] = "200"
    os.environ["LINES"] = "80"
