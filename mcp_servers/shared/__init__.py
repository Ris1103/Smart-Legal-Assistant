"""Shared utilities for MCP servers."""
import pathlib
import sys


def add_app_to_path() -> None:
    """Insert app/ into sys.path so MCP servers can import from it."""
    app_dir = str(pathlib.Path(__file__).resolve().parents[2] / "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
