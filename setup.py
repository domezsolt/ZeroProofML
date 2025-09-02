"""Setup script for backwards compatibility.

This file is provided for compatibility with older tools that don't support
pyproject.toml. The actual build configuration is in pyproject.toml.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
