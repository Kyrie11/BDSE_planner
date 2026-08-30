"""Compatibility shim for editable/legacy setuptools installs.

Package metadata lives in pyproject.toml.  Keeping this tiny setup.py supports
existing workflows that still run `pip install -e .` or invoke setuptools
through setup.py while avoiding a second, diverging dependency definition.
"""
from setuptools import setup


if __name__ == "__main__":
    setup()
