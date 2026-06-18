"""
conftest.py — shared pytest config for LineSight tests.
Adds the linesight source directory to sys.path automatically.
"""
import sys
import os

# Make the package importable from tests/ without installing it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
