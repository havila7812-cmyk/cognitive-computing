"""
conftest.py — adds the project root to sys.path so that
recommender.py, career_data.py, etc. are importable from tests/.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
