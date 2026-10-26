"""
career_data.py
--------------
Utility helpers for loading and caching the career dataset,
and for building the skill / interest / work-style picklists
used by the Streamlit UI.
"""

import pathlib
from functools import lru_cache
from typing import Optional

import pandas as pd

from recommender import _parse_field, load_dataset

_DEFAULT_PATH = (
    pathlib.Path(__file__).resolve().parent
    / "data" / "processed" / "career_dataset.csv"
)


@lru_cache(maxsize=1)
def get_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load and cache the career dataset (singleton per path)."""
    p = pathlib.Path(path) if path else _DEFAULT_PATH
    return load_dataset(p)


def all_skills(df: pd.DataFrame) -> list[str]:
    """Return a sorted, deduplicated list of all essential skills."""
    skills: set[str] = set()
    for val in df["essential_skills"].dropna():
        for s in _parse_field(str(val)):
            if s:
                skills.add(s.title())
    return sorted(skills)


def all_software(df: pd.DataFrame) -> list[str]:
    """Return a sorted, deduplicated list of all software technologies."""
    items: set[str] = set()
    for val in df["software_skills"].dropna():
        for s in _parse_field(str(val)):
            if s:
                items.add(s.title())
    return sorted(items)


def all_interests(df: pd.DataFrame) -> list[str]:
    """Return a sorted, deduplicated list of all career interests."""
    items: set[str] = set()
    for val in df["career_interests"].dropna():
        for s in _parse_field(str(val)):
            if s:
                items.add(s.title())
    return sorted(items)


def all_work_styles(df: pd.DataFrame) -> list[str]:
    """Return a sorted, deduplicated list of all work styles."""
    items: set[str] = set()
    for val in df["work_styles"].dropna():
        for s in _parse_field(str(val)):
            if s:
                items.add(s.title())
    return sorted(items)


def careers_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame for the dataset info panel."""
    summary = df[["occupation_code", "occupation_title"]].copy()
    summary["skills_count"] = df["essential_skills"].apply(
        lambda v: len(_parse_field(str(v)))
    )
    summary["software_count"] = df["software_skills"].apply(
        lambda v: len(_parse_field(str(v)))
    )
    return summary
