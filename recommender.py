"""
recommender.py
--------------
Rule-based career recommendation engine for the Explainable Career
Recommendation System.

Cognitive Computing concepts implemented:
  - Knowledge Representation: structured O*NET occupational dataset
  - Reasoning: weighted rule-based comparisons
  - Explainability: detailed match reasons for every score
  - Personalization: rankings change with profile inputs
  - Decision Support: skill-gap analysis and learning suggestions

NOTE: Scores represent a *profile match percentage*, NOT a probability
      of employment or success in a career.
"""

import pathlib
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Dataset path
# ---------------------------------------------------------------------------
_DEFAULT_DATASET = (
    pathlib.Path(__file__).resolve().parent
    / "data" / "processed" / "career_dataset.csv"
)

# ---------------------------------------------------------------------------
# Scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHT_SKILLS    = 0.60   # technical + software skill match
WEIGHT_INTERESTS = 0.30   # career interest match
WEIGHT_STYLES    = 0.10   # work-style match

# ---------------------------------------------------------------------------
# Skill alias mapping
# (normalised source -> canonical term used in dataset)
# Only semantically equivalent terms are mapped here.
# ---------------------------------------------------------------------------
SKILL_ALIASES: dict[str, str] = {
    # Programming languages
    "python programming": "python",
    "py":                 "python",
    "javascript":         "javascript",
    "js":                 "javascript",
    "typescript":         "typescript",
    "ts":                 "typescript",
    "c++":                "c++",
    "cpp":                "c++",
    "c#":                 "c#",
    "csharp":             "c#",
    # Common skill synonyms
    "ml":                 "machine learning",
    "ai":                 "artificial intelligence",
    "nlp":                "natural language processing",
    "cv":                 "computer vision",
    "oop":                "object-oriented programming",
    "sql server":         "microsoft sql server",
    "mssql":              "microsoft sql server",
    "postgres":           "postgresql",
    "mongo":              "mongodb",
    "k8s":                "kubernetes",
    "aws":                "amazon web services",
    "gcp":                "google cloud platform",
    "azure":              "microsoft azure",
    "ui":                 "user interface design",
    "ux":                 "user experience design",
    "qa":                 "quality assurance",
    "sec":                "information security",
    "infosec":            "information security",
    "devops":             "devops",
    "ci/cd":              "continuous integration",
    "cicd":               "continuous integration",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StudentProfile:
    """Represents a student's self-reported profile."""
    technical_skills: list[str] = field(default_factory=list)
    software_skills:  list[str] = field(default_factory=list)
    career_interests: list[str] = field(default_factory=list)
    work_styles:      list[str] = field(default_factory=list)
    education_level:  Optional[str] = None
    career_goals:     Optional[str] = None


@dataclass
class CareerMatch:
    """Result record for one occupation."""
    occupation_code:  str
    occupation_title: str
    description:      str

    overall_score:    float  # 0-100
    skill_score:      float  # 0-100
    interest_score:   float  # 0-100
    style_score:      float  # 0-100

    matched_skills:        list[str] = field(default_factory=list)
    matched_interests:     list[str] = field(default_factory=list)
    matched_styles:        list[str] = field(default_factory=list)

    unreported_skills:     list[str] = field(default_factory=list)   # not yet reported
    suggested_skills:      list[str] = field(default_factory=list)   # top skills to learn

    explanation:           str = ""
    rank:                  int = 0


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip, collapse multiple spaces."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _apply_aliases(tokens: list[str]) -> list[str]:
    """Expand aliases so student inputs map to canonical terms."""
    result = []
    for t in tokens:
        n = _normalize(t)
        result.append(SKILL_ALIASES.get(n, n))
    return result


def _parse_field(value: str) -> list[str]:
    """
    Parse a semicolon-separated field from the dataset into a list of
    normalized strings, deduplicating and removing blanks.
    """
    if not value or not isinstance(value, str):
        return []
    parts = [_normalize(p) for p in value.split(";") if p.strip()]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _match_count(student_items: list[str], career_items: list[str]) -> tuple[list[str], list[str]]:
    """
    Return (matched, unmatched_career_items).
    Matching is substring-aware: a student item matches if it appears
    as a substring of a career item OR vice versa.
    """
    matched: list[str] = []
    unmatched: list[str] = []

    for ci in career_items:
        ci_n = _normalize(ci)
        found = False
        for si in student_items:
            si_n = _normalize(si)
            if si_n and (si_n in ci_n or ci_n in si_n):
                if ci not in matched:
                    matched.append(ci)
                found = True
                break
        if not found:
            unmatched.append(ci)

    return matched, unmatched


def _score_match(matched: list[str], total: int) -> float:
    """Return a 0-100 match percentage."""
    if total == 0:
        return 0.0
    return round(min(100.0, 100.0 * len(matched) / total), 2)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(path: Optional[pathlib.Path] = None) -> pd.DataFrame:
    """
    Load and validate the career dataset.
    Raises FileNotFoundError if the CSV does not exist.
    Raises ValueError if required columns are missing.
    """
    dataset_path = path or _DEFAULT_DATASET

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Career dataset not found: {dataset_path}\n"
            "Run `python scripts/prepare_dataset.py` first."
        )

    try:
        df = pd.read_csv(dataset_path, encoding="utf-8", low_memory=False)
    except Exception as exc:
        raise ValueError(f"Cannot read dataset CSV: {exc}") from exc

    required = ["occupation_code", "occupation_title"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Ensure text columns exist (fill blanks if absent)
    for col in ["description", "essential_skills", "career_interests",
                "work_styles", "software_skills"]:
        if col not in df.columns:
            df[col] = ""
        # Cast to object first to avoid Pandas 2.x dtype mismatch warning
        df[col] = df[col].astype(object).fillna("").astype(str)

    return df


# ---------------------------------------------------------------------------
# Core recommendation engine
# ---------------------------------------------------------------------------

def recommend(
    profile: StudentProfile,
    dataset: Optional[pd.DataFrame] = None,
    dataset_path: Optional[pathlib.Path] = None,
    top_n: int = 10,
) -> list[CareerMatch]:
    """
    Generate ranked career recommendations for a StudentProfile.

    Returns a list of CareerMatch objects sorted by overall_score
    descending, then occupation_title ascending (for tie-breaking).

    If the profile contains no usable matching information, returns an
    empty list and the caller should prompt the student for more input.
    """
    # ---- load data ----
    if dataset is None:
        dataset = load_dataset(dataset_path)

    # ---- normalize profile inputs ----
    tech_skills = _apply_aliases(
        [_normalize(s) for s in profile.technical_skills if s.strip()]
    )
    soft_skills = _apply_aliases(
        [_normalize(s) for s in profile.software_skills if s.strip()]
    )
    all_student_skills = list(set(tech_skills + soft_skills))

    interests = [_normalize(i) for i in profile.career_interests if i.strip()]
    styles    = [_normalize(s) for s in profile.work_styles     if s.strip()]

    # ---- determine which categories are available ----
    has_skills    = len(all_student_skills) > 0
    has_interests = len(interests) > 0
    has_styles    = len(styles) > 0

    # ---- renormalize weights if some categories are absent ----
    raw_weights = {
        "skills":    WEIGHT_SKILLS    if has_skills    else 0.0,
        "interests": WEIGHT_INTERESTS if has_interests else 0.0,
        "styles":    WEIGHT_STYLES    if has_styles    else 0.0,
    }
    total_weight = sum(raw_weights.values())

    if total_weight == 0:
        # No usable inputs at all
        return []

    weights = {k: v / total_weight for k, v in raw_weights.items()}

    # ---- score each occupation ----
    results: list[CareerMatch] = []

    for _, row in dataset.iterrows():
        occ_code  = str(row["occupation_code"])
        occ_title = str(row["occupation_title"])
        occ_desc  = str(row["description"])

        career_skills   = _parse_field(str(row.get("essential_skills", "")))
        career_software = _parse_field(str(row.get("software_skills", "")))
        career_interests_list = _parse_field(str(row.get("career_interests", "")))
        career_styles   = _parse_field(str(row.get("work_styles", "")))

        # Combined career skill set (technical + software)
        all_career_skills = career_skills + career_software

        # ---- skill matching ----
        if has_skills and all_career_skills:
            matched_sk, unmatched_sk = _match_count(all_student_skills, all_career_skills)
            skill_score = _score_match(matched_sk, len(all_career_skills))
        elif has_skills and not all_career_skills:
            # Career has no skill data — treat as 0, not a perfect match
            matched_sk, unmatched_sk = [], []
            skill_score = 0.0
        else:
            matched_sk, unmatched_sk = [], list(career_skills)
            skill_score = 0.0

        # ---- interest matching ----
        if has_interests and career_interests_list:
            matched_int, _ = _match_count(interests, career_interests_list)
            interest_score = _score_match(matched_int, len(career_interests_list))
        elif has_interests and not career_interests_list:
            matched_int = []
            interest_score = 0.0
        else:
            matched_int = []
            interest_score = 0.0

        # ---- work-style matching ----
        if has_styles and career_styles:
            matched_sty, _ = _match_count(styles, career_styles)
            style_score = _score_match(matched_sty, len(career_styles))
        elif has_styles and not career_styles:
            matched_sty = []
            style_score = 0.0
        else:
            matched_sty = []
            style_score = 0.0

        # ---- combined weighted score ----
        overall = round(
            weights["skills"]    * skill_score +
            weights["interests"] * interest_score +
            weights["styles"]    * style_score,
            2,
        )

        # ---- skill gap: not-yet-reported skills ----
        # Only list career skills not covered by the student; we label
        # them as "not yet reported", not "definitely lacking".
        unreported = [s for s in unmatched_sk if s not in matched_sk]
        # Suggest the top 5 unreported skills as "learn next"
        suggested  = unreported[:5]

        # ---- human-readable explanation ----
        explanation = _build_explanation(
            occ_title, overall, skill_score, interest_score, style_score,
            matched_sk, matched_int, matched_sty,
            unreported, weights, has_skills, has_interests, has_styles,
        )

        results.append(CareerMatch(
            occupation_code=occ_code,
            occupation_title=occ_title,
            description=occ_desc,
            overall_score=overall,
            skill_score=skill_score,
            interest_score=interest_score,
            style_score=style_score,
            matched_skills=matched_sk,
            matched_interests=matched_int,
            matched_styles=matched_sty,
            unreported_skills=unreported,
            suggested_skills=suggested,
            explanation=explanation,
        ))

    # ---- sort: descending score, then ascending title for determinism ----
    results.sort(key=lambda m: (-m.overall_score, m.occupation_title))
    for i, m in enumerate(results, 1):
        m.rank = i

    return results[:top_n]


# ---------------------------------------------------------------------------
# Explanation builder
# ---------------------------------------------------------------------------

def _build_explanation(
    title: str,
    overall: float,
    skill_score: float,
    interest_score: float,
    style_score: float,
    matched_skills: list[str],
    matched_interests: list[str],
    matched_styles: list[str],
    unreported: list[str],
    weights: dict,
    has_skills: bool,
    has_interests: bool,
    has_styles: bool,
) -> str:
    lines = [
        f"Profile match for '{title}': {overall:.1f}/100",
        "",
        "Scoring breakdown:",
    ]

    if has_skills:
        lines.append(
            f"  Skills (weight {weights['skills']*100:.0f}%): {skill_score:.1f}/100"
        )
        if matched_skills:
            preview = ", ".join(matched_skills[:5])
            if len(matched_skills) > 5:
                preview += f" (+{len(matched_skills)-5} more)"
            lines.append(f"    Matched: {preview}")
        else:
            lines.append("    No matching skills found in profile.")

    if has_interests:
        lines.append(
            f"  Interests (weight {weights['interests']*100:.0f}%): {interest_score:.1f}/100"
        )
        if matched_interests:
            lines.append(f"    Matched: {', '.join(matched_interests)}")
        else:
            lines.append("    No matching interests found.")

    if has_styles:
        lines.append(
            f"  Work Styles (weight {weights['styles']*100:.0f}%): {style_score:.1f}/100"
        )
        if matched_styles:
            preview = ", ".join(matched_styles[:3])
            lines.append(f"    Matched: {preview}")
        else:
            lines.append("    No matching work styles found.")

    if unreported:
        preview = ", ".join(unreported[:5])
        if len(unreported) > 5:
            preview += f" (+{len(unreported)-5} more)"
        lines.append(f"\nSkills not yet reported: {preview}")
        lines.append("  (These are listed for awareness, not as a confirmed deficit.)")

    lines.append(
        "\nNote: This score represents a profile match percentage, "
        "not a probability of employment or career success."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profile validation
# ---------------------------------------------------------------------------

def validate_profile(profile: StudentProfile) -> list[str]:
    """
    Return a list of validation messages.  Empty list = profile is usable.
    """
    messages: list[str] = []
    all_inputs = (
        profile.technical_skills
        + profile.software_skills
        + profile.career_interests
        + profile.work_styles
    )
    non_empty = [s for s in all_inputs if s.strip()]
    if not non_empty:
        messages.append(
            "Your profile has no matching information. "
            "Please provide at least one skill, interest, or work style."
        )
    return messages
