"""
test_recommender.py
-------------------
Pytest test suite for the Explainable Career Recommendation System.

Tests cover:
  - Dataset loading and validation
  - Occupation-code join correctness
  - Duplicate and missing data handling
  - Parsing semicolon-separated fields
  - Skill, interest and work-style matching
  - Score bounds (0–100)
  - Descending ranking order
  - Deterministic results
  - Different profiles producing different rankings
  - Missing optional inputs
  - Empty profile validation
  - Accurate explanation content
  - Skill-gap (unreported skills) labelling
  - Missing dataset error
  - Malformed CSV error handling
"""

import pathlib
import tempfile
import textwrap
import os

import pandas as pd
import pytest

from recommender import (
    CareerMatch,
    StudentProfile,
    _apply_aliases,
    _match_count,
    _normalize,
    _parse_field,
    _score_match,
    load_dataset,
    recommend,
    validate_profile,
)

# ---------------------------------------------------------------------------
# Fixture: minimal synthetic dataset (no real CSV required for unit tests)
# ---------------------------------------------------------------------------

SAMPLE_CSV = textwrap.dedent("""\
occupation_code,occupation_title,description,essential_skills,career_interests,work_styles,software_skills
15-1252.00,Software Developers,Design and develop software applications.,Programming;Critical Thinking;Systems Analysis;Active Listening,Investigative;Realistic,Analytical Thinking;Attention to Detail,Python;Git;Docker
15-1212.00,Information Security Analysts,Protect computer networks from cyber attacks.,Systems Analysis;Critical Thinking;Active Listening;Programming,Investigative;Conventional,Analytical Thinking;Integrity,Wireshark;Splunk;Python
15-2051.00,Data Scientists,Use data to build predictive models.,Critical Thinking;Mathematics;Programming;Statistics,Investigative;Artistic,Analytical Thinking;Initiative,Python;R;SQL;Jupyter
15-1254.00,Web Developers,Build and maintain websites.,Programming;Active Listening;Writing,Artistic;Realistic,Attention to Detail;Cooperation,JavaScript;HTML;CSS;React
15-1242.00,Database Administrators,Manage and maintain database systems.,Systems Analysis;Critical Thinking;Programming,Conventional;Investigative,Analytical Thinking;Dependability,SQL;Oracle;PostgreSQL
""")


@pytest.fixture()
def sample_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write the sample CSV and return its path."""
    p = tmp_path / "career_dataset.csv"
    p.write_text(SAMPLE_CSV, encoding="utf-8")
    return p


@pytest.fixture()
def sample_df(sample_csv: pathlib.Path) -> pd.DataFrame:
    return load_dataset(sample_csv)


# ===========================================================================
# 1. Dataset loading and validation
# ===========================================================================

class TestDatasetLoading:
    def test_load_returns_dataframe(self, sample_df):
        assert isinstance(sample_df, pd.DataFrame)

    def test_required_columns_present(self, sample_df):
        for col in ["occupation_code", "occupation_title"]:
            assert col in sample_df.columns

    def test_row_count(self, sample_df):
        assert len(sample_df) == 5

    def test_optional_columns_filled(self, sample_df):
        for col in ["essential_skills", "career_interests", "work_styles", "software_skills"]:
            assert col in sample_df.columns
            assert sample_df[col].isna().sum() == 0

    def test_missing_dataset_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.csv")

    def test_malformed_csv_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        # No required columns
        bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required columns"):
            load_dataset(bad)


# ===========================================================================
# 2. Occupation-code join correctness
# ===========================================================================

class TestOccupationCodeJoin:
    def test_codes_are_preserved(self, sample_df):
        expected_codes = {
            "15-1252.00", "15-1212.00", "15-2051.00",
            "15-1254.00", "15-1242.00",
        }
        # normalize comparison: strip trailing .00
        actual = set(sample_df["occupation_code"].astype(str).str.strip())
        # at minimum the normalized prefixes should match
        for code in expected_codes:
            assert any(code in c or c in code for c in actual), \
                f"Expected code {code} not found in dataset"

    def test_no_join_by_position(self, sample_df):
        """Shuffle rows — titles must still match their codes."""
        shuffled = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)
        for _, row in shuffled.iterrows():
            code = str(row["occupation_code"])
            title = str(row["occupation_title"])
            assert "Software Developers" in title if "1252" in code else True
            assert "Data Scientists" in title if "2051" in code else True


# ===========================================================================
# 3. Duplicate and missing data handling
# ===========================================================================

class TestDuplicateAndMissing:
    def test_load_handles_duplicate_rows(self, tmp_path):
        dup_csv = SAMPLE_CSV + "15-1252.00,Software Developers,Duplicate row.,,,,"
        p = tmp_path / "dup.csv"
        p.write_text(dup_csv, encoding="utf-8")
        df = load_dataset(p)
        # Should load without error (prepare_dataset deduplicates; load_dataset just reads)
        assert len(df) >= 5

    def test_missing_text_fields_filled(self, tmp_path):
        partial = (
            "occupation_code,occupation_title,description,essential_skills,"
            "career_interests,work_styles,software_skills\n"
            "15-1252.00,Software Developers,Desc.,,,,\n"
        )
        p = tmp_path / "partial.csv"
        p.write_text(partial, encoding="utf-8")
        df = load_dataset(p)
        assert df.iloc[0]["essential_skills"] == ""


# ===========================================================================
# 4. Parsing semicolon-separated fields
# ===========================================================================

class TestParseField:
    def test_basic(self):
        result = _parse_field("Python;JavaScript;SQL")
        assert result == ["python", "javascript", "sql"]

    def test_deduplication(self):
        result = _parse_field("Python;python;PYTHON")
        assert len(result) == 1

    def test_empty_string(self):
        assert _parse_field("") == []

    def test_none_value(self):
        assert _parse_field(None) == []

    def test_whitespace_trimmed(self):
        result = _parse_field("  Python ;  SQL ")
        assert result == ["python", "sql"]

    def test_single_item(self):
        assert _parse_field("Programming") == ["programming"]


# ===========================================================================
# 5. Skill normalization and aliases
# ===========================================================================

class TestNormalize:
    def test_lowercase(self):
        assert _normalize("Python Programming") == "python programming"

    def test_strips_whitespace(self):
        assert _normalize("  SQL  ") == "sql"

    def test_collapses_spaces(self):
        assert _normalize("Critical   Thinking") == "critical thinking"


class TestAliases:
    def test_alias_py_to_python(self):
        result = _apply_aliases(["py"])
        assert "python" in result

    def test_alias_js_to_javascript(self):
        result = _apply_aliases(["js"])
        assert "javascript" in result

    def test_unknown_term_unchanged(self):
        result = _apply_aliases(["blockchain"])
        assert "blockchain" in result


# ===========================================================================
# 6. Match count logic
# ===========================================================================

class TestMatchCount:
    def test_exact_match(self):
        matched, unmatched = _match_count(["python"], ["python"])
        assert "python" in matched
        assert len(unmatched) == 0

    def test_substring_match(self):
        # student says "python", career has "python programming"
        matched, _ = _match_count(["python"], ["python programming"])
        assert len(matched) > 0

    def test_no_match(self):
        matched, unmatched = _match_count(["golang"], ["python", "java"])
        assert len(matched) == 0
        assert len(unmatched) == 2

    def test_empty_student(self):
        matched, unmatched = _match_count([], ["python"])
        assert len(matched) == 0
        assert unmatched == ["python"]

    def test_empty_career(self):
        matched, unmatched = _match_count(["python"], [])
        assert len(matched) == 0


# ===========================================================================
# 7. Score bounds
# ===========================================================================

class TestScoreBounds:
    def test_score_match_max(self):
        score = _score_match(["a", "b"], 2)
        assert score == 100.0

    def test_score_match_zero(self):
        score = _score_match([], 5)
        assert score == 0.0

    def test_score_match_partial(self):
        score = _score_match(["a"], 4)
        assert 0.0 < score < 100.0

    def test_score_match_empty_total(self):
        score = _score_match([], 0)
        assert score == 0.0

    def test_overall_scores_in_range(self, sample_df):
        profile = StudentProfile(
            technical_skills=["programming", "critical thinking"],
            career_interests=["investigative"],
            work_styles=["analytical thinking"],
        )
        results = recommend(profile, dataset=sample_df)
        for m in results:
            assert 0.0 <= m.overall_score <= 100.0
            assert 0.0 <= m.skill_score <= 100.0
            assert 0.0 <= m.interest_score <= 100.0
            assert 0.0 <= m.style_score <= 100.0


# ===========================================================================
# 8. Ranking order (descending)
# ===========================================================================

class TestRankingOrder:
    def test_descending_order(self, sample_df):
        profile = StudentProfile(
            technical_skills=["programming", "critical thinking", "systems analysis"],
            career_interests=["investigative"],
        )
        results = recommend(profile, dataset=sample_df)
        scores = [m.overall_score for m in results]
        assert scores == sorted(scores, reverse=True)

    def test_rank_numbers_sequential(self, sample_df):
        profile = StudentProfile(technical_skills=["programming"])
        results = recommend(profile, dataset=sample_df)
        ranks = [m.rank for m in results]
        assert ranks == list(range(1, len(results) + 1))


# ===========================================================================
# 9. Deterministic results
# ===========================================================================

class TestDeterminism:
    def test_same_profile_same_results(self, sample_df):
        profile = StudentProfile(
            technical_skills=["programming", "mathematics"],
            career_interests=["investigative"],
            work_styles=["analytical thinking"],
        )
        r1 = recommend(profile, dataset=sample_df)
        r2 = recommend(profile, dataset=sample_df)
        assert [m.occupation_code for m in r1] == [m.occupation_code for m in r2]
        assert [m.overall_score for m in r1] == [m.overall_score for m in r2]


# ===========================================================================
# 10. Different profiles produce different rankings
# ===========================================================================

class TestProfileDifferences:
    def test_data_scientist_profile(self, sample_df):
        profile = StudentProfile(
            technical_skills=["programming", "mathematics", "statistics", "critical thinking"],
            software_skills=["python", "r", "sql"],
            career_interests=["investigative", "artistic"],
        )
        results = recommend(profile, dataset=sample_df)
        assert results[0].occupation_code == "15-2051.00"  # Data Scientists

    def test_web_developer_profile(self, sample_df):
        profile = StudentProfile(
            technical_skills=["programming", "active listening", "writing"],
            software_skills=["javascript", "html", "css", "react"],
            career_interests=["artistic", "realistic"],
        )
        results = recommend(profile, dataset=sample_df)
        top_titles = [m.occupation_title for m in results[:3]]
        assert any("Web" in t for t in top_titles)

    def test_security_profile(self, sample_df):
        profile = StudentProfile(
            technical_skills=["systems analysis", "critical thinking", "active listening"],
            software_skills=["wireshark", "splunk"],
            career_interests=["investigative", "conventional"],
        )
        results = recommend(profile, dataset=sample_df)
        top_titles = [m.occupation_title for m in results[:2]]
        assert any("Security" in t for t in top_titles)


# ===========================================================================
# 11. Missing optional inputs
# ===========================================================================

class TestMissingOptionalInputs:
    def test_no_software_skills(self, sample_df):
        profile = StudentProfile(
            technical_skills=["programming"],
            career_interests=["investigative"],
        )
        results = recommend(profile, dataset=sample_df)
        assert len(results) > 0

    def test_no_interests(self, sample_df):
        profile = StudentProfile(
            technical_skills=["programming", "critical thinking"],
        )
        results = recommend(profile, dataset=sample_df)
        assert len(results) > 0

    def test_no_styles(self, sample_df):
        profile = StudentProfile(
            technical_skills=["programming"],
            career_interests=["investigative"],
        )
        results = recommend(profile, dataset=sample_df)
        assert len(results) > 0

    def test_education_ignored_in_scoring(self, sample_df):
        """Education is optional metadata; not changing the score."""
        profile1 = StudentProfile(technical_skills=["programming"], education_level=None)
        profile2 = StudentProfile(technical_skills=["programming"], education_level="Master's")
        r1 = recommend(profile1, dataset=sample_df)
        r2 = recommend(profile2, dataset=sample_df)
        assert [m.overall_score for m in r1] == [m.overall_score for m in r2]


# ===========================================================================
# 12. Empty profile validation
# ===========================================================================

class TestEmptyProfileValidation:
    def test_empty_profile_returns_errors(self):
        profile = StudentProfile()
        errors = validate_profile(profile)
        assert len(errors) > 0

    def test_empty_profile_recommend_returns_empty(self, sample_df):
        profile = StudentProfile()
        results = recommend(profile, dataset=sample_df)
        assert results == []

    def test_whitespace_only_profile_fails(self):
        profile = StudentProfile(technical_skills=["   ", "  "])
        errors = validate_profile(profile)
        assert len(errors) > 0


# ===========================================================================
# 13. Accurate matching explanations
# ===========================================================================

class TestExplanations:
    def test_explanation_present(self, sample_df):
        profile = StudentProfile(technical_skills=["programming"])
        results = recommend(profile, dataset=sample_df)
        for m in results:
            assert m.explanation.strip() != ""

    def test_explanation_contains_score(self, sample_df):
        profile = StudentProfile(technical_skills=["programming"])
        results = recommend(profile, dataset=sample_df)
        assert str(int(results[0].overall_score)) in results[0].explanation \
            or f"{results[0].overall_score:.1f}" in results[0].explanation

    def test_explanation_contains_match_disclaimer(self, sample_df):
        profile = StudentProfile(technical_skills=["programming"])
        results = recommend(profile, dataset=sample_df)
        assert "profile match" in results[0].explanation.lower()

    def test_matched_skills_in_explanation(self, sample_df):
        profile = StudentProfile(technical_skills=["programming"])
        results = recommend(profile, dataset=sample_df)
        top = results[0]
        if top.matched_skills:
            assert "matched" in top.explanation.lower()


# ===========================================================================
# 14. Skill-gap / unreported skills
# ===========================================================================

class TestSkillGaps:
    def test_unreported_skills_are_present(self, sample_df):
        profile = StudentProfile(technical_skills=["programming"])
        results = recommend(profile, dataset=sample_df)
        # At least some occupations should have unreported skills
        has_gaps = any(len(m.unreported_skills) > 0 for m in results)
        assert has_gaps

    def test_unreported_not_in_matched(self, sample_df):
        profile = StudentProfile(technical_skills=["programming"])
        results = recommend(profile, dataset=sample_df)
        for m in results:
            overlap = set(m.matched_skills) & set(m.unreported_skills)
            assert len(overlap) == 0, \
                f"Skills appear in both matched and unreported: {overlap}"

    def test_suggested_skills_subset_of_unreported(self, sample_df):
        profile = StudentProfile(technical_skills=["programming"])
        results = recommend(profile, dataset=sample_df)
        for m in results:
            for s in m.suggested_skills:
                assert s in m.unreported_skills

    def test_unreported_labeled_not_confirmed_absent(self, sample_df):
        """The explanation must NOT say 'you definitely lack'."""
        profile = StudentProfile(technical_skills=["programming"])
        results = recommend(profile, dataset=sample_df)
        for m in results:
            assert "definitely lack" not in m.explanation.lower()
            # Should say 'not yet reported' when gaps exist
            if m.unreported_skills:
                assert "not yet reported" in m.explanation.lower()
