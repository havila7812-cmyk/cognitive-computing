"""
prepare_dataset.py
------------------
Reads the five O*NET CSV source files, inspects their schemas,
joins them on the O*NET-SOC occupation code, filters to IT
occupations and writes the cleaned combined dataset to
data/processed/career_dataset.csv.

Source: O*NET 31.0 Database, National Center for O*NET Development
        https://www.onetcenter.org/database.html

Run:
    python scripts/prepare_dataset.py
"""

import pathlib
import shutil
import sys
import textwrap

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV   = PROCESSED_DIR / "career_dataset.csv"
DATA_DICT    = PROCESSED_DIR / "data_dictionary.md"

SOURCE_DIR = pathlib.Path(r"C:\Users\Havila\Downloads\csv_files")

# ---------------------------------------------------------------------------
# Target IT occupation codes (O*NET-SOC)
# We list a broad set; we'll keep whatever is actually present in the data.
# ---------------------------------------------------------------------------
TARGET_SOC_PREFIXES = [
    "15-1211",  # Computer Systems Analysts
    "15-1212",  # Information Security Analysts
    "15-1221",  # Computer and Information Research Scientists
    "15-1231",  # Computer Network Support Specialists
    "15-1241",  # Computer Network Architects
    "15-1242",  # Database Administrators
    "15-1243",  # Database Architects
    "15-1244",  # Network and Computer Systems Administrators
    "15-1251",  # Computer Programmers
    "15-1252",  # Software Developers
    "15-1253",  # Software Quality Assurance Analysts and Testers
    "15-1254",  # Web Developers
    "15-1255",  # Web and Digital Interface Designers
    "15-1299",  # Computer Occupations, All Other
    "15-2051",  # Data Scientists
    "15-2052",  # Data Analysts
]


def _read_csv(path: pathlib.Path, label: str) -> pd.DataFrame:
    """Read a CSV file, trying UTF-8 then latin-1."""
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"  [{label}] Read {len(df):,} rows | encoding={enc} | cols={list(df.columns)}")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot read {path} with utf-8 or latin-1")


def _copy_raw(src: pathlib.Path) -> None:
    """Copy source files to data/raw/ without modifying them."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dst = RAW_DIR / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        print(f"  Copied {src.name} -> data/raw/")
    else:
        print(f"  Already in data/raw/ -> {src.name}")


def _detect_occ_col(df: pd.DataFrame) -> str:
    """Return the occupation-code column name (flexible search)."""
    candidates = [c for c in df.columns
                  if any(kw in c.lower() for kw in ("o*net", "onet", "soc", "onetsoc"))]
    if candidates:
        return candidates[0]
    # fall back to first column that looks like a SOC code
    for c in df.columns:
        sample = df[c].dropna().astype(str).head(10)
        if sample.str.match(r"^\d{2}-\d{4}").any():
            return c
    raise KeyError(f"Cannot identify occupation-code column in: {list(df.columns)}")


def _normalize_soc(series: pd.Series) -> pd.Series:
    """Strip trailing '.00' or similar and keep XX-NNNN[.NN] format."""
    return series.astype(str).str.strip().str.replace(r"\.00$", "", regex=True)


def prepare() -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. Verify source directory
    # ------------------------------------------------------------------
    if not SOURCE_DIR.exists():
        print(f"\n[ERROR] Source folder not found: {SOURCE_DIR}")
        print("Please copy the five CSV files into data/raw/ and re-run.")
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Step 1: Inspecting source files ===")

    # ------------------------------------------------------------------
    # 2. Load all five files
    # ------------------------------------------------------------------
    occ_raw      = _read_csv(SOURCE_DIR / "occupation_data.csv",       "occupation_data")
    skills_raw   = _read_csv(SOURCE_DIR / "essential_skills.csv",      "essential_skills")
    interest_raw = _read_csv(SOURCE_DIR / "career_interest_types.csv", "career_interest_types")
    styles_raw   = _read_csv(SOURCE_DIR / "work_styles.csv",           "work_styles")
    software_raw = _read_csv(SOURCE_DIR / "software_skills.csv",       "software_skills")

    # ------------------------------------------------------------------
    # 3. Copy to data/raw/ (do not modify originals)
    # ------------------------------------------------------------------
    print("\n=== Step 2: Copying to data/raw/ ===")
    for fname in ["occupation_data.csv", "essential_skills.csv",
                  "career_interest_types.csv", "work_styles.csv", "software_skills.csv"]:
        _copy_raw(SOURCE_DIR / fname)

    # ------------------------------------------------------------------
    # 4. Identify occupation-code columns
    # ------------------------------------------------------------------
    print("\n=== Step 3: Detecting occupation-code columns ===")
    occ_col_occ   = _detect_occ_col(occ_raw)
    occ_col_skills = _detect_occ_col(skills_raw)
    occ_col_int   = _detect_occ_col(interest_raw)
    occ_col_styles = _detect_occ_col(styles_raw)
    occ_col_soft  = _detect_occ_col(software_raw)
    print(f"  occupation_data     -> '{occ_col_occ}'")
    print(f"  essential_skills    -> '{occ_col_skills}'")
    print(f"  career_interest     -> '{occ_col_int}'")
    print(f"  work_styles         -> '{occ_col_styles}'")
    print(f"  software_skills     -> '{occ_col_soft}'")

    # ------------------------------------------------------------------
    # 5. Normalize SOC codes
    # ------------------------------------------------------------------
    occ_raw[occ_col_occ]         = _normalize_soc(occ_raw[occ_col_occ])
    skills_raw[occ_col_skills]   = _normalize_soc(skills_raw[occ_col_skills])
    interest_raw[occ_col_int]    = _normalize_soc(interest_raw[occ_col_int])
    styles_raw[occ_col_styles]   = _normalize_soc(styles_raw[occ_col_styles])
    software_raw[occ_col_soft]   = _normalize_soc(software_raw[occ_col_soft])

    # ------------------------------------------------------------------
    # 6. Build occupation base table
    # ------------------------------------------------------------------
    print("\n=== Step 4: Building occupation base table ===")
    title_col = next((c for c in occ_raw.columns if "title" in c.lower()), None)
    desc_col  = next((c for c in occ_raw.columns if "description" in c.lower()), None)
    print(f"  Title col: {title_col!r}, Description col: {desc_col!r}")

    keep_cols = [occ_col_occ]
    if title_col:
        keep_cols.append(title_col)
    if desc_col:
        keep_cols.append(desc_col)

    occ = occ_raw[keep_cols].copy()
    rename_map = {occ_col_occ: "occupation_code"}
    if title_col:
        rename_map[title_col] = "occupation_title"
    if desc_col:
        rename_map[desc_col] = "description"
    occ = occ.rename(columns=rename_map)
    if "description" not in occ.columns:
        occ["description"] = ""
    if "occupation_title" not in occ.columns:
        occ["occupation_title"] = occ["occupation_code"]

    occ = occ.drop_duplicates(subset=["occupation_code"])
    print(f"  Unique occupations in source: {len(occ):,}")

    # ------------------------------------------------------------------
    # 7. Filter to target IT occupations
    # ------------------------------------------------------------------
    print("\n=== Step 5: Filtering to target IT occupations ===")
    pattern = "|".join(TARGET_SOC_PREFIXES)
    occ_it = occ[occ["occupation_code"].str.contains(pattern, regex=True)].copy()
    print(f"  Matched {len(occ_it)} IT occupations")
    if occ_it.empty:
        occ_it = occ[occ["occupation_code"].str.startswith("15-")].copy()
        print(f"  Fallback: kept all 15-XXXX codes -> {len(occ_it)} occupations")

    it_codes = set(occ_it["occupation_code"])

    # ------------------------------------------------------------------
    # 8. Aggregate essential skills
    # ------------------------------------------------------------------
    print("\n=== Step 6: Aggregating essential skills ===")
    sk = skills_raw[skills_raw[occ_col_skills].isin(it_codes)].copy()
    print(f"  Skills rows for IT occupations: {len(sk):,}")

    elem_col  = next((c for c in sk.columns if "element name" in c.lower()), None)
    scale_col = next((c for c in sk.columns if "scale" in c.lower() and "id" in c.lower()), None)
    val_col   = next((c for c in sk.columns
                      if "data value" in c.lower() or c.lower() == "value"), None)
    print(f"  elem={elem_col!r}, scale={scale_col!r}, val={val_col!r}")

    sk_im = sk.copy()
    if scale_col and "IM" in sk[scale_col].values:
        sk_im = sk[sk[scale_col] == "IM"].copy()

    if val_col and not sk_im.empty:
        sk_im[val_col] = pd.to_numeric(sk_im[val_col], errors="coerce")
        sk_im = sk_im[sk_im[val_col] >= 3].copy()

    if elem_col and not sk_im.empty:
        skills_agg = (
            sk_im.groupby(occ_col_skills)[elem_col]
            .apply(lambda x: ";".join(sorted(set(
                str(v).strip() for v in x
                if pd.notna(v) and str(v).strip()
            ))))
            .reset_index()
        )
        skills_agg.columns = ["occupation_code", "essential_skills"]
    else:
        skills_agg = pd.DataFrame({"occupation_code": list(it_codes), "essential_skills": [""] * len(it_codes)})

    print(f"  Aggregated skills for {len(skills_agg)} occupations")

    # ------------------------------------------------------------------
    # 9. Aggregate career interests
    # ------------------------------------------------------------------
    print("\n=== Step 7: Aggregating career interests ===")
    int_it = interest_raw[interest_raw[occ_col_int].isin(it_codes)].copy()
    print(f"  Interest rows for IT occupations: {len(int_it):,}")

    elem_col_int  = next((c for c in int_it.columns if "element name" in c.lower()), None)
    scale_col_int = next((c for c in int_it.columns if "scale" in c.lower() and "id" in c.lower()), None)
    val_col_int   = next((c for c in int_it.columns
                          if "data value" in c.lower() or c.lower() == "value"), None)
    print(f"  elem={elem_col_int!r}, scale={scale_col_int!r}, val={val_col_int!r}")

    int_oi = int_it.copy()
    if scale_col_int and "OI" in int_it[scale_col_int].values:
        int_oi = int_it[int_it[scale_col_int] == "OI"].copy()

    if elem_col_int and val_col_int and not int_oi.empty:
        int_oi[val_col_int] = pd.to_numeric(int_oi[val_col_int], errors="coerce")
        max_val = int_oi[val_col_int].max()
        threshold = 50 if max_val > 10 else 2
        int_filtered = int_oi[int_oi[val_col_int] >= threshold].copy()
        if int_filtered.empty:
            int_filtered = int_oi.copy()

        interest_agg = (
            int_filtered.groupby(occ_col_int)[elem_col_int]
            .apply(lambda x: ";".join(sorted(set(
                str(v).strip() for v in x
                if pd.notna(v) and str(v).strip()
            ))))
            .reset_index()
        )
        interest_agg.columns = ["occupation_code", "career_interests"]
    else:
        interest_agg = pd.DataFrame({"occupation_code": list(it_codes), "career_interests": [""] * len(it_codes)})

    print(f"  Aggregated interests for {len(interest_agg)} occupations")

    # ------------------------------------------------------------------
    # 10. Aggregate work styles
    # ------------------------------------------------------------------
    print("\n=== Step 8: Aggregating work styles ===")
    ws_it = styles_raw[styles_raw[occ_col_styles].isin(it_codes)].copy()
    print(f"  Work-style rows for IT occupations: {len(ws_it):,}")

    elem_col_ws = next((c for c in ws_it.columns if "element name" in c.lower()), None)
    val_col_ws  = next((c for c in ws_it.columns
                        if "data value" in c.lower() or c.lower() == "value"), None)
    scale_col_ws = next((c for c in ws_it.columns if "scale" in c.lower() and "id" in c.lower()), None)
    print(f"  elem={elem_col_ws!r}, scale={scale_col_ws!r}, val={val_col_ws!r}")

    ws_filtered = ws_it.copy()
    if val_col_ws and not ws_filtered.empty:
        ws_filtered[val_col_ws] = pd.to_numeric(ws_filtered[val_col_ws], errors="coerce")
        max_ws = ws_filtered[val_col_ws].max()
        ws_thresh = 3 if max_ws <= 5 else 60
        ws_filtered = ws_filtered[ws_filtered[val_col_ws] >= ws_thresh].copy()

    if elem_col_ws and not ws_filtered.empty:
        styles_agg = (
            ws_filtered.groupby(occ_col_styles)[elem_col_ws]
            .apply(lambda x: ";".join(sorted(set(
                str(v).strip() for v in x
                if pd.notna(v) and str(v).strip()
            ))))
            .reset_index()
        )
        styles_agg.columns = ["occupation_code", "work_styles"]
    else:
        styles_agg = pd.DataFrame({"occupation_code": list(it_codes), "work_styles": [""] * len(it_codes)})

    print(f"  Aggregated work styles for {len(styles_agg)} occupations")

    # ------------------------------------------------------------------
    # 11. Aggregate software skills
    # ------------------------------------------------------------------
    print("\n=== Step 9: Aggregating software skills ===")
    soft_it = software_raw[software_raw[occ_col_soft].isin(it_codes)].copy()
    print(f"  Software rows for IT occupations: {len(soft_it):,}")

    commodity_col  = next((c for c in soft_it.columns if "commodity" in c.lower()), None)
    example_col    = next((c for c in soft_it.columns if "example" in c.lower()), None)
    hot_col        = next((c for c in soft_it.columns if "hot" in c.lower()), None)
    title_col_soft = next((c for c in soft_it.columns
                           if "title" in c.lower() and c != occ_col_soft), None)
    print(f"  commodity={commodity_col!r}, example={example_col!r}, hot={hot_col!r}")

    tech_col = example_col or commodity_col or title_col_soft
    if tech_col and not soft_it.empty:
        if hot_col:
            hot_soft = soft_it[soft_it[hot_col].astype(str).str.strip().str.upper() == "Y"].copy()
            if len(hot_soft) < 5:
                hot_soft = soft_it.copy()
        else:
            hot_soft = soft_it.copy()

        software_agg = (
            hot_soft.groupby(occ_col_soft)[tech_col]
            .apply(lambda x: ";".join(sorted(set(
                str(v).strip() for v in x
                if pd.notna(v) and str(v).strip()
                and str(v).strip().lower() not in ("nan", "")
            ))))
            .reset_index()
        )
        software_agg.columns = ["occupation_code", "software_skills"]
    else:
        software_agg = pd.DataFrame({"occupation_code": list(it_codes), "software_skills": [""] * len(it_codes)})

    print(f"  Aggregated software skills for {len(software_agg)} occupations")

    # ------------------------------------------------------------------
    # 12. Join everything on occupation_code
    # ------------------------------------------------------------------
    print("\n=== Step 10: Joining datasets ===")
    combined = occ_it.copy()
    combined = combined.merge(skills_agg,   on="occupation_code", how="left")
    combined = combined.merge(interest_agg, on="occupation_code", how="left")
    combined = combined.merge(styles_agg,   on="occupation_code", how="left")
    combined = combined.merge(software_agg, on="occupation_code", how="left")

    for col in ["essential_skills", "career_interests", "work_styles", "software_skills"]:
        if col in combined.columns:
            combined[col] = combined[col].fillna("").astype(str)

    print(f"  Combined dataset: {len(combined)} rows, {combined.shape[1]} columns")

    # ------------------------------------------------------------------
    # 13. Quality checks
    # ------------------------------------------------------------------
    print("\n=== Step 11: Quality checks ===")
    dup_codes = combined["occupation_code"].duplicated().sum()
    print(f"  Duplicate occupation codes: {dup_codes}")
    combined = combined.drop_duplicates(subset=["occupation_code"])

    missing = combined.isna().sum()
    print("  Missing values per column:")
    for col, cnt in missing.items():
        if cnt > 0:
            print(f"    {col}: {cnt}")

    empty_skills = (combined["essential_skills"].str.strip() == "").sum()
    print(f"  Occupations with no essential skills: {empty_skills}")

    # ------------------------------------------------------------------
    # 14. Save output
    # ------------------------------------------------------------------
    print("\n=== Step 12: Saving output ===")
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"  Saved {len(combined)} occupations to: {OUTPUT_CSV}")

    # ------------------------------------------------------------------
    # 15. Print summary
    # ------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"  Source rows (occupation_data): {len(occ_raw):,}")
    print(f"  Unique codes in source:        {occ_raw[occ_col_occ].nunique():,}")
    print(f"  IT occupations retained:       {len(combined)}")
    print(f"  Duplicate codes removed:       {dup_codes}")
    print(f"  Output columns: {list(combined.columns)}")
    print()
    print("  Retained occupations:")
    for _, row in combined[["occupation_code", "occupation_title"]].iterrows():
        print(f"    {row['occupation_code']} | {row['occupation_title']}")

    _write_data_dictionary(combined)
    return combined


def _write_data_dictionary(df: pd.DataFrame) -> None:
    """Write a markdown data dictionary for the combined dataset."""
    text = textwrap.dedent(f"""\
    # Data Dictionary - career_dataset.csv

    **Source**: O*NET 31.0 Database, National Center for O*NET Development
    **URL**: https://www.onetcenter.org/database.html

    ## Column Descriptions

    | Column | Type | Description |
    |--------|------|-------------|
    | occupation_code | string | O*NET-SOC occupation code (e.g., 15-1252.00) |
    | occupation_title | string | Official O*NET occupation title |
    | description | string | Official O*NET occupation description |
    | essential_skills | string | Semicolon-separated essential skills (IM scale >= 3) |
    | career_interests | string | Semicolon-separated Holland interest codes (OI scale) |
    | work_styles | string | Semicolon-separated relevant work styles |
    | software_skills | string | Semicolon-separated software/tools technologies |

    ## Record Count

    Total occupations: **{len(df)}**

    ## Notes

    - O*NET data is primarily based on US occupational information and may need
      adaptation for students in other countries such as India.
    - Skill gap labels indicate skills *not yet reported* by the student; they do
      not confirm that the student lacks the skill.
    """)
    DATA_DICT.write_text(text, encoding="utf-8")
    print(f"  Data dictionary written to: {DATA_DICT}")


if __name__ == "__main__":
    prepare()
