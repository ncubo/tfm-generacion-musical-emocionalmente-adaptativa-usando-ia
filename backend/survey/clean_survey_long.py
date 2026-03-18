"""
clean_survey_long.py
====================
Takes the long-format survey dataset (survey_long.xlsx) produced by
wide_to_long_survey.py and applies a second round of cleaning and enrichment:

  1. Normalise the "modelo" column to canonical names.
  2. Create ground-truth categorical columns (valencia_real_cat, arousal_real_cat).
  3. Recompute binary response columns with a strict 3-class scheme that
     explicitly marks the midpoint (4) as neutral.
  4. Print a validation summary.
  5. Export to survey_long_cleaned.xlsx and survey_long_cleaned.csv.

All original columns are preserved — only new columns are added.

Usage
-----
    python clean_survey_long.py
"""

import sys
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent

INPUT_FILE   = BASE_DIR / "survey_long.xlsx"
OUTPUT_XLSX  = BASE_DIR / "survey_long_cleaned.xlsx"
OUTPUT_CSV   = BASE_DIR / "survey_long_cleaned.csv"

# Mapping from raw "modelo" strings → canonical names.
# Keys are matched after lowercasing + stripping the raw value.
MODELO_MAP: dict[str, str] = {
    "fine-tuned":  "finetuned",
    "transformer": "pretrained",
    "baseline":    "baseline",   # already correct; included for explicitness
}

# ---------------------------------------------------------------------------
# TRANSFORMATION FUNCTIONS
# ---------------------------------------------------------------------------

def normalise_modelo(series: pd.Series) -> pd.Series:
    """
    Lowercase + strip every value, then apply the canonical name mapping.
    Any value not present in MODELO_MAP is left as-is so unexpected values
    are visible rather than silently lost.
    """
    cleaned = series.str.strip().str.lower()
    return cleaned.replace(MODELO_MAP)


def make_valencia_real_cat(series: pd.Series) -> pd.Series:
    """
    Converts the continuous ground-truth valence (e.g. -0.8, 0.8) to a
    two-class categorical:
        ≥ 0  →  "positiva"
        < 0  →  "negativa"
    """
    return series.apply(
        lambda v: "positiva" if pd.notna(v) and v >= 0 else "negativa"
    )


def make_arousal_real_cat(series: pd.Series) -> pd.Series:
    """
    Converts the continuous ground-truth arousal (e.g. -0.8, 0.8) to a
    two-class categorical:
        ≥ 0  →  "alto"
        < 0  →  "bajo"
    """
    return series.apply(
        lambda v: "alto" if pd.notna(v) and v >= 0 else "bajo"
    )


def make_valencia_binaria_strict(series: pd.Series) -> pd.Series:
    """
    Converts the 1–7 Likert valence response to a three-class scheme that
    treats the midpoint explicitly as neutral instead of forcing it into a
    binary bucket:
        1–3  →  "negativa"
        4    →  "neutra"
        5–7  →  "positiva"
    NaN values are preserved.
    """
    def classify(v):
        if pd.isna(v):
            return pd.NA
        v = int(v)
        if v <= 3:
            return "negativa"
        if v >= 5:
            return "positiva"
        return "neutra"   # v == 4

    return series.apply(classify)


def make_arousal_binario_strict(series: pd.Series) -> pd.Series:
    """
    Same three-class scheme applied to the 1–7 Likert arousal response:
        1–3  →  "bajo"
        4    →  "neutro"
        5–7  →  "alto"
    NaN values are preserved.
    """
    def classify(v):
        if pd.isna(v):
            return pd.NA
        v = int(v)
        if v <= 3:
            return "bajo"
        if v >= 5:
            return "alto"
        return "neutro"   # v == 4

    return series.apply(classify)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Load ──────────────────────────────────────────────────────────────
    if not INPUT_FILE.exists():
        print(f"ERROR: input file not found → {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_excel(INPUT_FILE)
    print(f"Loaded: {INPUT_FILE}")
    print(f"  Shape: {df.shape}")

    # ── 1. Normalise "modelo" (overwrites the column with canonical names) ─
    df["modelo"] = normalise_modelo(df["modelo"])

    # ── 2. Ground-truth categorical columns ───────────────────────────────
    df["valencia_real_cat"] = make_valencia_real_cat(df["valencia_real"])
    df["arousal_real_cat"]  = make_arousal_real_cat(df["arousal_real"])

    # ── 3. Strict three-class binary response columns ─────────────────────
    df["valencia_binaria_strict"] = make_valencia_binaria_strict(df["valencia_resp"])
    df["arousal_binario_strict"]  = make_arousal_binario_strict(df["arousal_resp"])

    # ── 4. Validation summary ─────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)

    print("\nUnique values of 'modelo':")
    print(df["modelo"].value_counts().to_string())

    print("\nDistribution of 'valencia_real_cat':")
    print(df["valencia_real_cat"].value_counts().to_string())

    print("\nDistribution of 'arousal_real_cat':")
    print(df["arousal_real_cat"].value_counts().to_string())

    print("\nDistribution of 'valencia_binaria_strict':")
    print(df["valencia_binaria_strict"].value_counts().to_string())

    print("\nDistribution of 'arousal_binario_strict':")
    print(df["arousal_binario_strict"].value_counts().to_string())

    print("\nMissing values per column:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    print(pd.concat([missing, missing_pct], axis=1, keys=["count", "%"]).to_string())
    print("=" * 50)

    # ── 5. Export ─────────────────────────────────────────────────────────
    df.to_excel(OUTPUT_XLSX, index=False)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nExported:")
    print(f"  XLSX → {OUTPUT_XLSX}")
    print(f"  CSV  → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
