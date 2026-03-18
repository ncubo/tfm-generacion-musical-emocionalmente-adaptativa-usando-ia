"""
wide_to_long_survey.py
======================
Transforms the perceptual survey Excel (Google Forms / Excel export) from wide
format (one row per participant, repeated column groups) to long format (one row
per participant × audio stimulus).

Usage
-----
    python wide_to_long_survey.py

Outputs (same folder as this script)
--------------------------------------
    survey_long.xlsx
    survey_long.csv

Configuration
-------------
If the input filename changes, update INPUT_FILE below.
If the survey gains more or fewer audio blocks, no other change is needed —
the script detects the number of blocks automatically from the column suffixes.
"""

import re
import sys
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION — the only section you should need to edit
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent

# Path to the raw Excel file exported from Google Forms
INPUT_FILE = BASE_DIR / "version_1.xlsx"

# Output paths (will be created / overwritten in the same folder)
OUTPUT_XLSX = BASE_DIR / "survey_long.xlsx"
OUTPUT_CSV  = BASE_DIR / "survey_long.csv"

# ── Keyword fragments used to locate each repeated column group ────────────
# The script searches for these substrings (case-insensitive) in the base
# column name (i.e. after stripping the .1 / .2 … numeric suffix).
KEYWORD_MODELO       = "motor"
KEYWORD_VAL_REAL     = "valencia"
KEYWORD_AROUS_REAL   = "activacion"
KEYWORD_VAL_RESP     = "positiva o negativa"   # part of the question text
KEYWORD_AROUS_RESP   = "nivel de energ"        # part of the question text

# ── Demographic column keywords ────────────────────────────────────────────
# Map: desired output column name → substring present in the original column name
DEMO_KEYWORDS: dict[str, str] = {
    "genero":               "género",
    "edad":                 "edad",
    "conocimiento_musical": "formación musical",
}

# ---------------------------------------------------------------------------
# INTERNAL HELPERS  (no need to edit below this line)
# ---------------------------------------------------------------------------

# Regex to strip the Excel-style numeric suffix (.1, .2, .10, etc.)
_SUFFIX_RE = re.compile(r"^(.*?)\.(\d+)$", re.DOTALL)


def _strip_suffix(col: str) -> tuple[str, int]:
    """Return (base_name, block_index).  Columns without a suffix → index 0."""
    m = _SUFFIX_RE.match(col)
    if m:
        return m.group(1).strip(), int(m.group(2))
    return col.strip(), 0


def _classify_columns(columns) -> tuple[list[str], dict[str, dict[int, str]]]:
    """
    Split all DataFrame columns into two groups:

    demo_cols   – original column names that appear only once (no suffix twin)
    block_map   – { base_name: { block_index: original_col_name } }
                  only contains bases that appear in ≥ 2 suffix variants
    """
    groups: dict[str, dict[int, str]] = {}
    for col in columns:
        base, idx = _strip_suffix(col)
        groups.setdefault(base, {})[idx] = col

    demo_cols = [list(v.values())[0] for base, v in groups.items() if len(v) == 1]
    block_map = {base: v for base, v in groups.items() if len(v) > 1}
    return demo_cols, block_map


def _find_demo_col(demo_cols: list[str], keyword: str) -> str:
    """Return the first demographic column whose name contains keyword."""
    kw = keyword.lower()
    for col in demo_cols:
        if kw in col.lower():
            return col
    raise ValueError(
        f"No single-occurrence column found containing '{keyword}'.\n"
        f"Available demographic columns: {demo_cols}"
    )


def _find_block_base(block_map: dict, keyword: str) -> str:
    """
    Return the block group whose base name contains keyword (case-insensitive).
    Raises ValueError if none or more than one match is found.
    """
    kw = keyword.lower()
    matches = [b for b in block_map if kw in b.lower()]
    if not matches:
        raise ValueError(
            f"No repeated column group found containing '{keyword}'.\n"
            f"Available repeated-column bases: {list(block_map.keys())}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple repeated column groups match '{keyword}': {matches}.\n"
            "Refine the keyword in the CONFIGURATION section."
        )
    return matches[0]


def _safe_float(value) -> float:
    """Convert a cell value to float; return NaN on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _normalize_text(value) -> str:
    """Lowercase + strip a text value; return empty string for missing values."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


# ---------------------------------------------------------------------------
# MAIN TRANSFORMATION
# ---------------------------------------------------------------------------

def build_long_format(input_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read the wide-format Excel and return (df_raw, df_long).

    df_long columns
    ---------------
    participante_id, audio_id, modelo, valencia_real, arousal_real,
    valencia_resp, arousal_resp, valencia_binaria, arousal_binario,
    genero, edad, conocimiento_musical
    """
    # ── 1. Load ────────────────────────────────────────────────────────────
    df_raw = pd.read_excel(input_file)
    print(f"  Raw shape: {df_raw.shape}  "
          f"({df_raw.shape[0]} participants × {df_raw.shape[1]} columns)")

    # ── 2. Classify columns ────────────────────────────────────────────────
    demo_cols, block_map = _classify_columns(df_raw.columns)
    n_blocks = max(len(v) for v in block_map.values())
    print(f"  Audio blocks detected: {n_blocks}")

    # ── 3. Map demographic keywords → original column names ───────────────
    demo_col_map: dict[str, str] = {}
    for out_name, keyword in DEMO_KEYWORDS.items():
        col = _find_demo_col(demo_cols, keyword)
        demo_col_map[out_name] = col
        print(f"  Demographic '{out_name}' → '{col}'")

    # ── 4. Map block keywords → base names in block_map ───────────────────
    modelo_base      = _find_block_base(block_map, KEYWORD_MODELO)
    val_real_base    = _find_block_base(block_map, KEYWORD_VAL_REAL)
    arous_real_base  = _find_block_base(block_map, KEYWORD_AROUS_REAL)
    val_resp_base    = _find_block_base(block_map, KEYWORD_VAL_RESP)
    arous_resp_base  = _find_block_base(block_map, KEYWORD_AROUS_RESP)

    print(f"  Block bases detected:")
    print(f"    modelo       → '{modelo_base}'")
    print(f"    valencia_real → '{val_real_base}'")
    print(f"    arousal_real  → '{arous_real_base}'")
    print(f"    valencia_resp → '{val_resp_base}'")
    print(f"    arousal_resp  → '{arous_resp_base}'")

    # ── 5. Build long-format records ───────────────────────────────────────
    block_indices = sorted(block_map[modelo_base].keys())
    records = []

    for part_idx, row in df_raw.iterrows():
        participante_id = int(part_idx) + 1  # 1-based

        # Demographic values (carried into every audio row)
        demo_vals = {
            out_col: row[orig_col]
            for out_col, orig_col in demo_col_map.items()
        }

        for block_idx in block_indices:
            # Resolve each column for this block
            def col(base: str) -> str:
                return block_map[base][block_idx]

            # ── Text fields: lowercase + strip ────────────────────────────
            modelo_val     = _normalize_text(row[col(modelo_base)])

            # valencia_real / arousal_real: keep numeric if numeric,
            # otherwise normalize as text
            val_real_raw   = row[col(val_real_base)]
            arous_real_raw = row[col(arous_real_base)]

            if pd.api.types.is_numeric_dtype(type(val_real_raw)):
                valencia_real = val_real_raw if not pd.isna(val_real_raw) else float("nan")
            else:
                valencia_real = _normalize_text(val_real_raw)

            if pd.api.types.is_numeric_dtype(type(arous_real_raw)):
                arousal_real = arous_real_raw if not pd.isna(arous_real_raw) else float("nan")
            else:
                arousal_real = _normalize_text(arous_real_raw)

            # ── Numeric response fields (1–7 Likert scale) ────────────────
            val_resp   = _safe_float(row[col(val_resp_base)])
            arous_resp = _safe_float(row[col(arous_resp_base)])

            # ── Derived binary columns ────────────────────────────────────
            if pd.isna(val_resp):
                val_bin = pd.NA
            else:
                val_bin = "positiva" if val_resp >= 4 else "negativa"

            if pd.isna(arous_resp):
                arous_bin = pd.NA
            else:
                arous_bin = "alto" if arous_resp >= 4 else "bajo"

            records.append({
                "participante_id":   participante_id,
                "audio_id":          block_idx + 1,   # 1-based audio index
                "modelo":            modelo_val,
                "valencia_real":     valencia_real,
                "arousal_real":      arousal_real,
                "valencia_resp":     val_resp,
                "arousal_resp":      arous_resp,
                "valencia_binaria":  val_bin,
                "arousal_binario":   arous_bin,
                **demo_vals,
            })

    # ── 6. Assemble DataFrame with explicit column order ───────────────────
    col_order = [
        "participante_id",
        "audio_id",
        "modelo",
        "valencia_real",
        "arousal_real",
        "valencia_resp",
        "arousal_resp",
        "valencia_binaria",
        "arousal_binario",
        "genero",
        "edad",
        "conocimiento_musical",
    ]
    df_long = pd.DataFrame(records)[col_order]
    return df_raw, df_long


# ---------------------------------------------------------------------------
# EXPORT + VALIDATION
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\nReading:  {INPUT_FILE}")
    if not INPUT_FILE.exists():
        print(f"ERROR: File not found → {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    df_raw, df_long = build_long_format(INPUT_FILE)

    # ── Export ─────────────────────────────────────────────────────────────
    df_long.to_excel(OUTPUT_XLSX, index=False)
    df_long.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nExported:")
    print(f"  XLSX → {OUTPUT_XLSX}")
    print(f"  CSV  → {OUTPUT_CSV}")

    # ── Validation summary ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Original shape   : {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    print(f"Transformed shape: {df_long.shape[0]} rows × {df_long.shape[1]} columns")
    print(f"  (= {df_raw.shape[0]} participants × "
          f"{df_long['audio_id'].nunique()} audio blocks)")

    print("\nUnique models:")
    print(df_long["modelo"].value_counts().to_string())

    print("\nRows per audio_id:")
    print(df_long["audio_id"].value_counts().sort_index().to_string())

    print("\nMissing values per column:")
    missing = df_long.isnull().sum()
    missing_pct = (missing / len(df_long) * 100).round(1)
    report = pd.concat([missing, missing_pct], axis=1)
    report.columns = ["count", "%"]
    print(report.to_string())
    print("=" * 50)


if __name__ == "__main__":
    main()
