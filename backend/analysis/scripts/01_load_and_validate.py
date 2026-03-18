#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_load_and_validate.py
Load and validate the survey dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SURVEY_DATA = PROJECT_ROOT / "backend" / "survey" / "survey_long_cleaned.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "analysis" / "outputs"

def load_and_validate():
    """Load and validate the survey dataset"""
    
    print("="*60)
    print("LOADING AND VALIDATING SURVEY DATA")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from: {SURVEY_DATA}")
    df = pd.read_excel(SURVEY_DATA)
    
    print(f"✓ Data loaded successfully")
    print(f"  Shape: {df.shape}")
    
    # Check columns
    expected_cols = [
        'participante_id', 'audio_id', 'modelo', 
        'valencia_real', 'arousal_real',
        'valencia_resp', 'arousal_resp',
        'valencia_real_cat', 'arousal_real_cat',
        'valencia_binaria_strict', 'arousal_binario_strict',
        'genero', 'edad', 'conocimiento_musical'
    ]
    
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        print(f"\n⚠ WARNING: Missing expected columns: {missing_cols}")
    else:
        print(f"✓ All expected columns present")
    
    # Check for missing values
    print(f"\n--- Missing Values ---")
    missing = df[expected_cols].isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values in key columns")
    else:
        print(missing[missing > 0])
    
    # Unique values
    print(f"\n--- Key Statistics ---")
    print(f"Total responses: {len(df)}")
    print(f"Unique participants: {df['participante_id'].nunique()}")
    print(f"Unique audios: {df['audio_id'].nunique()}")
    
    print(f"\n--- Models ---")
    print(df['modelo'].value_counts())
    
    print(f"\n--- Target Emotional Conditions ---")
    print("Valencia categories:")
    print(df['valencia_real_cat'].value_counts())
    print("\nArousal categories:")
    print(df['arousal_real_cat'].value_counts())
    
    print(f"\n--- Response Categories ---")
    print("Valencia response (strict):")
    print(df['valencia_binaria_strict'].value_counts())
    print("\nArousal response (strict):")
    print(df['arousal_binario_strict'].value_counts())
    
    # Validate data integrity
    print(f"\n--- Data Integrity Checks ---")
    
    # Check valencia_resp range
    val_resp = df['valencia_resp']
    if val_resp.min() >= 1 and val_resp.max() <= 7:
        print(f"✓ valencia_resp in valid range [1,7]: [{val_resp.min()}, {val_resp.max()}]")
    else:
        print(f"⚠ valencia_resp out of range: [{val_resp.min()}, {val_resp.max()}]")
    
    # Check arousal_resp range
    ar_resp = df['arousal_resp']
    if ar_resp.min() >= 1 and ar_resp.max() <= 7:
        print(f"✓ arousal_resp in valid range [1,7]: [{ar_resp.min()}, {ar_resp.max()}]")
    else:
        print(f"⚠ arousal_resp out of range: [{ar_resp.min()}, {ar_resp.max()}]")
    
    # Check model names
    expected_models = ['baseline', 'pretrained', 'finetuned']
    actual_models = df['modelo'].unique().tolist()
    if set(actual_models) == set(expected_models):
        print(f"✓ All three models present: {actual_models}")
    else:
        print(f"⚠ Model names: {actual_models}")
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    
    return df

if __name__ == "__main__":
    df = load_and_validate()
    print(f"\n✓ Dataset ready for analysis: {df.shape}")
