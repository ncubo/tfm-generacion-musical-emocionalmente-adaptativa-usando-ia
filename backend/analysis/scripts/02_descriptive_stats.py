#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_descriptive_stats.py
Compute and export descriptive statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SURVEY_DATA = PROJECT_ROOT / "backend" / "survey" / "survey_long_cleaned.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "analysis" / "outputs"

def compute_descriptive_stats(df):
    """Compute descriptive statistics"""
    
    print("="*60)
    print("DESCRIPTIVE STATISTICS")
    print("="*60)
    
    # Participant demographics (unique participants only)
    participants = df.groupby('participante_id').first()
    
    n_participants = len(participants)
    n_responses = len(df)
    n_audios = df['audio_id'].nunique()
    
    print(f"\n--- Sample Size ---")
    print(f"Total participants: {n_participants}")
    print(f"Total responses: {n_responses}")
    print(f"Unique audios: {n_audios}")
    print(f"Expected responses per participant: 12")
    print(f"Actual responses per participant: {n_responses / n_participants:.1f}")
    
    # Gender distribution
    print(f"\n--- Gender Distribution ---")
    gender_counts = participants['genero'].value_counts()
    print(gender_counts)
    print("\nPercentages:")
    print((gender_counts / n_participants * 100).round(1))
    
    # Age statistics
    print(f"\n--- Age Distribution ---")
    print(participants['edad'].describe())
    
    # Musical knowledge
    print(f"\n--- Musical Knowledge ---")
    mk_counts = participants['conocimiento_musical'].value_counts().sort_index()
    print(mk_counts)
    print("\nPercentages:")
    print((mk_counts / n_participants * 100).round(1))
    
    # Responses per model
    print(f"\n--- Responses per Model ---")
    model_counts = df['modelo'].value_counts()
    print(model_counts)
    print("\nPercentages:")
    print((model_counts / n_responses * 100).round(1))
    
    # Target emotional conditions
    print(f"\n--- Target Emotional Conditions ---")
    print("\nValencia target distribution:")
    val_target = df['valencia_real_cat'].value_counts()
    print(val_target)
    
    print("\nArousal target distribution:")
    ar_target = df['arousal_real_cat'].value_counts()
    print(ar_target)
    
    # Response distributions (1-7 scale)
    print(f"\n--- Response Scale Statistics (1-7) ---")
    print("\nValencia responses:")
    print(df['valencia_resp'].describe())
    
    print("\nArousal responses:")
    print(df['arousal_resp'].describe())
    
    # Response categories
    print(f"\n--- Categorized Responses ---")
    print("\nValencia (strict binary):")
    val_resp_counts = df['valencia_binaria_strict'].value_counts()
    print(val_resp_counts)
    print(f"  Neutral rate: {val_resp_counts.get('neutra', 0) / n_responses * 100:.1f}%")
    
    print("\nArousal (strict binary):")
    ar_resp_counts = df['arousal_binario_strict'].value_counts()
    print(ar_resp_counts)
    print(f"  Neutral rate: {ar_resp_counts.get('neutro', 0) / n_responses * 100:.1f}%")
    
    # Create summary dataframes for export
    demographics_summary = pd.DataFrame({
        'Variable': ['N participantes', 'N respuestas', 'N audios únicos', 
                     'Respuestas/participante'],
        'Valor': [n_participants, n_responses, n_audios, f"{n_responses/n_participants:.1f}"]
    })
    
    gender_summary = pd.DataFrame({
        'Género': gender_counts.index,
        'N': gender_counts.values,
        'Porcentaje': (gender_counts.values / n_participants * 100).round(1)
    })
    
    age_summary = participants['edad'].describe().round(1).to_frame('Edad')
    
    mk_summary = pd.DataFrame({
        'Conocimiento Musical': mk_counts.index,
        'N': mk_counts.values,
        'Porcentaje': (mk_counts.values / n_participants * 100).round(1)
    })
    
    model_summary = pd.DataFrame({
        'Modelo': model_counts.index,
        'N Respuestas': model_counts.values,
        'Porcentaje': (model_counts.values / n_responses * 100).round(1)
    })
    
    # Save to CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    demographics_summary.to_csv(tables_dir / "demographics_basic.csv", index=False, encoding='utf-8')
    gender_summary.to_csv(tables_dir / "demographics_gender.csv", index=False, encoding='utf-8')
    age_summary.to_csv(tables_dir / "demographics_age.csv", encoding='utf-8')
    mk_summary.to_csv(tables_dir / "demographics_musical_knowledge.csv", index=False, encoding='utf-8')
    model_summary.to_csv(tables_dir / "responses_by_model.csv", index=False, encoding='utf-8')
    
    print(f"\n✓ Descriptive statistics saved to: {tables_dir}")
    
    return {
        'demographics': demographics_summary,
        'gender': gender_summary,
        'age': age_summary,
        'musical_knowledge': mk_summary,
        'model_counts': model_summary
    }

if __name__ == "__main__":
    df = pd.read_excel(SURVEY_DATA)
    stats = compute_descriptive_stats(df)
