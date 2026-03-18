#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_demographic_analysis.py
Exploratory analysis by gender and musical knowledge
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SURVEY_DATA = PROJECT_ROOT / "backend" / "survey" / "survey_long_cleaned.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "analysis" / "outputs"

def demographic_analysis(df):
    """Exploratory demographic analysis"""
    
    print("="*60)
    print("DEMOGRAPHIC ANALYSIS (Exploratory)")
    print("="*60)
    
    # Get unique participants
    participants = df.groupby('participante_id').first()
    
    print("\n⚠ NOTE: This analysis is exploratory due to small sample sizes.")
    print("Results should be interpreted with caution.\n")
    
    # Analysis by gender
    print("="*60)
    print("ANALYSIS BY GENDER")
    print("="*60)
    
    genders = participants['genero'].unique()
    print(f"\nGender groups: {list(genders)}")
    
    for gender in genders:
        # Get participant IDs for this gender
        participant_ids = participants[participants['genero'] == gender].index
        df_gender = df[df['participante_id'].isin(participant_ids)]
        
        n_participants = len(participant_ids)
        n_responses = len(df_gender)
        
        print(f"\n--- {gender} (N={n_participants} participants, {n_responses} responses) ---")
        
        # Valencia accuracy
        df_val = df_gender[df_gender['valencia_binaria_strict'].isin(['positiva', 'negativa'])]
        if len(df_val) > 0:
            val_acc = (df_val['valencia_real_cat'] == df_val['valencia_binaria_strict']).mean()
            print(f"Valencia accuracy: {val_acc:.3f} ({val_acc*100:.1f}%) on {len(df_val)} responses")
        else:
            print(f"Valencia accuracy: N/A (no valid responses)")
        
        # Arousal accuracy
        df_ar = df_gender[df_gender['arousal_binario_strict'].isin(['alto', 'bajo'])]
        if len(df_ar) > 0:
            ar_acc = (df_ar['arousal_real_cat'] == df_ar['arousal_binario_strict']).mean()
            print(f"Arousal accuracy:  {ar_acc:.3f} ({ar_acc*100:.1f}%) on {len(df_ar)} responses")
        else:
            print(f"Arousal accuracy: N/A (no valid responses)")
    
    # Analysis by musical knowledge
    print("\n" + "="*60)
    print("ANALYSIS BY MUSICAL KNOWLEDGE")
    print("="*60)
    
    mk_levels = participants['conocimiento_musical'].unique()
    mk_levels_sorted = sorted(mk_levels)
    print(f"\nMusical knowledge levels: {mk_levels_sorted}")
    
    for mk in mk_levels_sorted:
        # Get participant IDs for this level
        participant_ids = participants[participants['conocimiento_musical'] == mk].index
        df_mk = df[df['participante_id'].isin(participant_ids)]
        
        n_participants = len(participant_ids)
        n_responses = len(df_mk)
        
        print(f"\n--- Level {mk} (N={n_participants} participants, {n_responses} responses) ---")
        
        # Valencia accuracy
        df_val = df_mk[df_mk['valencia_binaria_strict'].isin(['positiva', 'negativa'])]
        if len(df_val) > 0:
            val_acc = (df_val['valencia_real_cat'] == df_val['valencia_binaria_strict']).mean()
            print(f"Valencia accuracy: {val_acc:.3f} ({val_acc*100:.1f}%) on {len(df_val)} responses")
        else:
            print(f"Valencia accuracy: N/A (no valid responses)")
        
        # Arousal accuracy
        df_ar = df_mk[df_mk['arousal_binario_strict'].isin(['alto', 'bajo'])]
        if len(df_ar) > 0:
            ar_acc = (df_ar['arousal_real_cat'] == df_ar['arousal_binario_strict']).mean()
            print(f"Arousal accuracy:  {ar_acc:.3f} ({ar_acc*100:.1f}%) on {len(df_ar)} responses")
        else:
            print(f"Arousal accuracy: N/A (no valid responses)")
    
    # Grouped analysis: categories of musical knowledge
    print("\n" + "="*60)
    print("GROUPED ANALYSIS BY MUSICAL KNOWLEDGE CATEGORIES")
    print("="*60)
    
    # Map categories (since they're strings)
    basic_categories = ['Sin formación musical', 'Formación básica']
    advanced_categories = ['Formación avanzada']
    
    # Basic/None knowledge
    participant_ids_basic = participants[participants['conocimiento_musical'].isin(basic_categories)].index
    df_basic = df[df['participante_id'].isin(participant_ids_basic)]
    
    print(f"\n--- Basic/No Musical Training ---")
    print(f"Categories: {basic_categories}")
    print(f"N participants: {len(participant_ids_basic)}")
    print(f"N responses: {len(df_basic)}")
    
    df_val_basic = df_basic[df_basic['valencia_binaria_strict'].isin(['positiva', 'negativa'])]
    if len(df_val_basic) > 0:
        val_acc_basic = (df_val_basic['valencia_real_cat'] == df_val_basic['valencia_binaria_strict']).mean()
        print(f"Valencia accuracy: {val_acc_basic:.3f} ({val_acc_basic*100:.1f}%)")
    
    df_ar_basic = df_basic[df_basic['arousal_binario_strict'].isin(['alto', 'bajo'])]
    if len(df_ar_basic) > 0:
        ar_acc_basic = (df_ar_basic['arousal_real_cat'] == df_ar_basic['arousal_binario_strict']).mean()
        print(f"Arousal accuracy:  {ar_acc_basic:.3f} ({ar_acc_basic*100:.1f}%)")
    
    # Advanced knowledge
    participant_ids_adv = participants[participants['conocimiento_musical'].isin(advanced_categories)].index
    df_adv = df[df['participante_id'].isin(participant_ids_adv)]
    
    print(f"\n--- Advanced Musical Training ---")
    print(f"Categories: {advanced_categories}")
    print(f"N participants: {len(participant_ids_adv)}")
    print(f"N responses: {len(df_adv)}")
    
    df_val_adv = df_adv[df_adv['valencia_binaria_strict'].isin(['positiva', 'negativa'])]
    if len(df_val_adv) > 0:
        val_acc_adv = (df_val_adv['valencia_real_cat'] == df_val_adv['valencia_binaria_strict']).mean()
        print(f"Valencia accuracy: {val_acc_adv:.3f} ({val_acc_adv*100:.1f}%)")
    
    df_ar_adv = df_adv[df_adv['arousal_binario_strict'].isin(['alto', 'bajo'])]
    if len(df_ar_adv) > 0:
        ar_acc_adv = (df_ar_adv['arousal_real_cat'] == df_ar_adv['arousal_binario_strict']).mean()
        print(f"Arousal accuracy:  {ar_acc_adv:.3f} ({ar_acc_adv*100:.1f}%)")
    
    # Save demographic tables
    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Gender table
    gender_results = []
    for gender in genders:
        participant_ids = participants[participants['genero'] == gender].index
        df_gender = df[df['participante_id'].isin(participant_ids)]
        
        df_val = df_gender[df_gender['valencia_binaria_strict'].isin(['positiva', 'negativa'])]
        val_acc = (df_val['valencia_real_cat'] == df_val['valencia_binaria_strict']).mean() if len(df_val) > 0 else np.nan
        
        df_ar = df_gender[df_gender['arousal_binario_strict'].isin(['alto', 'bajo'])]
        ar_acc = (df_ar['arousal_real_cat'] == df_ar['arousal_binario_strict']).mean() if len(df_ar) > 0 else np.nan
        
        gender_results.append({
            'Género': gender,
            'N Participantes': len(participant_ids),
            'Valencia Accuracy': val_acc,
            'Arousal Accuracy': ar_acc
        })
    
    gender_df = pd.DataFrame(gender_results)
    gender_df.to_csv(tables_dir / "demographic_gender_accuracy.csv", index=False, encoding='utf-8')
    
    # Musical knowledge table
    mk_results = []
    for mk in mk_levels_sorted:
        participant_ids = participants[participants['conocimiento_musical'] == mk].index
        df_mk = df[df['participante_id'].isin(participant_ids)]
        
        df_val = df_mk[df_mk['valencia_binaria_strict'].isin(['positiva', 'negativa'])]
        val_acc = (df_val['valencia_real_cat'] == df_val['valencia_binaria_strict']).mean() if len(df_val) > 0 else np.nan
        
        df_ar = df_mk[df_mk['arousal_binario_strict'].isin(['alto', 'bajo'])]
        ar_acc = (df_ar['arousal_real_cat'] == df_ar['arousal_binario_strict']).mean() if len(df_ar) > 0 else np.nan
        
        mk_results.append({
            'Conocimiento Musical': mk,
            'N Participantes': len(participant_ids),
            'Valencia Accuracy': val_acc,
            'Arousal Accuracy': ar_acc
        })
    
    mk_df = pd.DataFrame(mk_results)
    mk_df.to_csv(tables_dir / "demographic_musical_knowledge_accuracy.csv", index=False, encoding='utf-8')
    
    print(f"\n✓ Demographic analysis saved to: {tables_dir}")
    
    print("\n" + "="*60)
    print("INTERPRETATION NOTES")
    print("="*60)
    print("""
These results are exploratory due to small sample sizes per subgroup.
Statistical tests are not reliable with such small samples.
Patterns observed here should be confirmed with larger studies.
    """)

if __name__ == "__main__":
    df = pd.read_excel(SURVEY_DATA)
    demographic_analysis(df)
