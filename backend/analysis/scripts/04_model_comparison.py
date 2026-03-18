#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_model_comparison.py
Compare the three models statistically
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SURVEY_DATA = PROJECT_ROOT / "backend" / "survey" / "survey_long_cleaned.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "analysis" / "outputs"

def compare_models(df):
    """Statistical comparison between models"""
    
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Prepare data
    models = ['baseline', 'pretrained', 'finetuned']
    
    # Valencia comparison
    print("\n--- VALENCIA DIMENSION ---")
    
    valencia_accuracies = {}
    for modelo in models:
        df_model = df[
            (df['modelo'] == modelo) & 
            (df['valencia_binaria_strict'].isin(['positiva', 'negativa']))
        ]
        if len(df_model) > 0:
            correct = (df_model['valencia_real_cat'] == df_model['valencia_binaria_strict']).sum()
            total = len(df_model)
            acc = correct / total
            valencia_accuracies[modelo] = {
                'correct': correct,
                'total': total,
                'accuracy': acc
            }
            print(f"{modelo:12s}: {correct:3d}/{total:3d} = {acc:.3f} ({acc*100:.1f}%)")
    
    # Find best model for valencia
    best_val_model = max(valencia_accuracies.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n✓ Best model for valencia: {best_val_model[0]} ({best_val_model[1]['accuracy']*100:.1f}%)")
    
    # Arousal comparison
    print("\n--- AROUSAL/ACTIVACIÓN DIMENSION ---")
    
    arousal_accuracies = {}
    for modelo in models:
        df_model = df[
            (df['modelo'] == modelo) & 
            (df['arousal_binario_strict'].isin(['alto', 'bajo']))
        ]
        if len(df_model) > 0:
            correct = (df_model['arousal_real_cat'] == df_model['arousal_binario_strict']).sum()
            total = len(df_model)
            acc = correct / total
            arousal_accuracies[modelo] = {
                'correct': correct,
                'total': total,
                'accuracy': acc
            }
            print(f"{modelo:12s}: {correct:3d}/{total:3d} = {acc:.3f} ({acc*100:.1f}%)")
    
    # Find best model for arousal
    best_ar_model = max(arousal_accuracies.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n✓ Best model for arousal: {best_ar_model[0]} ({best_ar_model[1]['accuracy']*100:.1f}%)")
    
    # Proportion test (chi-square) for valencia
    print("\n--- Statistical Testing: Valencia ---")
    try:
        # Create contingency table for chi-square test
        contingency_val = []
        for modelo in models:
            data = valencia_accuracies[modelo]
            contingency_val.append([data['correct'], data['total'] - data['correct']])
        
        chi2_val, p_val_val = stats.chi2_contingency(contingency_val)[:2]
        print(f"Chi-square test: χ² = {chi2_val:.3f}, p = {p_val_val:.4f}")
        
        if p_val_val < 0.05:
            print("→ Differences between models are statistically significant (p < 0.05)")
        else:
            print("→ No significant differences between models (p ≥ 0.05)")
    except Exception as e:
        print(f"⚠ Could not perform chi-square test: {e}")
    
    # Proportion test (chi-square) for arousal
    print("\n--- Statistical Testing: Arousal ---")
    try:
        # Create contingency table for chi-square test
        contingency_ar = []
        for modelo in models:
            data = arousal_accuracies[modelo]
            contingency_ar.append([data['correct'], data['total'] - data['correct']])
        
        chi2_ar, p_val_ar = stats.chi2_contingency(contingency_ar)[:2]
        print(f"Chi-square test: χ² = {chi2_ar:.3f}, p = {p_val_ar:.4f}")
        
        if p_val_ar < 0.05:
            print("→ Differences between models are statistically significant (p < 0.05)")
        else:
            print("→ No significant differences between models (p ≥ 0.05)")
    except Exception as e:
        print(f"⚠ Could not perform chi-square test: {e}")
    
    # Mean responses on 1-7 scale by model
    print("\n" + "="*60)
    print("MEAN RESPONSE VALUES (1-7 scale) BY MODEL AND TARGET")
    print("="*60)
    
    print("\n--- Valencia Responses by Target and Model ---")
    for val_target in ['negativa', 'positiva']:
        print(f"\nTarget: {val_target}")
        for modelo in models:
            df_subset = df[(df['modelo'] == modelo) & (df['valencia_real_cat'] == val_target)]
            if len(df_subset) > 0:
                mean_resp = df_subset['valencia_resp'].mean()
                std_resp = df_subset['valencia_resp'].std()
                print(f"  {modelo:12s}: M = {mean_resp:.2f}, SD = {std_resp:.2f}, N = {len(df_subset)}")
    
    print("\n--- Arousal Responses by Target and Model ---")
    for ar_target in ['bajo', 'alto']:
        print(f"\nTarget: {ar_target}")
        for modelo in models:
            df_subset = df[(df['modelo'] == modelo) & (df['arousal_real_cat'] == ar_target)]
            if len(df_subset) > 0:
                mean_resp = df_subset['arousal_resp'].mean()
                std_resp = df_subset['arousal_resp'].std()
                print(f"  {modelo:12s}: M = {mean_resp:.2f}, SD = {std_resp:.2f}, N = {len(df_subset)}")
    
    # Calculate confidence intervals (95%)
    print("\n" + "="*60)
    print("CONFIDENCE INTERVALS (95%)")
    print("="*60)
    
    print("\n--- Valencia Accuracy ---")
    for modelo in models:
        data = valencia_accuracies[modelo]
        acc = data['accuracy']
        n = data['total']
        # Wilson score interval
        z = 1.96  # 95% CI
        denominator = 1 + z**2/n
        center = (acc + z**2/(2*n)) / denominator
        margin = z * np.sqrt(acc*(1-acc)/n + z**2/(4*n**2)) / denominator
        ci_low = center - margin
        ci_high = center + margin
        print(f"{modelo:12s}: {acc:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
    
    print("\n--- Arousal Accuracy ---")
    for modelo in models:
        data = arousal_accuracies[modelo]
        acc = data['accuracy']
        n = data['total']
        # Wilson score interval
        z = 1.96  # 95% CI
        denominator = 1 + z**2/n
        center = (acc + z**2/(2*n)) / denominator
        margin = z * np.sqrt(acc*(1-acc)/n + z**2/(4*n**2)) / denominator
        ci_low = center - margin
        ci_high = center + margin
        print(f"{modelo:12s}: {acc:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
    
    # Save comparison table
    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_df = pd.DataFrame({
        'Modelo': models,
        'Valencia Accuracy': [valencia_accuracies[m]['accuracy'] for m in models],
        'Valencia N': [valencia_accuracies[m]['total'] for m in models],
        'Arousal Accuracy': [arousal_accuracies[m]['accuracy'] for m in models],
        'Arousal N': [arousal_accuracies[m]['total'] for m in models]
    })
    comparison_df.to_csv(tables_dir / "model_comparison.csv", index=False, encoding='utf-8')
    
    print(f"\n✓ Model comparison saved to: {tables_dir}")
    
    return {
        'valencia': valencia_accuracies,
        'arousal': arousal_accuracies,
        'best_valencia': best_val_model[0],
        'best_arousal': best_ar_model[0]
    }

if __name__ == "__main__":
    df = pd.read_excel(SURVEY_DATA)
    results = compare_models(df)
