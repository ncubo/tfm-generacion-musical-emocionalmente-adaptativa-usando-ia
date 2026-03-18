#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_accuracy_and_confusion.py
Compute accuracy metrics and confusion matrices for human evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SURVEY_DATA = PROJECT_ROOT / "backend" / "survey" / "survey_long_cleaned.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "analysis" / "outputs"

def compute_accuracy_metrics(df):
    """Compute accuracy metrics excluding neutral responses"""
    
    print("="*60)
    print("ACCURACY METRICS AND CONFUSION MATRICES")
    print("="*60)
    
    results = {}
    
    # Overall accuracy
    print("\n" + "="*60)
    print("OVERALL ACCURACY (excluding neutral responses)")
    print("="*60)
    
    # Valencia accuracy
    df_val = df[df['valencia_binaria_strict'].isin(['positiva', 'negativa'])].copy()
    n_total_val = len(df)
    n_valid_val = len(df_val)
    n_neutral_val = n_total_val - n_valid_val
    
    val_accuracy = accuracy_score(df_val['valencia_real_cat'], df_val['valencia_binaria_strict'])
    
    print(f"\n--- Valencia ---")
    print(f"Total responses: {n_total_val}")
    print(f"Valid (non-neutral): {n_valid_val} ({n_valid_val/n_total_val*100:.1f}%)")
    print(f"Neutral: {n_neutral_val} ({n_neutral_val/n_total_val*100:.1f}%)")
    print(f"Accuracy (on valid): {val_accuracy:.3f} ({val_accuracy*100:.1f}%)")
    
    # Arousal accuracy
    df_ar = df[df['arousal_binario_strict'].isin(['alto', 'bajo'])].copy()
    n_total_ar = len(df)
    n_valid_ar = len(df_ar)
    n_neutral_ar = n_total_ar - n_valid_ar
    
    ar_accuracy = accuracy_score(df_ar['arousal_real_cat'], df_ar['arousal_binario_strict'])
    
    print(f"\n--- Arousal/Activación ---")
    print(f"Total responses: {n_total_ar}")
    print(f"Valid (non-neutral): {n_valid_ar} ({n_valid_ar/n_total_ar*100:.1f}%)")
    print(f"Neutral: {n_neutral_ar} ({n_neutral_ar/n_total_ar*100:.1f}%)")
    print(f"Accuracy (on valid): {ar_accuracy:.3f} ({ar_accuracy*100:.1f}%)")
    
    results['overall'] = {
        'valencia_accuracy': val_accuracy,
        'valencia_n_valid': n_valid_val,
        'valencia_n_neutral': n_neutral_val,
        'arousal_accuracy': ar_accuracy,
        'arousal_n_valid': n_valid_ar,
        'arousal_n_neutral': n_neutral_ar
    }
    
    # Confusion matrices - Overall
    print("\n" + "="*60)
    print("CONFUSION MATRICES - OVERALL")
    print("="*60)
    
    # Valencia confusion matrix
    cm_val = confusion_matrix(
        df_val['valencia_real_cat'], 
        df_val['valencia_binaria_strict'],
        labels=['negativa', 'positiva']
    )
    
    print("\n--- Valencia Confusion Matrix ---")
    print("                 Predicted")
    print("                 negativa  positiva")
    print(f"True negativa    {cm_val[0,0]:8d}  {cm_val[0,1]:8d}")
    print(f"True positiva    {cm_val[1,0]:8d}  {cm_val[1,1]:8d}")
    
    # Arousal confusion matrix
    cm_ar = confusion_matrix(
        df_ar['arousal_real_cat'], 
        df_ar['arousal_binario_strict'],
        labels=['bajo', 'alto']
    )
    
    print("\n--- Arousal Confusion Matrix ---")
    print("              Predicted")
    print("              bajo  alto")
    print(f"True bajo    {cm_ar[0,0]:4d}  {cm_ar[0,1]:4d}")
    print(f"True alto    {cm_ar[1,0]:4d}  {cm_ar[1,1]:4d}")
    
    results['confusion_matrices'] = {
        'valencia': cm_val,
        'arousal': cm_ar
    }
    
    # Accuracy by model
    print("\n" + "="*60)
    print("ACCURACY BY MODEL")
    print("="*60)
    
    model_results = {}
    
    for modelo in ['baseline', 'pretrained', 'finetuned']:
        df_model = df[df['modelo'] == modelo]
        
        # Valencia
        df_model_val = df_model[df_model['valencia_binaria_strict'].isin(['positiva', 'negativa'])]
        if len(df_model_val) > 0:
            val_acc = accuracy_score(
                df_model_val['valencia_real_cat'], 
                df_model_val['valencia_binaria_strict']
            )
        else:
            val_acc = np.nan
        
        # Arousal
        df_model_ar = df_model[df_model['arousal_binario_strict'].isin(['alto', 'bajo'])]
        if len(df_model_ar) > 0:
            ar_acc = accuracy_score(
                df_model_ar['arousal_real_cat'], 
                df_model_ar['arousal_binario_strict']
            )
        else:
            ar_acc = np.nan
        
        print(f"\n--- {modelo.upper()} ---")
        print(f"Valencia accuracy: {val_acc:.3f} ({val_acc*100:.1f}%) on {len(df_model_val)} valid responses")
        print(f"Arousal accuracy:  {ar_acc:.3f} ({ar_acc*100:.1f}%) on {len(df_model_ar)} valid responses")
        
        model_results[modelo] = {
            'valencia_accuracy': val_acc,
            'valencia_n': len(df_model_val),
            'arousal_accuracy': ar_acc,
            'arousal_n': len(df_model_ar),
            'cm_valencia': confusion_matrix(
                df_model_val['valencia_real_cat'], 
                df_model_val['valencia_binaria_strict'],
                labels=['negativa', 'positiva']
            ) if len(df_model_val) > 0 else None,
            'cm_arousal': confusion_matrix(
                df_model_ar['arousal_real_cat'], 
                df_model_ar['arousal_binario_strict'],
                labels=['bajo', 'alto']
            ) if len(df_model_ar) > 0 else None
        }
    
    results['by_model'] = model_results
    
    # Accuracy by emotional target class
    print("\n" + "="*60)
    print("ACCURACY BY TARGET EMOTIONAL CLASS")
    print("="*60)
    
    # Valencia by target
    print("\n--- Valencia by Target Class ---")
    for val_cat in ['negativa', 'positiva']:
        df_target = df[
            (df['valencia_real_cat'] == val_cat) & 
            (df['valencia_binaria_strict'].isin(['positiva', 'negativa']))
        ]
        if len(df_target) > 0:
            acc = (df_target['valencia_real_cat'] == df_target['valencia_binaria_strict']).mean()
            print(f"Target {val_cat}: {acc:.3f} ({acc*100:.1f}%) on {len(df_target)} responses")
    
    # Arousal by target
    print("\n--- Arousal by Target Class ---")
    for ar_cat in ['bajo', 'alto']:
        df_target = df[
            (df['arousal_real_cat'] == ar_cat) & 
            (df['arousal_binario_strict'].isin(['alto', 'bajo']))
        ]
        if len(df_target) > 0:
            acc = (df_target['arousal_real_cat'] == df_target['arousal_binario_strict']).mean()
            print(f"Target {ar_cat}: {acc:.3f} ({acc*100:.1f}%) on {len(df_target)} responses")
    
    # Save results to CSV
    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Overall accuracy table
    overall_df = pd.DataFrame({
        'Dimensión': ['Valencia', 'Arousal/Activación'],
        'N válidas': [results['overall']['valencia_n_valid'], results['overall']['arousal_n_valid']],
        'N neutras': [results['overall']['valencia_n_neutral'], results['overall']['arousal_n_neutral']],
        'Accuracy': [results['overall']['valencia_accuracy'], results['overall']['arousal_accuracy']],
        'Accuracy %': [results['overall']['valencia_accuracy']*100, results['overall']['arousal_accuracy']*100]
    })
    overall_df.to_csv(tables_dir / "accuracy_overall.csv", index=False, encoding='utf-8')
    
    # Model accuracy table
    model_df = pd.DataFrame({
        'Modelo': ['baseline', 'pretrained', 'finetuned'],
        'Valencia Acc': [model_results[m]['valencia_accuracy'] for m in ['baseline', 'pretrained', 'finetuned']],
        'Valencia N': [model_results[m]['valencia_n'] for m in ['baseline', 'pretrained', 'finetuned']],
        'Arousal Acc': [model_results[m]['arousal_accuracy'] for m in ['baseline', 'pretrained', 'finetuned']],
        'Arousal N': [model_results[m]['arousal_n'] for m in ['baseline', 'pretrained', 'finetuned']]
    })
    model_df.to_csv(tables_dir / "accuracy_by_model.csv", index=False, encoding='utf-8')
    
    # Confusion matrices
    np.savetxt(tables_dir / "confusion_matrix_valencia.csv", cm_val, delimiter=',', fmt='%d')
    np.savetxt(tables_dir / "confusion_matrix_arousal.csv", cm_ar, delimiter=',', fmt='%d')
    
    print(f"\n✓ Accuracy metrics saved to: {tables_dir}")
    
    return results

if __name__ == "__main__":
    df = pd.read_excel(SURVEY_DATA)
    results = compute_accuracy_metrics(df)
