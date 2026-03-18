#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_export_latex_tables.py
Export results as LaTeX tables for thesis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SURVEY_DATA = PROJECT_ROOT / "backend" / "survey" / "survey_long_cleaned.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "analysis" / "outputs"

def export_latex_tables(df):
    """Export all results as LaTeX tables"""
    
    print("="*60)
    print("EXPORTING LATEX TABLES")
    print("="*60)
    
    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique participants
    participants = df.groupby('participante_id').first()
    n_participants = len(participants)
    n_responses = len(df)
    
    # 1. Demographics summary table
    print("\n[1/6] Demographics summary...")
    
    gender_counts = participants['genero'].value_counts()
    age_stats = participants['edad'].describe()
    mk_counts = participants['conocimiento_musical'].value_counts().sort_index()
    
    latex_demographics = r"""\begin{table}[htbp]
\centering
\caption{Características demográficas de la muestra de participantes}
\label{tab:demographics}
\begin{tabular}{ll}
\toprule
\textbf{Variable} & \textbf{Valor} \\
\midrule
N participantes & """ + f"{n_participants}" + r""" \\
N respuestas totales & """ + f"{n_responses}" + r""" \\
Respuestas por participante & """ + f"{n_responses/n_participants:.1f}" + r""" \\
\midrule
\textbf{Género} & \\
"""
    
    for gender, count in gender_counts.items():
        pct = count / n_participants * 100
        latex_demographics += f"  {gender.capitalize()} & {count} ({pct:.1f}\\%) \\\\\n"
    
    latex_demographics += r"""\midrule
\textbf{Edad} & \\
  Media (DE) & """ + f"{age_stats['mean']:.1f} ({age_stats['std']:.1f})" + r""" \\
  Rango & """ + f"[{int(age_stats['min'])}, {int(age_stats['max'])}]" + r""" \\
\midrule
\textbf{Conocimiento musical} & \\
"""
    
    for mk, count in mk_counts.items():
        pct = count / n_participants * 100
        latex_demographics += f"  Nivel {mk} & {count} ({pct:.1f}\\%) \\\\\n"
    
    latex_demographics += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(tables_dir / "latex_demographics.tex", 'w', encoding='utf-8') as f:
        f.write(latex_demographics)
    
    # 2. Overall accuracy table
    print("[2/6] Overall accuracy...")
    
    # Valencia
    df_val = df[df['valencia_binaria_strict'].isin(['positiva', 'negativa'])]
    val_accuracy = (df_val['valencia_real_cat'] == df_val['valencia_binaria_strict']).mean()
    n_valid_val = len(df_val)
    n_neutral_val = n_responses - n_valid_val
    
    # Arousal
    df_ar = df[df['arousal_binario_strict'].isin(['alto', 'bajo'])]
    ar_accuracy = (df_ar['arousal_real_cat'] == df_ar['arousal_binario_strict']).mean()
    n_valid_ar = len(df_ar)
    n_neutral_ar = n_responses - n_valid_ar
    
    latex_overall = r"""\begin{table}[htbp]
\centering
\caption{Accuracy de percepción emocional global}
\label{tab:accuracy_overall}
\begin{tabular}{lrrr}
\toprule
\textbf{Dimensión} & \textbf{N válidas} & \textbf{N neutras} & \textbf{Accuracy} \\
\midrule
Valencia & """ + f"{n_valid_val} & {n_neutral_val} & {val_accuracy:.3f} ({val_accuracy*100:.1f}\\%)" + r""" \\
Activación & """ + f"{n_valid_ar} & {n_neutral_ar} & {ar_accuracy:.3f} ({ar_accuracy*100:.1f}\\%)" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(tables_dir / "latex_accuracy_overall.tex", 'w', encoding='utf-8') as f:
        f.write(latex_overall)
    
    # 3. Accuracy by model
    print("[3/6] Accuracy by model...")
    
    models = ['baseline', 'pretrained', 'finetuned']
    model_names = ['Baseline', 'Transformer preentrenado', 'Transformer fine-tuned']
    
    latex_model = r"""\begin{table}[htbp]
\centering
\caption{Accuracy de percepción emocional por motor de generación}
\label{tab:accuracy_by_model}
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{\textbf{Valencia}} & \multicolumn{2}{c}{\textbf{Activación}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
\textbf{Modelo} & \textbf{Accuracy} & \textbf{N} & \textbf{Accuracy} & \textbf{N} \\
\midrule
"""
    
    for modelo, nombre in zip(models, model_names):
        # Valencia
        df_model_val = df[
            (df['modelo'] == modelo) & 
            (df['valencia_binaria_strict'].isin(['positiva', 'negativa']))
        ]
        val_acc = (df_model_val['valencia_real_cat'] == df_model_val['valencia_binaria_strict']).mean()
        n_val = len(df_model_val)
        
        # Arousal
        df_model_ar = df[
            (df['modelo'] == modelo) & 
            (df['arousal_binario_strict'].isin(['alto', 'bajo']))
        ]
        ar_acc = (df_model_ar['arousal_real_cat'] == df_model_ar['arousal_binario_strict']).mean()
        n_ar = len(df_model_ar)
        
        latex_model += f"{nombre} & {val_acc:.3f} & {n_val} & {ar_acc:.3f} & {n_ar} \\\\\n"
    
    latex_model += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(tables_dir / "latex_accuracy_by_model.tex", 'w', encoding='utf-8') as f:
        f.write(latex_model)
    
    # 4. Confusion matrix - Valencia
    print("[4/6] Confusion matrix - Valencia...")
    
    cm_val = confusion_matrix(
        df_val['valencia_real_cat'], 
        df_val['valencia_binaria_strict'],
        labels=['negativa', 'positiva']
    )
    
    latex_cm_val = r"""\begin{table}[htbp]
\centering
\caption{Matriz de confusión: dimensión de valencia}
\label{tab:confusion_valencia}
\begin{tabular}{lcc}
\toprule
& \multicolumn{2}{c}{\textbf{Valencia percibida}} \\
\cmidrule{2-3}
\textbf{Valencia objetivo} & \textbf{Negativa} & \textbf{Positiva} \\
\midrule
Negativa & """ + f"{cm_val[0,0]} & {cm_val[0,1]}" + r""" \\
Positiva & """ + f"{cm_val[1,0]} & {cm_val[1,1]}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(tables_dir / "latex_confusion_valencia.tex", 'w', encoding='utf-8') as f:
        f.write(latex_cm_val)
    
    # 5. Confusion matrix - Arousal
    print("[5/6] Confusion matrix - Arousal...")
    
    cm_ar = confusion_matrix(
        df_ar['arousal_real_cat'], 
        df_ar['arousal_binario_strict'],
        labels=['bajo', 'alto']
    )
    
    latex_cm_ar = r"""\begin{table}[htbp]
\centering
\caption{Matriz de confusión: dimensión de activación}
\label{tab:confusion_arousal}
\begin{tabular}{lcc}
\toprule
& \multicolumn{2}{c}{\textbf{Activación percibida}} \\
\cmidrule{2-3}
\textbf{Activación objetivo} & \textbf{Bajo} & \textbf{Alto} \\
\midrule
Bajo & """ + f"{cm_ar[0,0]} & {cm_ar[0,1]}" + r""" \\
Alto & """ + f"{cm_ar[1,0]} & {cm_ar[1,1]}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(tables_dir / "latex_confusion_arousal.tex", 'w', encoding='utf-8') as f:
        f.write(latex_cm_ar)
    
    # 6. Mean responses by target (1-7 scale)
    print("[6/6] Mean responses by target...")
    
    latex_means = r"""\begin{table}[htbp]
\centering
\caption{Respuestas medias en escala 1--7 por condición emocional objetivo y modelo}
\label{tab:mean_responses}
\begin{tabular}{llccc}
\toprule
\textbf{Dimensión} & \textbf{Objetivo} & \textbf{Baseline} & \textbf{Preentrenado} & \textbf{Fine-tuned} \\
\midrule
"""
    
    # Valencia means
    for val_cat in ['negativa', 'positiva']:
        latex_means += f"Valencia & {val_cat.capitalize()}"
        for modelo in models:
            df_subset = df[(df['valencia_real_cat'] == val_cat) & (df['modelo'] == modelo)]
            if len(df_subset) > 0:
                mean_resp = df_subset['valencia_resp'].mean()
                std_resp = df_subset['valencia_resp'].std()
                latex_means += f" & {mean_resp:.2f} ({std_resp:.2f})"
            else:
                latex_means += " & ---"
        latex_means += " \\\\\n"
    
    latex_means += r"""\midrule
"""
    
    # Arousal means
    for ar_cat in ['bajo', 'alto']:
        latex_means += f"Activación & {ar_cat.capitalize()}"
        for modelo in models:
            df_subset = df[(df['arousal_real_cat'] == ar_cat) & (df['modelo'] == modelo)]
            if len(df_subset) > 0:
                mean_resp = df_subset['arousal_resp'].mean()
                std_resp = df_subset['arousal_resp'].std()
                latex_means += f" & {mean_resp:.2f} ({std_resp:.2f})"
            else:
                latex_means += " & ---"
        latex_means += " \\\\\n"
    
    latex_means += r"""\bottomrule
\multicolumn{5}{l}{\footnotesize Valores mostrados como: M (DE)} \\
\end{tabular}
\end{table}
"""
    
    with open(tables_dir / "latex_mean_responses.tex", 'w', encoding='utf-8') as f:
        f.write(latex_means)
    
    print(f"\n✓ All LaTeX tables saved to: {tables_dir}")
    print("\nGenerated LaTeX tables:")
    for tex_file in sorted(tables_dir.glob("latex_*.tex")):
        print(f"  - {tex_file.name}")
    
    print("\nThese tables can be included in the thesis using:")
    print("  \\input{analysis/human_evaluation/outputs/tables/<filename>.tex}")

if __name__ == "__main__":
    df = pd.read_excel(SURVEY_DATA)
    export_latex_tables(df)
