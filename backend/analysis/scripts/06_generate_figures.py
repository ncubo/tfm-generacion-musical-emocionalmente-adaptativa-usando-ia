#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_generate_figures.py
Generate publication-ready figures for human evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SURVEY_DATA = PROJECT_ROOT / "backend" / "survey" / "survey_long_cleaned.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "analysis" / "outputs"

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("Set2")

def generate_figures(df):
    """Generate all figures for the thesis"""
    
    print("="*60)
    print("GENERATING FIGURES")
    print("="*60)
    
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique participants
    participants = df.groupby('participante_id').first()
    
    # 1. Gender distribution
    print("\n[1/11] Gender distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    gender_counts = participants['genero'].value_counts()
    ax.bar(gender_counts.index, gender_counts.values, color=['#66c2a5', '#fc8d62', '#8da0cb'])
    ax.set_xlabel('Género', fontsize=12)
    ax.set_ylabel('Número de participantes', fontsize=12)
    ax.set_title('Distribución de participantes por género', fontsize=14, fontweight='bold')
    for i, v in enumerate(gender_counts.values):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / "demographics_gender.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Age distribution
    print("[2/11] Age distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(participants['edad'], bins=10, color='#8da0cb', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Edad (años)', fontsize=12)
    ax.set_ylabel('Número de participantes', fontsize=12)
    ax.set_title('Distribución de edad de los participantes', fontsize=14, fontweight='bold')
    ax.axvline(participants['edad'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {participants["edad"].mean():.1f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "demographics_age.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Musical knowledge distribution
    print("[3/11] Musical knowledge distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    mk_counts = participants['conocimiento_musical'].value_counts().sort_index()
    ax.bar(mk_counts.index, mk_counts.values, color='#a6d854', edgecolor='black')
    ax.set_xlabel('Conocimiento musical (1=bajo, 5=alto)', fontsize=12)
    ax.set_ylabel('Número de participantes', fontsize=12)
    ax.set_title('Distribución de conocimiento musical', fontsize=14, fontweight='bold')
    ax.set_xticks(mk_counts.index)
    for i, v in enumerate(mk_counts.values):
        ax.text(mk_counts.index[i], v + 0.3, str(v), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / "demographics_musical_knowledge.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Valencia accuracy by model
    print("[4/11] Valencia accuracy by model...")
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['baseline', 'pretrained', 'finetuned']
    model_labels = ['Baseline', 'Transformer\nPreentrenado', 'Transformer\nFine-tuned']
    val_accuracies = []
    for modelo in models:
        df_model = df[
            (df['modelo'] == modelo) & 
            (df['valencia_binaria_strict'].isin(['positiva', 'negativa']))
        ]
        if len(df_model) > 0:
            acc = (df_model['valencia_real_cat'] == df_model['valencia_binaria_strict']).mean()
            val_accuracies.append(acc * 100)
        else:
            val_accuracies.append(0)
    
    bars = ax.bar(model_labels, val_accuracies, color=['#e78ac3', '#8da0cb', '#66c2a5'], edgecolor='black')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy de percepción de valencia por modelo', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Chance (50%)')
    for i, v in enumerate(val_accuracies):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "accuracy_valencia_by_model.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Arousal accuracy by model
    print("[5/11] Arousal accuracy by model...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ar_accuracies = []
    for modelo in models:
        df_model = df[
            (df['modelo'] == modelo) & 
            (df['arousal_binario_strict'].isin(['alto', 'bajo']))
        ]
        if len(df_model) > 0:
            acc = (df_model['arousal_real_cat'] == df_model['arousal_binario_strict']).mean()
            ar_accuracies.append(acc * 100)
        else:
            ar_accuracies.append(0)
    
    bars = ax.bar(model_labels, ar_accuracies, color=['#e78ac3', '#8da0cb', '#66c2a5'], edgecolor='black')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy de percepción de activación por modelo', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Chance (50%)')
    for i, v in enumerate(ar_accuracies):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "accuracy_arousal_by_model.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Valencia confusion matrix - overall
    print("[6/11] Valencia confusion matrix (overall)...")
    df_val = df[df['valencia_binaria_strict'].isin(['positiva', 'negativa'])]
    cm_val = confusion_matrix(
        df_val['valencia_real_cat'], 
        df_val['valencia_binaria_strict'],
        labels=['negativa', 'positiva']
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Negativa', 'Positiva'],
                yticklabels=['Negativa', 'Positiva'],
                ax=ax, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    ax.set_xlabel('Valencia percibida', fontsize=12)
    ax.set_ylabel('Valencia objetivo', fontsize=12)
    ax.set_title('Matriz de confusión: Valencia (todos los modelos)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix_valencia_overall.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Arousal confusion matrix - overall
    print("[7/11] Arousal confusion matrix (overall)...")
    df_ar = df[df['arousal_binario_strict'].isin(['alto', 'bajo'])]
    cm_ar = confusion_matrix(
        df_ar['arousal_real_cat'], 
        df_ar['arousal_binario_strict'],
        labels=['bajo', 'alto']
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_ar, annot=True, fmt='d', cmap='Oranges', cbar=True,
                xticklabels=['Bajo', 'Alto'],
                yticklabels=['Bajo', 'Alto'],
                ax=ax, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    ax.set_xlabel('Activación percibida', fontsize=12)
    ax.set_ylabel('Activación objetivo', fontsize=12)
    ax.set_title('Matriz de confusión: Activación (todos los modelos)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix_arousal_overall.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Valencia confusion matrices by model
    print("[8/11] Valencia confusion matrices by model...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (modelo, label) in enumerate(zip(models, model_labels)):
        df_model = df[
            (df['modelo'] == modelo) & 
            (df['valencia_binaria_strict'].isin(['positiva', 'negativa']))
        ]
        cm = confusion_matrix(
            df_model['valencia_real_cat'], 
            df_model['valencia_binaria_strict'],
            labels=['negativa', 'positiva']
        )
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Neg', 'Pos'],
                    yticklabels=['Neg', 'Pos'],
                    ax=axes[idx], annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        axes[idx].set_title(label, fontsize=12, fontweight='bold')
        if idx == 0:
            axes[idx].set_ylabel('Valencia objetivo', fontsize=11)
        axes[idx].set_xlabel('Valencia percibida', fontsize=11)
    
    plt.suptitle('Matrices de confusión: Valencia por modelo', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix_valencia_by_model.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. Arousal confusion matrices by model
    print("[9/11] Arousal confusion matrices by model...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (modelo, label) in enumerate(zip(models, model_labels)):
        df_model = df[
            (df['modelo'] == modelo) & 
            (df['arousal_binario_strict'].isin(['alto', 'bajo']))
        ]
        cm = confusion_matrix(
            df_model['arousal_real_cat'], 
            df_model['arousal_binario_strict'],
            labels=['bajo', 'alto']
        )
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
                    xticklabels=['Bajo', 'Alto'],
                    yticklabels=['Bajo', 'Alto'],
                    ax=axes[idx], annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        axes[idx].set_title(label, fontsize=12, fontweight='bold')
        if idx == 0:
            axes[idx].set_ylabel('Activación objetivo', fontsize=11)
        axes[idx].set_xlabel('Activación percibida', fontsize=11)
    
    plt.suptitle('Matrices de confusión: Activación por modelo', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix_arousal_by_model.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 10. Boxplot: valencia responses by target and model
    print("[10/11] Boxplot: Valencia responses by target and model...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for boxplot
    plot_data = []
    for val_cat in ['negativa', 'positiva']:
        for modelo in models:
            df_subset = df[(df['valencia_real_cat'] == val_cat) & (df['modelo'] == modelo)]
            for val_resp in df_subset['valencia_resp']:
                plot_data.append({
                    'Valencia objetivo': val_cat.capitalize(),
                    'Modelo': modelo,
                    'Respuesta (1-7)': val_resp
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create boxplot
    sns.boxplot(data=plot_df, x='Valencia objetivo', y='Respuesta (1-7)', 
                hue='Modelo', ax=ax, palette=['#e78ac3', '#8da0cb', '#66c2a5'])
    ax.set_title('Respuestas de valencia por objetivo y modelo (escala 1-7)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Respuesta de valencia (1=negativa, 7=positiva)', fontsize=12)
    ax.set_xlabel('Valencia objetivo', fontsize=12)
    ax.axhline(4, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Punto neutro (4)')
    ax.legend(title='Modelo', loc='upper right')
    plt.tight_layout()
    plt.savefig(figures_dir / "boxplot_valencia_responses.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 11. Boxplot: arousal responses by target and model
    print("[11/11] Boxplot: Arousal responses by target and model...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for boxplot
    plot_data = []
    for ar_cat in ['bajo', 'alto']:
        for modelo in models:
            df_subset = df[(df['arousal_real_cat'] == ar_cat) & (df['modelo'] == modelo)]
            for ar_resp in df_subset['arousal_resp']:
                plot_data.append({
                    'Activación objetivo': ar_cat.capitalize(),
                    'Modelo': modelo,
                    'Respuesta (1-7)': ar_resp
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create boxplot
    sns.boxplot(data=plot_df, x='Activación objetivo', y='Respuesta (1-7)', 
                hue='Modelo', ax=ax, palette=['#e78ac3', '#8da0cb', '#66c2a5'])
    ax.set_title('Respuestas de activación por objetivo y modelo (escala 1-7)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Respuesta de activación (1=bajo, 7=alto)', fontsize=12)
    ax.set_xlabel('Activación objetivo', fontsize=12)
    ax.axhline(4, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Punto neutro (4)')
    ax.legend(title='Modelo', loc='upper right')
    plt.tight_layout()
    plt.savefig(figures_dir / "boxplot_arousal_responses.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ All figures saved to: {figures_dir}")
    print(f"\nGenerated figures:")
    for fig_file in sorted(figures_dir.glob("*.png")):
        print(f"  - {fig_file.name}")

if __name__ == "__main__":
    df = pd.read_excel(SURVEY_DATA)
    generate_figures(df)
