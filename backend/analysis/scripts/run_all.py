#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py
Main script to run the complete human evaluation analysis pipeline
"""

import sys
from pathlib import Path

# Add scripts to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "analysis" / "scripts"))

# Import all analysis modules
import pandas as pd
from importlib import import_module

def run_pipeline():
    """Run the complete analysis pipeline"""
    
    print("\n" + "="*70)
    print(" "*15 + "HUMAN EVALUATION ANALYSIS PIPELINE")
    print("="*70 + "\n")
    
    # Load data
    SURVEY_DATA = PROJECT_ROOT / "backend" / "survey" / "survey_long_cleaned.xlsx"
    print(f"Loading data from: {SURVEY_DATA}\n")
    df = pd.read_excel(SURVEY_DATA)
    
    # Define analysis scripts in order
    scripts = [
        ("01_load_and_validate", "Data validation"),
        ("02_descriptive_stats", "Descriptive statistics"),
        ("03_accuracy_and_confusion", "Accuracy metrics and confusion matrices"),
        ("04_model_comparison", "Model comparison"),
        ("05_demographic_analysis", "Demographic analysis"),
        ("06_generate_figures", "Figure generation"),
        ("07_export_latex_tables", "LaTeX table export"),
    ]
    
    results = {}
    
    for i, (script_name, description) in enumerate(scripts, 1):
        print("\n" + "="*70)
        print(f"STEP {i}/{len(scripts)}: {description}")
        print("="*70 + "\n")
        
        try:
            # Import and run the module
            module = import_module(script_name)
            
            # Each module has a main function that takes df
            if hasattr(module, 'load_and_validate'):
                results[script_name] = module.load_and_validate()
            elif hasattr(module, 'compute_descriptive_stats'):
                results[script_name] = module.compute_descriptive_stats(df)
            elif hasattr(module, 'compute_accuracy_metrics'):
                results[script_name] = module.compute_accuracy_metrics(df)
            elif hasattr(module, 'compare_models'):
                results[script_name] = module.compare_models(df)
            elif hasattr(module, 'demographic_analysis'):
                results[script_name] = module.demographic_analysis(df)
            elif hasattr(module, 'generate_figures'):
                results[script_name] = module.generate_figures(df)
            elif hasattr(module, 'export_latex_tables'):
                results[script_name] = module.export_latex_tables(df)
            
            print(f"\n✓ Step {i} completed successfully")
            
        except Exception as e:
            print(f"\n✗ Error in step {i}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Final summary
    print("\n" + "="*70)
    print(" "*20 + "PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")
    
    OUTPUT_DIR = PROJECT_ROOT / "backend" / "analysis" / "outputs"
    
    print("OUTPUT FILES:")
    print("\nTables:")
    tables_dir = OUTPUT_DIR / "tables"
    if tables_dir.exists():
        for f in sorted(tables_dir.glob("*")):
            print(f"  ✓ {f.name}")
    
    print("\nFigures:")
    figures_dir = OUTPUT_DIR / "figures"
    if figures_dir.exists():
        for f in sorted(figures_dir.glob("*.png")):
            print(f"  ✓ {f.name}")
    
    print("\nLaTeX tables:")
    if tables_dir.exists():
        for f in sorted(tables_dir.glob("latex_*.tex")):
            print(f"  ✓ {f.name}")
    
    print("\n" + "="*70)
    print("\nNext steps:")
    print("  1. Review outputs in: backend/analysis/outputs/")
    print("  2. Update CAPITULO_5.tex with results")
    print("  3. Update CAPITULO_4.tex with methodology (if needed)"
    print("\n" + "="*70 + "\n")
    
    return True

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
