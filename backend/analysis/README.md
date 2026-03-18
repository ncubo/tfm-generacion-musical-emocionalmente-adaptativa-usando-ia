# Human Evaluation Analysis Pipeline

This directory contains the complete analysis pipeline for the human evaluation study of the emotion-conditioned music generation system.

## Structure

```
backend/analysis/
├── scripts/                  # Analysis scripts
│   ├── 01_load_and_validate.py
│   ├── 02_descriptive_stats.py
│   ├── 03_accuracy_and_confusion.py
│   ├── 04_model_comparison.py
│   ├── 05_demographic_analysis.py
│   ├── 06_generate_figures.py
│   ├── 07_export_latex_tables.py
│   └── run_all.py           # Main pipeline coordinator
├── outputs/                  # Generated outputs
│   ├── tables/              # CSV tables and LaTeX tables
│   └── figures/             # PNG figures for thesis
└── README.md                # This file
```

## Data Source

The analysis uses: `backend/survey/survey_long_cleaned.xlsx`

This dataset contains 372 responses from participants who evaluated 12 audio fragments (4 per model: baseline, pretrained, finetuned).

## Key Variables

- **participante_id**: Unique participant identifier
- **audio_id**: Audio fragment identifier
- **modelo**: Generation engine (baseline, pretrained, finetuned)
- **valencia_real_cat**: Ground truth valence (positiva/negativa)
- **arousal_real_cat**: Ground truth arousal (alto/bajo)
- **valencia_resp**: Valence response on 1-7 scale
- **arousal_resp**: Arousal response on 1-7 scale
- **valencia_binaria_strict**: Categorized valence response (positiva/negativa/neutra)
- **arousal_binario_strict**: Categorized arousal response (alto/bajo/neutro)
- **genero**: Participant gender
- **edad**: Participant age
- **conocimiento_musical**: Musical knowledge (1-5 scale)

## Running the Analysis

### Quick Start (Run Everything)

```bash
cd backend/analysis/scripts
python run_all.py
```

This will execute all analysis steps and generate all outputs.

### Run Individual Scripts

Each script can be run independently:

```bash
# 1. Validate data
python 01_load_and_validate.py

# 2. Descriptive statistics
python 02_descriptive_stats.py

# 3. Accuracy and confusion matrices
python 03_accuracy_and_confusion.py

# 4. Model comparison
python 04_model_comparison.py

# 5. Demographic analysis
python 05_demographic_analysis.py

# 6. Generate figures
python 06_generate_figures.py

# 7. Export LaTeX tables
python 07_export_latex_tables.py
```

## Analysis Overview

### 1. Data Validation
- Checks data integrity
- Validates ranges and categories
- Reports missing values

### 2. Descriptive Statistics
- Sample size and demographics
- Response distributions
- Basic summaries

### 3. Accuracy Metrics
- Overall accuracy for valence and arousal
- Accuracy by model
- Accuracy by target emotional class
- Confusion matrices

### 4. Model Comparison
- Statistical comparison between baseline, pretrained, and finetuned
- Chi-square tests
- Confidence intervals
- Mean responses by model

### 5. Demographic Analysis
- Exploratory analysis by gender
- Exploratory analysis by musical knowledge
- Note: Small sample sizes, interpret cautiously

### 6. Figures
Generates 11 publication-ready figures:
- Demographics (gender, age, musical knowledge)
- Accuracy bar charts (valence, arousal by model)
- Confusion matrices (overall and by model)
- Boxplots (responses by target and model)

### 7. LaTeX Tables
Exports 6 LaTeX-formatted tables ready for thesis inclusion:
- Demographics summary
- Overall accuracy
- Accuracy by model
- Confusion matrices (valence, arousal)
- Mean responses by target

## Key Findings (To Be Computed)

The pipeline computes:
- **Valence perception accuracy**: % of correct valence classifications
- **Arousal perception accuracy**: % of correct arousal classifications
- **Neutral response rate**: % of responses classified as neutral
- **Model rankings**: Which model performs best for each dimension
- **Confusion patterns**: Common misclassifications

## Outputs for Thesis

### Figures
Ready to include in Chapter 5 using:
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{../backend/analysis/outputs/figures/accuracy_valencia_by_model.png}
\caption{...}
\label{fig:accuracy_valencia}
\end{figure}
```

### Tables
Ready to include in Chapter 5 using:
```latex
\input{../backend/analysis/outputs/tables/latex_accuracy_overall.tex}
```

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- openpyxl (for reading Excel files)

## Notes

- All scripts are reproducible and deterministic
- Neutral responses are excluded from accuracy calculations but reported separately
- Demographic analyses are exploratory due to small sample sizes
- Statistical tests assume independence of observations (participants × audios)
- All outputs use UTF-8 encoding for Spanish text compatibility

## Author

Miguel Mayorga
Master's Thesis Project
Emotion-Conditioned Music Generation Using AI
