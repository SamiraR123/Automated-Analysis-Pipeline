# Automated Analysis Pipeline: Diabetes Risk Classification
## Author
Samira Rahman

## Purpose
This notebook demonstrates an automated, adaptive analytical workflow for
health data science: automated data ingestion, missing-value handling,
exploratory data analysis, inferential statistics, and supervised machine
learning classification, applied to the Pima Indians Diabetes Database.
The workflow adjusts its behavior based on dataset characteristics (file
format, which columns have missing data, how outcome labels are encoded)
rather than being hard-coded to one fixed dataset shape.

## How to Run
1. Click the "Open in Colab" badge below, or open the notebook directly in
   Google Colab.
2. Run the first cell and upload `Example_Dataset_Diabetes.csv` when prompted.
3. Run all remaining cells in order (Runtime > Run all). No manual edits are
   required for the notebook to complete successfully.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m4iqiR7x2T08iPHsCwBqsKM53YBCgdWo?usp=sharing)

## Dataset
Pima Indians Diabetes Database (NIH-NIDDK), 768 rows, 9 columns: diagnostic
measurements from women aged 21+ of Pima Indian heritage, with a binary
outcome indicating diabetes diagnosis. Included in this repository as
`Example_Dataset_Diabetes.csv`. Used here for instructional purposes only.

## Modifications from the Original Pipeline
1. **Readability**: consolidated the repeated significance-test decision
   logic (one-sample t-test, independent t-test, ANOVA, chi-square) into a
   single `summarize_significance()` helper function.
2. **Extended conditional logic + added model**: added K-Nearest Neighbors
   and Random Forest to the automated model comparison; widened the
   scaling-decision logic to correctly scale KNN's features.
3. **Visualization**: added a grouped bar chart comparing all four models
   across accuracy, recall, precision, F1, and ROC AUC.
4. **Extended conditional logic**: added missingness-severity flagging
   (High/Low, 30% threshold) to the missing-value detection step.

## Requirements
- Python 3.x
- pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels
- phik (installed automatically by the notebook via pip)

All of the above are pre-installed in Google Colab except `phik`, which the
notebook installs itself in the relevant cell.

## Assumptions and Limitations
- This dataset is synthetic/instructional and is **not suitable for clinical
  inference or decision-making**.
- Biologically impossible zero values (Glucose, blood pressure, skin
  thickness, insulin, BMI) are treated as missing and imputed with the
  training-set median; this is a simplification for instructional purposes.
- Model comparisons use a single train/test split (70/30, stratified,
  `random_state=42`); results are not cross-validated except for the
  decision tree's hyperparameter search.

  ## GenAI Disclosure
  AI assistance (Claude) was used to help with the python code and organize this README. All code execution and final analysis are my own.
