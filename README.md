# Allstate Claims Severity

Predicting insurance claim severity using exploratory data analysis and machine learning.

## Project Overview

This project explores the Allstate Claims Severity dataset to understand insurance claim patterns and develop regression models for predicting claim amounts.

- **Problem Type:** Supervised Regression
- **Target Variable:** `loss`
- **Primary Evaluation Metric:** Mean Absolute Error (MAE)

## Dataset

Source: [Kaggle — Allstate Claims Severity](https://www.kaggle.com/competitions/allstate-claims-severity)

- Training samples: 188,318
- Test samples: 125,546
- Categorical features: 116
- Continuous features: 14

The dataset files are not included in this repository.

Place `train.csv` and `test.csv` inside `data/raw/` before running the notebooks.

## Project Structure

```text
allstate-claims-severity/
├── data/
│   └── raw/                   # Local datasets (Git ignored)
├── notebooks/
│   └── 01_data_overview.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/masoud-vosoughi/allstate-claims-severity.git
cd allstate-claims-severity

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
```

Open the project in VS Code and select the `.venv` environment as the Jupyter kernel.

## Exploratory Data Analysis

Completed analyses:

- Dataset structure and data quality assessment
- Missing values and duplicate detection
- Target distribution and descriptive statistics
- Skewness and outlier analysis
- High-loss claim contribution analysis

### Key Findings

- No missing values or duplicate records were detected.
- The target distribution is strongly right-skewed (skewness ≈ 3.79).
- Approximately 6.14% of claims are flagged as statistical outliers using IQR.
- These claims account for approximately 22.91% of the total claim amount.

## Project Status

**In Progress**

Next steps include numerical and categorical feature analysis, correlation analysis, preprocessing, baseline modeling, and model evaluation.

## Tech Stack

Python 3.12, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook, Git.