# Allstate Claims Severity

**End-to-end machine learning for predicting insurance claim severity using exploratory data analysis, regression modeling, and cross-validation.**

## 1. Project Overview

Predicting insurance claim severity is a supervised regression problem in which the objective is to estimate the monetary loss associated with an insurance claim.

This project uses the **Allstate Claims Severity** dataset to explore claim characteristics, establish regression baselines, compare nonlinear machine learning models, and develop a reproducible prediction workflow.

The analysis follows four main stages:

1. Exploratory data analysis and data quality assessment.
2. Preprocessing and baseline regression modeling.
3. Model comparison and cross-validation.
4. Final model training, persistence, and inference.

- **Problem Type:** Supervised Regression
- **Target Variable:** `loss`
- **Primary Evaluation Metric:** Mean Absolute Error (MAE)
- **Selected Model:** CatBoost Regressor

## 2. Dataset

**Source:** [Kaggle — Allstate Claims Severity](https://www.kaggle.com/competitions/allstate-claims-severity)

Each observation represents an insurance claim. The objective is to predict the `loss` associated with that claim using anonymized categorical and continuous features.

| Dataset Property | Value |
|---|---:|
| Training observations | 188,318 |
| Unlabeled test observations | 125,546 |
| Categorical features | 116 |
| Continuous features | 14 |
| Total predictive features | 130 |

The dataset includes:

- `id`: Unique record identifier, excluded from model training.
- `cat1`–`cat116`: Categorical predictors.
- `cont1`–`cont14`: Continuous predictors.
- `loss`: Target variable available in the training dataset.

### Data Availability

The original dataset is not distributed in this repository.

Download `train.csv` and `test.csv` from the dataset source and place them in:

```text
data/raw/
├── train.csv
└── test.csv
```

The raw data directory is excluded from Git tracking.

## 3. Project Structure

```text
allstate-claims-severity/
├── data/
│   └── raw/                         # Local datasets (Git ignored)
├── models/
│   └── catboost_final.cbm           # Trained CatBoost model
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_preprocessing_baseline.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_final_model_inference.ipynb
├── .gitignore
├── pyproject.toml                   # Ruff configuration
├── requirements.txt
└── README.md
```

### Notebook Guide

| Notebook | Description |
|---|---|
| `01_data_overview.ipynb` | Data quality checks, target analysis, and feature exploration |
| `02_preprocessing_baseline.ipynb` | Train-validation split, categorical encoding, and baseline regression |
| `03_model_comparison.ipynb` | CatBoost and XGBoost experiments, generalization analysis, and cross-validation |
| `04_final_model_inference.ipynb` | Final CatBoost training, prediction validation, model persistence, and reload verification |

## 4. Exploratory Data Analysis

The exploratory analysis examines the dataset structure, target distribution, potential data quality issues, and relationships between the available predictors and claim severity.

### Key Findings

- No missing values or duplicate records were detected.
- Claim severity follows a strongly right-skewed distribution, with skewness of approximately **3.79**.
- Approximately **6.14%** of training observations are identified as statistical outliers using the interquartile range (IQR) method.
- These high-loss observations account for approximately **22.91%** of the total claim amount.
- Individual continuous features exhibit relatively weak linear correlations with the target.
- Several categorical features contain many distinct categories, including infrequent values.

### EDA Decisions

High-loss observations were retained because they represent valid and potentially important claim outcomes rather than confirmed data errors.

The exploratory findings also motivated the evaluation of nonlinear models capable of capturing complex relationships and interactions between features.

## 5. Preprocessing and Baseline Modeling

An initial **80/20 train-validation split** was created using `random_state=42`.

The baseline workflow included:

- A median-based dummy regressor as a reference benchmark.
- One-Hot Encoding for categorical variables.
- Ridge Regression using a scikit-learn preprocessing pipeline.
- Non-negative clipping of Ridge predictions to enforce a valid claim severity range.

Categorical encoding was configured with `handle_unknown="ignore"` to accommodate previously unseen categories without causing prediction errors.

All preprocessing transformations were fitted on the relevant training data to avoid information leakage.

## 6. Model Comparison

Three modeling approaches were evaluated:

**Ridge Regression:** A regularized linear baseline using One-Hot Encoding.

**CatBoost Regressor:** A gradient boosting model that supports categorical predictors directly and captures nonlinear relationships.

**XGBoost Regressor:** A gradient boosting model trained using a One-Hot Encoding pipeline.

### Holdout Validation Results

All results below use Mean Absolute Error (MAE), where lower values indicate better predictive performance.

| Model | Validation MAE |
|---|---:|
| Dummy Regressor (Median) | 1,796.96 |
| Ridge Regression + Clipping | 1,288.44 |
| CatBoost — 500 Iterations | 1,147.31 |
| XGBoost — 500 Estimators | 1,145.58 |
| **CatBoost — 1,000 Iterations** | **1,138.68** |

CatBoost with 1,000 boosting iterations achieved the lowest MAE on the original validation split.

Compared with the clipped Ridge baseline, this represents an improvement of approximately **11.62%** in validation MAE.

The model comparison also showed that nonlinear gradient boosting approaches substantially outperformed the linear baseline.

## 7. Cross-Validation

To evaluate the stability of the leading models, **3-fold cross-validation** was performed on the training portion of the original holdout split.

The same shuffled folds (`random_state=42`) were used for both CatBoost and XGBoost.

Each model was trained independently within each fold. For XGBoost, categorical preprocessing was also fitted independently to prevent data leakage.

### Cross-Validation Results

| Fold | CatBoost MAE | XGBoost MAE |
|---|---:|---:|
| Fold 1 | 1,159.84 | 1,167.98 |
| Fold 2 | 1,146.62 | 1,154.28 |
| Fold 3 | 1,156.71 | 1,164.56 |
| **Mean MAE** | **1,154.39** | 1,162.28 |
| Standard Deviation | **5.64** | 5.82 |

### Model Selection

CatBoost achieved a lower MAE in all three folds and reduced mean cross-validation MAE by approximately **7.89** compared with XGBoost.

Both models demonstrated relatively consistent results across the evaluated folds.

CatBoost was selected as the final model based on its stronger validation performance and consistent cross-validation results.

The performance difference between the two gradient boosting models is modest and should not be interpreted as evidence that CatBoost will outperform XGBoost on every dataset or under every configuration.

## 8. Final Model and Inference

The selected CatBoost configuration was retrained on all **188,318 labeled training observations**.

### Final Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | CatBoostRegressor |
| Iterations | 1,000 |
| Learning Rate | 0.05 |
| Tree Depth | 6 |
| Loss Function | MAE |
| Random Seed | 42 |
| CPU Threads | 2 |

The final trained model is stored at:

```text
models/catboost_final.cbm
```

### Prediction Validation

The final model generated predictions for all **125,546 unlabeled test observations**.

The prediction validation checks confirmed that:

- All expected records received a prediction.
- No missing or non-finite predictions were produced.
- One negative prediction was identified and clipped to zero.
- All post-processed predictions satisfied the non-negative claim severity constraint.

### Model Persistence

The trained model was saved using CatBoost's native `.cbm` format.

Reload verification confirmed that the saved model reproduced the original test predictions exactly, with a maximum absolute difference of **0.0**.

Non-negative clipping is performed separately as part of the inference workflow.

## 9. Installation

### Prerequisites

- Python 3.12
- Git
- Jupyter Notebook or VS Code with the Jupyter extension

### Clone the Repository

```bash
git clone https://github.com/masoud-vosoughi/allstate-claims-severity.git
cd allstate-claims-severity
```

### Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Place the downloaded dataset files in `data/raw/` before running the notebooks.

### Run the Notebooks

Open the project in VS Code or Jupyter, select the virtual environment as the Python kernel, and run the notebooks in numerical order.

The notebooks document the complete workflow from data exploration through final model training and inference.

## 10. Using the Saved Model

The saved CatBoost model can be loaded directly without repeating model training.

Run the following example from the repository root after installing dependencies and placing `test.csv` in `data/raw/`:

```python
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

# Load the trained model
model = CatBoostRegressor()
model.load_model("models/catboost_final.cbm")

# Load unlabeled observations
test_data = pd.read_csv("data/raw/test.csv")

# Preserve the expected feature order
X_test = test_data[model.feature_names_]

# Generate predictions
predictions = model.predict(X_test)

# Enforce non-negative claim severity
predictions = np.clip(predictions, a_min=0, a_max=None)

# Inspect predictions alongside record identifiers
results = pd.DataFrame(
    {
        "id": test_data["id"],
        "predicted_loss": predictions,
    }
)

print(results.head())
```

This example uses the feature names stored with the trained model to maintain the expected input schema.

For the complete training, persistence, and verification workflow, see `notebooks/04_final_model_inference.ipynb`.

## 11. Evaluation Considerations

The reported validation and cross-validation results describe model performance on labeled subsets of the original training dataset.

The provided unlabeled test dataset does not contain target values; therefore, MAE cannot be calculated for those observations locally.

Additionally, the original validation split was used during model experimentation. Its best observed MAE should not be interpreted as an independent final test estimate.

Cross-validation provides an additional assessment of model stability, although results remain specific to the evaluated data, model configurations, and validation strategy.

## 12. Technology Stack

- **Language:** Python 3.12
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, CatBoost, XGBoost
- **Development:** Jupyter Notebook, VS Code
- **Code Quality:** Ruff
- **Version Control:** Git, GitHub

## 13. Conclusion

This project demonstrates a complete regression modeling workflow for insurance claim severity prediction.

Exploratory analysis identified a strongly skewed target distribution and a mixture of categorical and continuous predictors. Baseline experiments established the limitations of a linear model, while gradient boosting methods substantially reduced validation error.

CatBoost was selected following holdout validation and cross-validation, achieving a **mean 3-fold CV MAE of 1,154.39**.

The final model was trained on the complete labeled dataset, saved in a reusable format, and verified for consistent inference.

Potential future improvements include systematic hyperparameter optimization, expanded cross-validation, feature engineering, and additional analysis of prediction errors across claim severity ranges.