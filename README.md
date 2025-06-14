# Cardiovascular Disease Prediction

## Project Overview

This project aims to predict the presence of cardiovascular disease (CVD) based on clinical and demographic features using machine learning. The project involves data cleaning, exploratory data analysis (EDA), balancing the dataset, feature selection, model training, evaluation, and comparison using several machine learning algorithms.

## Project Structure

The code is modular and organized into the following steps:

- `clean_cardio_data(data)`: Cleans the raw dataset (e.g., converts age, filters implausible values, calculates BMI)
- `explore_cardio_data(data)`: Performs EDA, generating plots and summary statistics
- `check_class_balance(y)`: Assesses class imbalance
- `split_data(X, y)`: Splits and scales features using `StandardScaler`
- `evaluation_metrics(...)`: Calculates key evaluation metrics and plots confusion matrix and ROC curve
- `compare_all_models_comprehensive(...)`: Runs a full comparison across models with metrics and ROC curves

Model-specific training includes:

- `train_svm_basemodel(...)`
- `tune_svm_hyperparameters(...)`
- `train_svm_with_pca_manual(...)`
- `build_lr_models(...)`
- `build_best_rf(...)

## Dependencies

Install all required packages using the provided `requirements.txt`. Core libraries include:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy

To install:

```bash
pip install -r requirements.txt
```

## Data Cleaning

The cleaning process includes:
- Removing invalid blood pressure, height, weight, and BMI values
- Converting age from days to years
- Dropping unnecessary columns and missing values
- Creating a BMI column

## Exploratory Data Analysis (EDA)

EDA includes:
- Count plots of all variables
- Target variable distribution
- Heatmaps and boxplots
- Correlation matrix
- Summary statistics and feature relationship insights

## ⚖Handling Class Imbalance

Class imbalance is checked using the `check_class_balance()` function. If imbalance is detected, stratified splitting and model balancing (e.g., `class_weight='balanced'`) is applied.

## Model Training and Evaluation

Machine learning models trained and evaluated:
- Support Vector Machine (SVM)
- Logistic Regression (with RFE and polynomial features)
- Random Forest (base, selected, and tuned)

Each model undergoes:
- Feature selection
- Hyperparameter tuning
- Confusion matrix & ROC curve plotting
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC AUC

## Model Comparison

Comprehensive comparison includes:
- Evaluation on train and test sets
- Overfitting analysis
- Bar plots for all metrics
- ROC curve visualization
- Final ranking based on ROC AUC

## Output Files

All output files are saved to the `output/` folder:

- Cleaned dataset (`cardio_cleaned_normal_final.csv`)
- Target distribution & feature plots
- Confusion matrices (per model)
- ROC curves (individual + combined)
- Feature importances (Random Forest)
- Performance summary (`comprehensive_model_comparison_final.csv`)
- Final plots:
  - `comprehensive_model_comparison_final.png`
  - `roc_curves_all_models_final.png`

## How to Run

1. Place the dataset (`2025_cardio_train.csv`) in the root directory.
2. Ensure an `output/` folder exists (will be created if missing).
3. Run the main script:

```bash
python your_script_name.py
```

> Replace `your_script_name.py` with the actual filename (e.g., `cardio_prediction.py`).
