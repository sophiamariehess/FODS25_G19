# Cardiovascular Disease Prediction

## Project Overview

This project aims to predict the presence of cardiovascular disease (CVD) using clinical and demographic features through various machine learning models. The workflow includes:

- Data cleaning  
- Exploratory data analysis (EDA)  
- Handling class imbalance  
- Feature engineering and selection  
- Model training and evaluation  
- Comprehensive model comparison  

## Project Structure

The project is modular, with the following main components:

- `clean_cardio_data(data)`: Cleans the dataset (e.g., converts age, filters values, calculates BMI)  
- `explore_cardio_data(data)`: Performs EDA and generates plots/statistics  
- `check_class_balance(y)`: Checks for class imbalance  
- `split_data(X, y)`: Splits and scales data  
- `build_svm_models(...)`: Trains base, tuned, and PCA-based SVM models  
- `build_lr_models(...)`: Trains base, RFE-based, and polynomial logistic regression models  
- `build_rf_models(...)`: Trains base, selected, and tuned Random Forest models  
- `compare_models(...)`: Compares all models with metrics, ROC curves, and visualizations  

## Dependencies

The following libraries are used:

- `numpy`  
- `pandas`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`  
- `scipy`  

Install them using:

```bash
pip install -r requirements.txt
```

## Data Cleaning

Cleaning includes:

- Converting age (days to years)  
- Filtering invalid values for blood pressure, height, weight, and BMI  
- Calculating BMI  
- Dropping irrelevant columns (`id`)  
- Saving cleaned data to:  
  `output/cardio_cleaned_normal.csv`

## Exploratory Data Analysis (EDA)

EDA includes:

- Target distribution plot (`cardio`)  
- Correlation matrix  
- Boxplots (numerical features vs. `cardio`)  
- Bar plots (categorical features vs. `cardio`)  
- Saved plots in the `output/` directory  

## Handling Class Imbalance

The `check_class_balance()` function assesses class distribution. If imbalance is found (≥ 1.5 ratio), models use `class_weight='balanced'`.

## Model Training

### Support Vector Machine (SVM)

- **Base**: Linear kernel  
- **Tuned**: RBF kernel, manual grid search (C, gamma)  
- **PCA-based**: Trained on PCA-reduced features  

### Logistic Regression (LR)

- **Base**  
- **RFE-selected**: Recursive feature elimination + tuned  
- **Polynomial**: Degree 2 polynomial interactions + tuned  

### Random Forest (RF)

- **Base**  
- **Selected**: Based on feature importances (≥ 5%)  
- **Tuned**: Hyperparameter search via `GridSearchCV`  

## Model Comparison

Each model is evaluated on:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC AUC  
- ROC curves (train/test)  
- Feature count  
- Overfitting analysis  

## Output Files

All generated files are stored in the `output/` directory:

| File                                       | Description                            |
|-------------------------------------------|----------------------------------------|
| `cardio_cleaned_normal.csv`               | Cleaned dataset                        |
| `cardio_target_distribution.png`          | Class distribution                     |
| `boxplots_numerical_features.png`         | Boxplots of numerical features         |
| `barplots_categorical_features.png`       | Bar plots for categorical features     |
| `correlation_matrix.png`                  | Feature correlation matrix             |
| `feature_importances_RF_initial.png`      | RF feature importances                 |
| `model_comparison.png`                    | Bar charts comparing model metrics     |
| `roc_curves_all_models.png`               | Combined ROC curves                    |
| `model_comparison.csv`                    | Evaluation metrics summary             |

## How to Run

1. Place the dataset (`2025_cardio_train.csv`) in the root directory.  
2. Ensure an `output/` folder exists (it will be created if missing).  
3. Run the main script:

```bash
python your_script_name.py
```

> Replace `your_script_name.py` with the actual filename (e.g., `cardio_prediction.py`).
