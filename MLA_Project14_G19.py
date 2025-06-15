########################################## Libraries ##########################################
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE 
from sklearn.decomposition import PCA

########################################## General Functions ##########################################
# Data Cleaning
def clean_cardio_data(data):
    # Cleans the cardio dataset by performing: Conversion of age from days to years; Filtering unrealistic values for blood pressure, height, weight, and BMI; BMI calculation; Removal of unnecessary columns

    print("### Cleaning Cardio Data ###")
    data_copy = data.copy()

    # Convert age from days to years
    data_copy['age_years'] = (data_copy['age'] / 365).astype(int)

    # Filter unrealistic blood pressure values
    before_rows = data_copy.shape[0]
    data_copy = data_copy[
        (data_copy["ap_hi"] >= 70) & (data_copy["ap_hi"] <= 250) &
        (data_copy["ap_lo"] >= 40) & (data_copy["ap_lo"] <= 150)
    ]
    after_rows = data_copy.shape[0]
    print(f"Dropped {before_rows - after_rows} rows due to blood pressure filtering.")

    # Calculate BMI
    data_copy['bmi'] = data_copy['weight'] / ((data_copy['height'] / 100) ** 2)

    # Filter based on height, weight, and BMI ranges
    before_rows = data_copy.shape[0]
    data_copy = data_copy[
        (data_copy["height"] >= 100) & (data_copy["height"] <= 220) &
        (data_copy["weight"] >= 40) & (data_copy["weight"] <= 200) &
        (data_copy["bmi"] >= 15) & (data_copy["bmi"] <= 60)
    ]
    after_rows = data_copy.shape[0]
    print(f"Dropped {before_rows - after_rows} rows due to height, weight, and BMI filtering.")

    # Drop unnecessary columns
    data_copy = data_copy.drop(columns=["age", "id"])

    print("\nMissing values after cleaning:")
    print(data_copy.isna().sum())
    print(f"Shape of cleaned data: {data_copy.shape}")

    return data_copy

# Exploratory Data Analysis (EDA)
def explore_cardio_data(data):
    print("\n### Exploratory Data Analysis (EDA) ###")
    data_copy = data.copy()

    # Basic overview
    print("\nFirst 5 rows:")
    print(data_copy.head())
    print("\nDataset shape:", data_copy.shape)
    print("\nMissing values per column:")
    print(data_copy.isna().sum())
    print("\nDescriptive statistics:")
    print(data_copy.describe())

    # Plot Distribution of Target Variable
    plt.figure(figsize=(6, 4))
    sns.countplot(x='cardio', data=data_copy)
    plt.title('Distribution of Target Variable (cardio)')
    plt.xlabel('Cardiovascular Disease')
    plt.ylabel('Count')
    plt.savefig("output/cardio_target_distribution.png")
    plt.close()

    # Correlation matrix
    plt.figure(figsize=(12, 10))
    corr = data_copy.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.savefig("output/correlation_matrix.png")
    plt.close()

    # Create boxplots and barplots for numerical and categorical features
    numerical_features = ["age_years", "height", "weight", "bmi", "ap_hi", "ap_lo"]
    categorical_features = ["cholesterol", "gluc", "gender", "smoke", "alco", "active"]
    #for numerical features 
    plt.figure(figsize=(12, 5 * len(numerical_features)))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(len(numerical_features), 1, i)
        sns.boxplot(x='cardio', y=feature, data=data_copy)
        plt.title(f'{feature} vs Cardio')

    plt.tight_layout()
    plt.savefig("output/boxplots_numerical_features.png")
    plt.close()

    #for categorical features
    #Set up figure size: one subplot per feature
    plt.figure(figsize=(12, 4 * len(categorical_features)))
    for i, feature in enumerate(categorical_features, 1):
        plt.subplot(len(categorical_features), 1, i)
        sns.barplot(x=feature, y='cardio', data=data_copy, estimator=np.mean)
        plt.title(f'Mean Cardio Rate by {feature}')
        plt.ylim(0, 1)
        plt.ylabel("Proportion with Cardio")
        plt.xlabel(feature)

    plt.tight_layout()
    plt.savefig("output/barplots_categorical_features.png")
    plt.close()

# Train-Test Split and Feature Scaling
def split_data(X, y, test_size=0.2, random_state=42):
    # Splits the dataset into training and testing sets, and applies feature scaling
    print("\n### Splitting Data into Train and Test Sets ###")

    # Split the data using stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # Apply standard scaling to avoid data leakage
    scaler = StandardScaler()

    # Fit the scaler on training data and transform both train and test sets
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Check Class Balance
def check_class_balance(y):
    print("\n### Checking Class Balance ###")
    class_counts = pd.Series(y).value_counts()
    class_proportion = class_counts / len(y)

    print("Class counts:")
    print(class_counts)
    print("\nClass proportions:")
    print(class_proportion)

    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"\nImbalance ratio (majority:minority): {imbalance_ratio:.2f}")

    return imbalance_ratio > 1.5  # Return True if imbalanced (using 1.5 as the threshold)

# Model Comparison
def evaluate_model(model, features, preprocessing_info, X, y, model_name, dataset_name):
    # preprocessing step if needed
    if preprocessing_info is None:
        X_subset = X[features]
    elif 'pca_transformer' in preprocessing_info:
        pca_transformer = preprocessing_info['pca_transformer']
        
        # Transform the input data using PCA
        X_transformed = pca_transformer.transform(X)
        
        # Create DataFrame with PCA feature names
        X_subset = pd.DataFrame(
            X_transformed, 
            columns=features,
            index=X.index
        )
    elif 'poly_transformer' in preprocessing_info:
        poly_transformer = preprocessing_info['poly_transformer']
        poly_scaler = preprocessing_info['poly_scaler']
        
        # Transform to polynomial features
        X_poly = poly_transformer.transform(X)
        X_poly_scaled = poly_scaler.transform(X_poly)
        
        # Create DataFrame with polynomial feature names
        X_subset = pd.DataFrame(
            X_poly_scaled, 
            columns=features, 
            index=X.index
        )
    else:
        #check for errors
        raise ValueError(f"Unknown preprocessing type in preprocessing_info: {preprocessing_info}")
    
    # Get predictions and probabilities
    y_pred = model.predict(X_subset)
    y_prob = model.predict_proba(X_subset)[:, 1]
    
    # Calculate all metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "Features_Used": len(features),
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "ROC_AUC": auc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Probabilities": y_prob 
    }

def compare_models(X_train_scaled, y_train, X_test_scaled, y_test, svm_models, lr_models, rf_models):
    print("\n### Start Model Comparison ###")

    # Unpack model tuples
    svm_base, svm_tuned, svm_pca = svm_models
    lr_base, lr_rfe, lr_poly = lr_models
    rf_base, rf_selected, rf_tuned = rf_models

    # Define all models
    all_models = {
        "SVM (Base - Linear)": svm_base,
        "SVM (Tuned - RBF)": svm_tuned,
        "SVM (PCA + Tuned)": svm_pca,
        "LR (Base)": lr_base,           
        "LR (RFE)": lr_rfe,           
        "LR (Polynomial)": lr_poly,     
        "RF (Base)": (rf_base[0], rf_base[1], None),        
        "RF (Selected)": (rf_selected[0], rf_selected[1], None), 
        "RF (Tuned)": (rf_tuned[0], rf_tuned[1], None)  
    }
    
    # Evaluate all models
    all_results = []
    roc_data = {}  # Store ROC curve data
    
    for model_name, (model, features, preprocessing_info) in all_models.items():
        print(f"\nEvaluating {model_name}")
        print(f"   Features: {len(features)}")
        # Handle special preprocessing
        if preprocessing_info is not None:
            if 'pca_transformer' in preprocessing_info:
                # PCA case: transform data
                pca_transformer = preprocessing_info['pca_transformer']
                X_train_transformed = pca_transformer.transform(X_train_scaled)
                X_test_transformed = pca_transformer.transform(X_test_scaled)
                
                # Create DataFrames with PCA feature names
                X_train_eval = pd.DataFrame(X_train_transformed, columns=features, index=X_train_scaled.index)
                X_test_eval = pd.DataFrame(X_test_transformed, columns=features, index=X_test_scaled.index)
                
            elif 'poly_transformer' in preprocessing_info:
                # Polynomial case: use existing logic
                poly_transformer = preprocessing_info['poly_transformer']
                poly_scaler = preprocessing_info['poly_scaler']
                
                X_train_poly = poly_transformer.transform(X_train_scaled)
                X_train_poly_scaled = poly_scaler.transform(X_train_poly)
                X_train_eval = pd.DataFrame(X_train_poly_scaled, columns=features, index=X_train_scaled.index)
                
                X_test_poly = poly_transformer.transform(X_test_scaled)
                X_test_poly_scaled = poly_scaler.transform(X_test_poly)
                X_test_eval = pd.DataFrame(X_test_poly_scaled, columns=features, index=X_test_scaled.index)
        else:
            # Standard case: use features as-is
            X_train_eval = X_train_scaled[features]
            X_test_eval = X_test_scaled[features]
        
        # Training evaluation
        train_result = evaluate_model(
            model, features, None,
            X_train_eval, y_train, model_name, "Train"
        )
        
        # Test evaluation
        test_result = evaluate_model(
            model, features, None,
            X_test_eval, y_test, model_name, "Test"
        )
        
        # Store results
        all_results.extend([train_result, test_result])
        
        # Store ROC data for plotting
        roc_data[model_name] = {
            'train_probs': train_result['Probabilities'],
            'test_probs': test_result['Probabilities'],
            'train_auc': train_result['ROC_AUC'],
            'test_auc': test_result['ROC_AUC']
        }
        
        # Show overfitting check
        overfitting = train_result['ROC_AUC'] - test_result['ROC_AUC']
        print(f"   Train AUC: {train_result['ROC_AUC']:.4f}")
        print(f"   Test AUC:  {test_result['ROC_AUC']:.4f}")
        print(f"   Overfitting: {overfitting:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Create summary plots    
    # Performance Metrics Comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Create pivot table for plotting
        pivot_data = results_df.pivot_table(
            index='Model', columns='Dataset', values=metric, aggfunc='first'
        )
        
        # Plot grouped bar chart
        pivot_data.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.legend(['Train', 'Test'])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    # Features used comparison
    ax = axes[1, 2]
    test_results = results_df[results_df['Dataset'] == 'Test']
    feature_counts = test_results.set_index('Model')['Features_Used']
    bars = feature_counts.plot(kind='bar', ax=ax, color='lightblue')
    ax.set_title('Number of Features Used')
    ax.set_ylabel('Feature Count')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('output/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curves Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training ROC curves
    for model_name, data in roc_data.items():
        fpr, tpr, _ = roc_curve(y_train, data['train_probs'])
        ax1.plot(fpr, tpr, label=f"{model_name} (AUC={data['train_auc']:.3f})")
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax1.set_title('ROC Curves - Training Set')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Test ROC curves
    for model_name, data in roc_data.items():
        fpr, tpr, _ = roc_curve(y_test, data['test_probs'])
        ax2.plot(fpr, tpr, label=f"{model_name} (AUC={data['test_auc']:.3f})")
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax2.set_title('ROC Curves - Test Set')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/roc_curves_all_models.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    results_df.to_csv("output/model_comparison.csv", index=False)
    print("\nModel comparison completed.")

    return results_df, roc_data
    
########################################## All Models ##########################################
#### SVM ####
# SVM BASE MODEL 
def train_svm_basemodel(X_train_scaled, y_train):
    # Trains a simple base SVM model using a linear kernel on pre-scaled data.
    print("\n### Training base SVM model (linear kernel) ###")

    # Initialize model with a linear kernel
    base_model = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')

    # Train the model
    base_model.fit(X_train_scaled, y_train)

    return base_model
# SVM with Manual Grid Search
def tune_svm_hyperparameters(X, y, C_values, gamma_values):
    # Manually tunes hyperparameters using cross-validation and evaluates performance using ROC AUC
    best_auc = 0
    best_params = {}
    best_model = None

    # Use stratified 5-fold cross-validation to preserve class ratios
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

    # Grid search over all combinations of C and gamma
    for C in C_values:
        for gamma in gamma_values:
            auc_scores = []

            # Cross-validation loop
            for train_index, val_index in skf.split(X, y):
                # Split data into train and validation folds
                X_train_fold = X.iloc[train_index]
                X_val_fold = X.iloc[val_index]
                y_train_fold = y.iloc[train_index]
                y_val_fold = y.iloc[val_index]

                # Scale features based only on the training fold
                scaler = StandardScaler()
                X_train_fold_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train_fold),
                    columns=X_train_fold.columns,
                    index=X_train_fold.index
                )
                X_val_fold_scaled = pd.DataFrame(
                    scaler.transform(X_val_fold),
                    columns=X_val_fold.columns,
                    index=X_val_fold.index
                )

                # Train SVM with current hyperparameters
                clf = SVC(
                    C=C,
                    gamma=gamma,
                    kernel='rbf',
                    probability=True,
                    class_weight='balanced'
                )
                clf.fit(X_train_fold_scaled, y_train_fold)

                # Predict probabilities for ROC AUC evaluation
                y_val_prob = clf.predict_proba(X_val_fold_scaled)[:, 1]
                auc = roc_auc_score(y_val_fold, y_val_prob)
                auc_scores.append(auc)

            # Calculate mean AUC across folds
            mean_auc = np.mean(auc_scores)
            print(f"C={C}, gamma={gamma} → Mean AUC: {mean_auc:.4f}")

            # Track best model
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = {'C': C, 'gamma': gamma}
                best_model = clf 

    print(f"\nBest Parameters: {best_params} with AUC: {best_auc:.4f}")
    print("\nReturning best model from manual tuning:")
    print(f"C: {best_params.get('C')}, gamma: {best_params.get('gamma')}, AUC: {best_auc:.4f}")

    return best_model, best_params, best_auc
# SVM with PCA (Manual Grid Search)
def train_svm_with_pca_manual(X_train_scaled, y_train):
    # Trains an SVM with PCA (without using a pipeline) and performs a manual grid search over PCA components and SVM hyperparameters using cross-validation
    print("\n### Training SVM with PCA (Manual, No Pipeline) ###")
    best_auc = 0
    best_model = None
    best_params = {}

    # Hyperparameter search space
    pca_components = [12] # tried [1, 3, 5, 7, 9, 11, 12]
    C_values = [0.5] # tried [0.5, 0.6, 0.7, 0.8, 0.9]
    gamma_values = [0.01] # tried [0.005, 0.008, 0.01, 0.015]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

    for n_components in pca_components:
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)

        for C in C_values:
            for gamma in gamma_values:
                auc_scores = []

                for train_idx, val_idx in skf.split(X_train_pca, y_train):
                    X_fold_train = X_train_pca[train_idx]
                    X_fold_val = X_train_pca[val_idx]

                    y_fold_train = y_train[train_idx] if isinstance(y_train, np.ndarray) else y_train.iloc[train_idx]
                    y_fold_val = y_train[val_idx] if isinstance(y_train, np.ndarray) else y_train.iloc[val_idx]

                    clf = SVC(
                        C=C,
                        gamma=gamma,
                        kernel='rbf',
                        probability=True,
                        class_weight='balanced'
                    )
                    clf.fit(X_fold_train, y_fold_train)

                    y_val_prob = clf.predict_proba(X_fold_val)[:, 1]
                    auc = roc_auc_score(y_fold_val, y_val_prob)
                    auc_scores.append(auc)

                mean_auc = np.mean(auc_scores)
                print(f"PCA={n_components}, C={C}, gamma={gamma} → AUC: {mean_auc:.4f}")

                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_params = {'n_components': n_components, 'C': C, 'gamma': gamma}
                    best_model = clf

    print(f"\nBest PCA Parameters: {best_params} with AUC: {best_auc:.4f}")
    return best_model, best_params, best_auc
# all SVM models: base, tuned, and PCA-tuned
def build_svm_models(X_train_scaled, y_train):
    # Builds three SVM models with different approaches: Base model (linear kernel), Hyperparameter tuned model using existing tune_svm_hyperparameters function, PCA + tuned model using existing train_svm_with_pca_manual function
    # 1. Base SVM Model (Linear Kernel)
    print("\n### SVM 1: Base Model ###")
    base_svm = train_svm_basemodel(X_train_scaled, y_train)

    # 2. Hyperparameter Tuned SVM using existing function
    print("\n### SVM 2: Hyperparameter Tuned ###")
    
    # Define search space
    C_values = [1.0, 10.0] # tried [0.1, 0.5, 1.0, 10.0]
    gamma_values = [0.01, 0.1] # tried [0.005, 0.01, 0.05, 0.1]
    
    tuned_svm, best_params, best_auc = tune_svm_hyperparameters(
        X_train_scaled, y_train, C_values, gamma_values
    )

    # 3. PCA + TUNED SVM using existing function
    print("\n### SVM 3: PCA + Tuned ###")
    pca_svm, pca_params, pca_auc = train_svm_with_pca_manual(X_train_scaled, y_train)
    
    # Create PCA feature names based on the best n_components
    n_components = pca_params['n_components']
    pca_feature_names = [f'PC{i+1}' for i in range(n_components)]
    
    # Create PCA transformer for evaluation (recreate the best PCA)
    final_pca = PCA(n_components=n_components, random_state=42)
    final_pca.fit(X_train_scaled)
    
    return ((base_svm, list(X_train_scaled.columns), None), (tuned_svm, list(X_train_scaled.columns), None), (pca_svm, pca_feature_names, {'pca_transformer': final_pca}))

#### LOGISTIC REGRESSION ####
# Logistic regression with manual feature optimization 
def build_lr_models(X_train_scaled, y_train):
    # Builds three logistic regression models with Manual feature optimization: Base model, RFE selected features & tuned (manual optimal number of features), Polynomial features & tuned (manual optimal degree)

    # 1. Base Model
    print("\n### LR 1: Base Model ###")
    base_LR = LogisticRegression(
        solver='lbfgs',
        penalty='l2',
        C=1.0,
        max_iter=3000,
        random_state=42
    )
    base_LR.fit(X_train_scaled, y_train)
    print(f"Base model trained with {X_train_scaled.shape[1]} features")

    # 2. RFE selected features & tuned with manual optimization
    print("\n### LR 2: RFE Feature Selection (manual) ###")

    rfe_results = []
    base_estimator = LogisticRegression(solver='lbfgs', max_iter=3000, random_state=42)

    param_grid_rfe = [
        {
            'C': [0.1, 1.0, 10.0, 100.0],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga'],
            'class_weight': [None, 'balanced']
        },
        {
            'C': [0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1'],
            'solver': ['liblinear', 'saga'],
            'class_weight': [None, 'balanced']
        }
    ]

    for n_features in [12, 11, 10, 9, 8]:
        print(f"\n### RFE with {n_features} features ###")
        rfe = RFE(estimator=base_estimator, n_features_to_select=n_features, step=1)
        rfe.fit(X_train_scaled, y_train)

        selected_features = X_train_scaled.columns[rfe.support_].tolist()
        print(f"Selected features ({n_features}): {selected_features}")

        X_train_rfe = X_train_scaled[selected_features]

        grid_search_rfe = GridSearchCV(
            estimator=LogisticRegression(max_iter=3000, random_state=42),
            param_grid=param_grid_rfe,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='roc_auc'
        )

        grid_search_rfe.fit(X_train_rfe, y_train)

        print(f"Best params (RFE {n_features} features):", grid_search_rfe.best_params_)
        print(f"Best AUC (RFE {n_features} features):", grid_search_rfe.best_score_)

        rfe_results.append((grid_search_rfe.best_estimator_, selected_features, n_features, grid_search_rfe.best_score_))

    # Select best RFE model
    rfe_tuned_LR, selected_features, selected_n_features, selected_auc = max(rfe_results, key=lambda x: x[3])
    print(f"\nSelected RFE model: {selected_n_features} features, AUC = {selected_auc:.4f}")

    # 3. Polynomial Features & Tuned with Manual Optimization
    print("\n### LR 3: Polynomial Features (manual) ###")

    poly_results = []
    param_grid_poly = [
        {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga'],
            'class_weight': [None, 'balanced']
        },
        {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1'],
            'solver': ['liblinear', 'saga'],
            'class_weight': [None, 'balanced']
        }
    ]

    for degree in [2]:
        print(f"\n### Polynomial degree {degree} ###")

        poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)

        print(f"Polynomial features created: {X_train_poly.shape[1]} features (degree {degree})")

        poly_scaler = StandardScaler()
        X_train_poly_scaled = poly_scaler.fit_transform(X_train_poly)

        poly_feature_names = poly.get_feature_names_out(X_train_scaled.columns)
        X_train_poly_df = pd.DataFrame(X_train_poly_scaled, columns=poly_feature_names, index=X_train_scaled.index)

        grid_search_poly = GridSearchCV(
            estimator=LogisticRegression(max_iter=5000, random_state=42),
            param_grid=param_grid_poly,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='roc_auc'
        )

        grid_search_poly.fit(X_train_poly_df, y_train)

        print(f"Best params (Poly degree {degree}):", grid_search_poly.best_params_)
        print(f"Best AUC (Poly degree {degree}):", grid_search_poly.best_score_)

        poly_results.append((grid_search_poly.best_estimator_, poly_feature_names, degree, grid_search_poly.best_score_, poly, poly_scaler))

    # Select best polynomial model
    poly_tuned_LR, poly_feature_names, selected_degree, selected_auc, best_poly, best_poly_scaler = max(poly_results, key=lambda x: x[3])

    print(f"\nSelected Polynomial model: degree {selected_degree}, AUC = {selected_auc:.4f}")

    return (
        (base_LR, list(X_train_scaled.columns), None),
        (rfe_tuned_LR, selected_features, None),
        (poly_tuned_LR, list(poly_feature_names), {'poly_transformer': best_poly, 'poly_scaler': best_poly_scaler})
    )

#### RANDOM FOREST ####
# Random Forest Models: Base, Feature Selection and Tune Selected Model
def build_rf_models(X_train_scaled, y_train):
    # Builds three Random Forest models: Base model, Feature-selected model, and Feature-selected + tuned model
    # 1. Base Model
    print("\n### Random Forest Base Model ###")
    base_RF = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    base_RF.fit(X_train_scaled, y_train)

    # Feature Selection using feature importances
    print("\n### Feature Selection for Random Forest ###")
    feature_importances = base_RF.feature_importances_

    # Create DataFrame with feature importances
    feature_importance = pd.DataFrame({
        'Feature': X_train_scaled.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print("Feature importances:")
    print(feature_importance)

    # Plot feature importance for initial model
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title("Feature Importance in Base Random Forest Model")
    plt.tight_layout()
    plt.savefig("output/feature_importances_RF_initial.png")
    plt.close()

    # Select top features
    threshold = 0.05 # chose 0.05 based on feature importance overview from base model
    selected_features = feature_importance[feature_importance['Importance'] >= threshold]['Feature'].tolist()

    # if too few features are selected, take top 8 to keep model manageable
    if len(selected_features) < 5:
        print(f"Only {len(selected_features)} features met threshold. Taking top 8 instead.")
        selected_features = feature_importance.head(8)['Feature'].tolist()

    print(f"Selected features: {selected_features}")

    # Filter datasets to selected features
    X_train_sel_RF = X_train_scaled[selected_features]

    # 2. Train new model on selected features
    print("\n### Random Forest with selected Features ###")
    selected_RF = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    selected_RF.fit(X_train_sel_RF, y_train)

    # Hyperparameter tuning on the selected model
    print("\n### Random Forest Hyperparameter Tuning (with selected Features) ###")
    param_grid = {
        'n_estimators': [100, 200, 300,],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [2, 5]
    }

    # 3. Use the selected model for tuning
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='roc_auc'
    )

    # Time of grid search fit on selected features
    start_time = time.time()
    grid_search.fit(X_train_sel_RF, y_train)
    end_time = time.time()
    print(f"\nGrid search completed in {(end_time - start_time)/60:.2f} minutes")

    print("Best parameters found:", grid_search.best_params_)
    print("Best AUC score found:", grid_search.best_score_)

    # Get the best model (tuned & selected) 
    tuned_RF = grid_search.best_estimator_

    return (base_RF, list(X_train_scaled.columns)), (selected_RF, selected_features), (tuned_RF, selected_features)

########################################## Main Code Execution ##########################################

# Load dataset
data = pd.read_csv("2025_cardio_train.csv")

# create output directory 
os.makedirs("output", exist_ok=True)

# clean data
data_cleaned = clean_cardio_data(data)
data_cleaned.to_csv("output/cardio_cleaned_normal.csv", index=False)
print("\nSaved normal cleaned data to 'output/cardio_cleaned_normal.csv'.")

# Exploratory Data Analysis (EDA) 
explore_cardio_data(data_cleaned)

# Feature selection 
features_to_scale = ["age_years", "height", "weight", "bmi", "ap_hi", "ap_lo", "cholesterol", "gluc", "gender", "smoke", "alco", "active", ]

X = data_cleaned[features_to_scale]
y = data_cleaned["cardio"]

# Check class balance
is_imbalanced = check_class_balance(y)
print(f"Dataset is {'imbalanced' if is_imbalanced else 'balanced'}")

# Train/test split and scaling
X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_data(X, y)

### Train Models ###
print("\n### Training Models ###")

### SVM
print("Train SVM Model...")
svm_models = build_svm_models(X_train_scaled, y_train)

### Logistic Regression
print("Train LR Model...")
lr_models = build_lr_models(X_train_scaled, y_train)

### Random Forest
print("Train Random Forest Model...")
rf_models = build_rf_models(X_train_scaled, y_train)

### Compare all models ###
results_df, roc_data = compare_models(X_train_scaled, y_train, X_test_scaled, y_test, svm_models, lr_models, rf_models)