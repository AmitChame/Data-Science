"""
Model Training & Hyperparameter Tuning Module (Leak-Free)
Course: Advanced Data Science [MCA33PE17]
Student: Amit Chame | SYMCA PCCoE Pune
Academic Year: 2026-2027 | Semester: I
"""

import os
import json
import time
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    return X_train, y_train, preprocessor


def train_baseline_models():
    """
    Trains 5 distinct ML classification models using 5-Fold Stratified Cross-Validation:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. Support Vector Machine (SVC)
    5. k-Nearest Neighbors (k-NN)
    """
    X_train, y_train, preprocessor = load_training_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "filename": "logistic_regression.pkl"
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(max_depth=5, random_state=42),
            "filename": "decision_tree.pkl"
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, random_state=42),
            "filename": "random_forest.pkl"
        },
        "Support Vector Machine": {
            "model": SVC(probability=True, random_state=42),
            "filename": "svm.pkl"
        },
        "k-Nearest Neighbors": {
            "model": KNeighborsClassifier(n_neighbors=5),
            "filename": "knn.pkl"
        }
    }

    cv_results = {}
    trained_models = {}

    print("--- Training Base Models with 5-Fold Stratified Cross-Validation ---")
    for name, item in models.items():
        clf = item["model"]
        start_time = time.time()
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
        clf.fit(X_train, y_train)
        elapsed = time.time() - start_time

        trained_models[name] = clf
        cv_results[name] = {
            "cv_mean_accuracy": round(float(scores.mean()), 4),
            "cv_std_accuracy": round(float(scores.std()), 4),
            "cv_scores": [round(float(s), 4) for s in scores],
            "training_time_seconds": round(float(elapsed), 4)
        }

        save_path = MODELS_DIR / item["filename"]
        joblib.dump(clf, save_path)
        print(f"[{name}] Saved to {item['filename']} | 5-Fold CV: {scores.mean():.4f} (+/- {scores.std():.4f}) | Time: {elapsed:.4f}s")

    return trained_models, cv_results


def tune_random_forest():
    """
    Hyperparameter tuning using GridSearchCV on Random Forest.
    Tests n_estimators, max_depth, min_samples_split, min_samples_leaf, criterion.
    """
    X_train, y_train, preprocessor = load_training_data()
    print("\n--- Hyperparameter Tuning: GridSearchCV on Random Forest ---")

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 8],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"]
    }

    rf = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time

    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = float(grid_search.best_score_)

    print(f"[GridSearchCV RF] Best Parameters: {best_params}")
    print(f"[GridSearchCV RF] Best 5-Fold CV Score: {best_score:.4f}")
    print(f"[GridSearchCV RF] Tuning Time: {tuning_time:.2f}s")

    # Save tuned model
    tuned_rf_path = MODELS_DIR / "best_random_forest_tuned.pkl"
    joblib.dump(best_rf, tuned_rf_path)
    print(f"[SUCCESS] Tuned Random Forest saved to: {tuned_rf_path}")

    # Build and save end-to-end full Pipeline
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", best_rf)
    ])
    pipeline_path = MODELS_DIR / "full_pipeline_best.pkl"
    joblib.dump(full_pipeline, pipeline_path)
    print(f"[SUCCESS] End-to-end Pipeline saved to: {pipeline_path}")

    grid_results = {
        "algorithm": "Random Forest",
        "parameters_tested": param_grid,
        "best_parameters": best_params,
        "best_cv_score": round(best_score, 4),
        "tuning_time_seconds": round(tuning_time, 2)
    }

    grid_file = METRICS_DIR / "grid_search_results.json"
    with open(grid_file, "w") as f:
        json.dump(grid_results, f, indent=4)

    return best_rf, grid_results


def run_training():
    trained_models, cv_results = train_baseline_models()
    best_rf, grid_results = tune_random_forest()

    cv_file = METRICS_DIR / "cv_results.json"
    with open(cv_file, "w") as f:
        json.dump(cv_results, f, indent=4)
    print(f"[SUCCESS] Cross-validation metrics saved to: {cv_file}")
    print("--- Model Training Complete ---")


if __name__ == "__main__":
    run_training()