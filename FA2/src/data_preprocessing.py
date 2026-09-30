"""
Data Acquisition & Leakage-Free Preprocessing Pipeline
Course: Advanced Data Science [MCA33PE17]
Student: Amit Chame | SYMCA PCCoE Pune
Academic Year: 2026-2027 | Semester: I
"""

import os
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "heart_disease_raw.csv"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "heart_disease_clean.csv"
MODELS_DIR = BASE_DIR / "models"

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)",
    "trestbps": "Resting blood pressure in mm Hg on admission to the hospital",
    "chol": "Serum cholesterol in mg/dl",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: LV hypertrophy)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)",
    "ca": "Number of major vessels (0-3) colored by fluoroscopy",
    "thal": "Thalassemia status (3: normal, 6: fixed defect, 7: reversible defect)",
    "target": "Diagnosis of heart disease (0: < 50% diameter narrowing / healthy, 1: > 50% narrowing / disease)"
}


def download_raw_data(save_path=RAW_DATA_PATH):
    """Acquires raw Cleveland Heart Disease dataset from UCI ML Repository."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        print(f"[INFO] Raw dataset already exists at: {save_path}")
        return pd.read_csv(save_path)
    
    print(f"[INFO] Downloading dataset from UCI Repository: {UCI_URL}")
    try:
        urllib.request.urlretrieve(UCI_URL, save_path)
        print(f"[SUCCESS] Downloaded raw data to: {save_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download from UCI: {e}")
        raise
    
    df = pd.read_csv(save_path, names=FEATURE_NAMES, na_values="?")
    df.to_csv(save_path, index=False)
    return df


def load_and_preprocess(raw_path=RAW_DATA_PATH, test_size=0.20, random_state=42):
    """
    Leak-Free Data Preprocessing Workflow:
    1. Load raw data and replace '?' with NaN.
    2. Convert types and binarize target variable (0: healthy, 1: disease).
    3. Separate X and y.
    4. Split into train and test sets (80/20 stratified).
    5. Fit SimpleImputer and StandardScaler STRICTLY on X_train.
    6. Transform X_train and X_test using the fitted pipeline.
    7. Save processed splits and fitted preprocessor.
    """
    if not raw_path.exists():
        download_raw_data(raw_path)

    df = pd.read_csv(raw_path)
    df = df.replace("?", np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binarize target
    df["target"] = (df["target"] > 0).astype(int)

    # Deduplicate
    df = df.drop_duplicates().reset_index(drop=True)

    # Separate features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Stratified 80/20 train/test split BEFORE any imputation or scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Split shapes: X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"[INFO] Missing in X_train: {dict(X_train.isnull().sum()[X_train.isnull().sum() > 0])}")
    print(f"[INFO] Missing in X_test:  {dict(X_test.isnull().sum()[X_test.isnull().sum() > 0])}")

    # Build Pipeline: SimpleImputer (mode) + StandardScaler
    # Fitted ONLY on X_train to prevent data leakage
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler())
    ])
    preprocessor.set_output(transform="pandas")

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Save fitted preprocessor
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = MODELS_DIR / "preprocessor.pkl"
    joblib.dump(preprocessor, scaler_path)
    print(f"[SUCCESS] Fitted leak-free preprocessor saved to: {scaler_path}")

    # Save processed datasets
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train_proc.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test_proc.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    X_train.to_csv(PROCESSED_DIR / "X_train_raw.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test_raw.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    # Save a clean dataset with imputed values for EDA plotting
    # Imputed using X_train imputer to maintain consistency
    df_imputed = df.copy()
    df_imputed[X.columns] = preprocessor.named_steps["imputer"].transform(X)
    df_imputed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[SUCCESS] Clean dataset saved to: {PROCESSED_DATA_PATH}")

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor


if __name__ == "__main__":
    print("--- Running Leak-Free Data Preprocessing ---")
    load_and_preprocess()
    print("Preprocessing completed successfully.")