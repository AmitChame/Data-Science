# CardioPredict: Machine Learning & Streamlit Deployment for Cardiovascular Disease Risk Prediction

**Course:** Advanced Data Science [MCA33PE17]  
**Formative Assessment:** FA2 – Data Analysis Case Study  
**Institution:** Pimpri Chinchwad College of Engineering (PCCoE), Pune  
**Department:** Department of Master of Computer Applications (MCA)  
**Student Name:** Amit Chame  
**Academic Year:** 2026 – 2027 | **Class:** SYMCA Semester I  
**Course Teacher:** Prof. Prakash Ukhalkar  

---

## 1. Problem Statement & Case Study Overview
Cardiovascular diseases (CVDs) are the leading cause of mortality globally, claiming an estimated 17.9 million lives each year according to the World Health Organization (WHO). Traditional gold-standard clinical diagnostics (such as coronary angiography) are invasive, expensive, and carry procedural risks.

This case study designs, trains, evaluates, and deploys an end-to-end Machine Learning decision-support system that predicts coronary artery disease risk from routine non-invasive clinical biomarkers and stress test parameters.

---

## 2. Dataset Description & Clinical Features
- **Dataset:** Cleveland Heart Disease Dataset
- **Repository:** UC Irvine Machine Learning Repository (Dataset ID: 45)
- **Source:** Cleveland Clinic Foundation (Dr. Robert Detrano, M.D., Ph.D.)
- **URL:** [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Sample Size:** 303 patient records, 14 features (13 predictors + 1 binary target)
- **Target Distribution:** 164 Healthy (54.1%) vs 139 Heart Disease Present (45.9%)

### Clinical Feature Dictionary:
| Feature | Type | Description | Measurement / Category |
| :--- | :--- | :--- | :--- |
| `age` | Numeric | Patient age | Years (29 – 77) |
| `sex` | Categorical | Biological sex | 1 = Male; 0 = Female |
| `cp` | Categorical | Chest pain type | 1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal, 4: Asymptomatic |
| `trestbps` | Numeric | Resting blood pressure | mm Hg upon hospital admission (94 – 200) |
| `chol` | Numeric | Serum cholesterol | mg/dl (126 – 564) |
| `fbs` | Categorical | Fasting blood sugar > 120 mg/dl | 1 = True (elevated); 0 = False |
| `restecg` | Categorical | Resting electrocardiogram | 0 = Normal, 1 = ST-T wave abnormality, 2 = LV Hypertrophy |
| `thalach` | Numeric | Maximum heart rate achieved | bpm during treadmill exercise stress test (71 – 202) |
| `exang` | Categorical | Exercise-induced angina | 1 = Yes; 0 = No |
| `oldpeak` | Numeric | ST depression | Induced by exercise relative to rest (0.0 – 6.2 mm) |
| `slope` | Categorical | Slope of peak exercise ST | 1 = Upsloping, 2 = Flat, 3 = Downsloping |
| `ca` | Numeric | Major vessels colored | 0 to 3 vessels by fluoroscopy (imputed) |
| `thal` | Categorical | Thalassemia status | 3 = Normal, 6 = Fixed defect, 7 = Reversible defect (imputed) |
| `target` | Binary Target | Coronary artery disease | 0 = Healthy (<50% narrowing), 1 = Disease (>50% narrowing) |

---

## 3. Machine Learning Algorithms Implemented
We implemented **five distinct supervised classification algorithms** covering diverse theoretical paradigms:
1. **Logistic Regression:** Linear probabilistic log-odds classifier with L2 regularization.
2. **Decision Tree Classifier:** Non-parametric recursive splitting model (`max_depth=5`).
3. **Random Forest Classifier (Baseline):** Bagging ensemble of 100 randomized decision trees.
4. **Support Vector Machine (SVM):** Non-linear margin classifier with Radial Basis Function (RBF) kernel.
5. **k-Nearest Neighbors (k-NN):** Instance-based metric learner ($k=5$).
6. **Random Forest Classifier (Tuned):** Champion model optimized via **GridSearchCV** over 288 hyperparameter combinations across 5-Fold Stratified Cross-Validation (**1,440 fits**).

---

## 4. Hyperparameter Tuning Results (GridSearchCV)
- **Target Model:** Random Forest Classifier
- **Cross-Validation:** 5-Fold Stratified K-Fold (`cv=5`, `scoring="accuracy"`)
- **Search Space:**
  - `n_estimators`: [50, 100, 150, 200]
  - `max_depth`: [3, 5, 8, None]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]
  - `criterion`: ['gini', 'entropy']
- **Optimal Parameters Found:**  
  `{'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 50}`
- **Cross-Validation Accuracy:** Rose from **80.16%** (Default) to **82.64%** (Tuned).
- **Test Set Accuracy:** Increased from **88.52%** to **90.16%**.
- **Test Set Recall:** **92.86%** (identified 26 of 28 cardiac patients).
- **ROC-AUC Score:** **0.9643**.

---

## 5. Model Evaluation Summary Table (Held-Out Test Set: 61 Samples)
| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** 🏆 | **90.16%** | **0.8667** | **0.9286** | **0.8966** | **0.9643** |
| **k-Nearest Neighbors** | 88.52% | 0.8000 | 1.0000 | 0.8889 | 0.9232 |
| **Random Forest (Default)** | 88.52% | 0.8387 | 0.9286 | 0.8814 | 0.9518 |
| **Logistic Regression** | 86.89% | 0.8125 | 0.9286 | 0.8667 | 0.9513 |
| **Support Vector Machine** | 85.25% | 0.8065 | 0.8929 | 0.8475 | 0.9437 |
| **Decision Tree** | 78.69% | 0.7273 | 0.8571 | 0.7869 | 0.8047 |

---

## 6. Project Directory Structure
```
FA2/
├── data/
│   ├── raw/
│   │   └── heart_disease_raw.csv         # Direct UCI repository download
│   └── processed/
│       ├── heart_disease_clean.csv       # Imputed and cleaned dataset
│       ├── X_train.csv                   # 80% training features
│       ├── X_test.csv                    # 20% testing features
│       ├── y_train.csv                   # Training target labels
│       └── y_test.csv                    # Testing target labels
│
├── notebooks/
│   └── FA2_Data_Analysis.ipynb           # Complete executed Jupyter Notebook (57 cells)
│
├── src/
│   ├── __init__.py                       # Package initializer
│   ├── data_preprocessing.py             # Data loading, imputation, scaling, split
│   ├── eda.py                            # Generates 5 EDA charts & summary stats
│   ├── train_models.py                   # 5 baseline models + GridSearchCV
│   └── evaluate_models.py                # Test set evaluation, metrics, and plots
│
├── models/
│   ├── preprocessor.pkl                  # Fitted StandardScaler object
│   ├── logistic_regression.pkl           # Saved Logistic Regression model
│   ├── decision_tree.pkl                 # Saved Decision Tree model
│   ├── random_forest.pkl                 # Saved Baseline Random Forest
│   ├── svm.pkl                           # Saved Support Vector Machine
│   ├── knn.pkl                           # Saved k-Nearest Neighbors
│   ├── best_random_forest_tuned.pkl      # Saved Tuned Random Forest (Champion)
│   └── full_pipeline_best.pkl            # End-to-end Pipeline (Scaler + Model)
│
├── outputs/
│   ├── figures/                          # Publication-grade figures (300 DPI)
│   │   ├── 01_target_distribution.png
│   │   ├── 02_correlation_heatmap.png
│   │   ├── 03_age_vs_thalach_by_target.png
│   │   ├── 04_chest_pain_vs_target.png
│   │   ├── 05_numerical_distributions.png
│   │   ├── 06_confusion_matrices.png
│   │   ├── 07_roc_curves.png
│   │   ├── 08_model_comparison_bar.png
│   │   ├── 09_feature_importance.png
│   │   └── 10_streamlit_app_overview.png
│   ├── metrics/                          # Quantitative JSON and CSV metrics
│   │   ├── eda_summary.json
│   │   ├── cv_results.json
│   │   ├── grid_search_results.json
│   │   ├── model_comparison.json
│   │   └── model_comparison.csv
│   └── results/
│       └── classification_reports.txt    # Per-model classification reports
│
├── app.py                                # Production Streamlit web application
├── requirements.txt                      # Project dependency specification
├── README.md                             # Comprehensive technical documentation
├── FINAL_CHECKLIST.md                    # Rubric compliance checklist
└── report/
    └── FA2_Data_Analysis_Report.pdf      # Complete 12-page PDF academic report
```

---

## 7. Installation & Setup Instructions

### Prerequisites
- Python 3.10 or Python 3.11 recommended.

### Step 1: Open Terminal in Project Root
```bash
cd C:\MCA\DataScience\FA2
```

### Step 2: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Data Preprocessing Pipeline
```bash
python src/data_preprocessing.py
```

### Step 4: Run Exploratory Data Analysis (EDA)
```bash
python src/eda.py
```

### Step 5: Train Models & Run GridSearchCV
```bash
python src/train_models.py
```

### Step 6: Evaluate Models & Generate Figures
```bash
python src/evaluate_models.py
```

---

## 8. How to Run the Streamlit Application
Launch the interactive web application by running:
```bash
streamlit run app.py
```
Then open your web browser at:
`http://localhost:8501`

### Application Functionality:
1. **Interactive Form:** 3-column input form with sliders, number inputs, selectboxes, and radio buttons.
2. **Dynamic Preprocessing:** Applies the exact `preprocessor.pkl` fitted on training data.
3. **Model Selection:** Switch between all 6 models with live accuracy and ROC-AUC indicators.
4. **Instant Diagnostic Result:** Shows High/Low Risk banner, exact disease probability %, confidence progress bar, and key risk drivers.
5. **Multi-Model Consensus:** Expandable drawer comparing predictions across all 6 models for the current patient.
6. **Analytics Tabs:** Dedicated tabs for Model Comparison & ROC curves, Exploratory Data Analysis, and Academic Rubric details.

---

## 9. How to Open the Jupyter Notebook
```bash
jupyter notebook notebooks/FA2_Data_Analysis.ipynb
```
All 57 cells are pre-executed with cell outputs, tables, and visualization plots preserved.

---

## 10. Sample Test Profiles for Viva Demonstration

### Profile 1: High-Risk Cardiac Patient
- **Inputs:** Age: 67 | Sex: Male | Chest Pain: Asymptomatic (Type 4) | Resting BP: 160 | Cholesterol: 286 | Fasting BS: True (>120) | Resting ECG: LV Hypertrophy (2) | Max HR: 108 | Exercise Angina: Yes (1) | ST Depression: 2.6 | ST Slope: Flat (2) | Colored Vessels: 3 | Thalassemia: Reversible (7)
- **Predicted Diagnosis:** **High Risk / Heart Disease Detected**
- **Calculated Probability:** **88.7% Disease Probability**
- **Multi-Model Agreement:** 6 / 6 Models predict disease.

### Profile 2: Healthy Baseline Individual
- **Inputs:** Age: 38 | Sex: Female | Chest Pain: Non-Anginal (Type 3) | Resting BP: 115 | Cholesterol: 180 | Fasting BS: False | Resting ECG: Normal (0) | Max HR: 178 | Exercise Angina: No (0) | ST Depression: 0.0 | ST Slope: Upsloping (1) | Colored Vessels: 0 | Thalassemia: Normal (3)
- **Predicted Diagnosis:** **Low Risk / No Heart Disease Detected**
- **Calculated Probability:** **7.4% Disease Probability (92.6% Healthy Confidence)**
- **Multi-Model Agreement:** 6 / 6 Models predict healthy.