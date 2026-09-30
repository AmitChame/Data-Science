"""
Model Evaluation & Comparison Module (Leak-Free)
Course: Advanced Data Science [MCA33PE17]
Student: Amit Chame | SYMCA PCCoE Pune
Academic Year: 2026-2027 | Semester: I
"""

import os
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
RESULTS_DIR = BASE_DIR / "outputs" / "results"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_all_models():
    print("--- Starting Comprehensive Model Evaluation ---")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")

    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest (Default)": "random_forest.pkl",
        "Support Vector Machine": "svm.pkl",
        "k-Nearest Neighbors": "knn.pkl",
        "Tuned Random Forest": "best_random_forest_tuned.pkl"
    }

    results = []
    confusion_matrices = {}
    roc_data = {}
    reports_text = []

    for name, fname in model_files.items():
        m_path = MODELS_DIR / fname
        clf = joblib.load(m_path)

        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_proba = clf.decision_function(X_test)
        else:
            y_proba = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, target_names=["Healthy (0)", "Disease (1)"])

        confusion_matrices[name] = cm
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc)

        results.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4)
        })

        report_entry = (
            f"========================================\n"
            f"Model: {name}\n"
            f"========================================\n"
            f"{cr}\nConfusion Matrix:\n{cm}\n\n"
        )
        reports_text.append(report_entry)
        print(f"[{name:<24}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    # Save reports text
    reports_path = RESULTS_DIR / "classification_reports.txt"
    with open(reports_path, "w", encoding="utf-8") as f:
        f.writelines(reports_text)
    print(f"[SUCCESS] Classification reports saved to: {reports_path}")

    # Save Comparison Table
    df_results = pd.DataFrame(results)
    df_results.sort_values(by="F1-Score", ascending=False, inplace=True)
    csv_path = METRICS_DIR / "model_comparison.csv"
    json_path = METRICS_DIR / "model_comparison.json"
    df_results.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"[SUCCESS] Model comparison saved to: {csv_path} and {json_path}")

    # Plot 1: 6-Panel Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, (m_name, cm) in enumerate(confusion_matrices.items()):
        # Find matching accuracy
        m_acc = [r["Accuracy"] for r in results if r["Model"] == m_name][0]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Healthy", "Disease"],
                    yticklabels=["Healthy", "Disease"],
                    annot_kws={"size": 14, "weight": "bold"}, ax=axes[idx])
        axes[idx].set_title(f"{m_name}\nAccuracy: {m_acc:.2%}", fontsize=12, pad=10)
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")
    plt.tight_layout()
    cm_path = FIGURES_DIR / "06_confusion_matrices.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[SAVED] {cm_path}")

    # Plot 2: ROC Curves
    plt.figure(figsize=(9, 6.5))
    colors_list = ["#2ecc71", "#e67e22", "#3498db", "#9b59b6", "#f1c40f", "#e74c3c"]
    for idx, (m_name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        plt.plot(fpr, tpr, label=f"{m_name} (AUC = {auc_val:.3f})", color=colors_list[idx % len(colors_list)], linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)", linewidth=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("ROC Curves Comparison Across All Evaluated Models", pad=15)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    roc_path = FIGURES_DIR / "07_roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[SAVED] {roc_path}")

    # Plot 3: Performance Metrics Comparison Bar Chart
    df_plot = pd.DataFrame(results).melt(id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], var_name="Metric", value_name="Score")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x="Model", y="Score", hue="Metric", palette="tab10")
    plt.title("Machine Learning Models Evaluation Metrics Comparison", pad=15)
    plt.ylabel("Score (0.0 - 1.0)")
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=20, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    bar_path = FIGURES_DIR / "08_model_comparison_bar.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"[SAVED] {bar_path}")

    # Plot 4: Feature Importance from Tuned Random Forest
    tuned_rf = joblib.load(MODELS_DIR / "best_random_forest_tuned.pkl")
    importances = tuned_rf.feature_importances_
    features = X_test.columns
    feat_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(feat_df["Feature"], feat_df["Importance"], color="#16a085", edgecolor="black")
    plt.title("Feature Importance Ranking (Tuned Random Forest)", pad=15)
    plt.xlabel("Information Gain / Entropy Importance")
    plt.tight_layout()
    feat_path = FIGURES_DIR / "09_feature_importance.png"
    plt.savefig(feat_path, dpi=300)
    plt.close()
    print(f"[SAVED] {feat_path}")

    print("--- Model Evaluation Complete ---")
    return df_results


if __name__ == "__main__":
    evaluate_all_models()