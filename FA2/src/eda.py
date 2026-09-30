"""
Exploratory Data Analysis (EDA) Module
Course: Advanced Data Science [MCA33PE17]
Student: Amit Chame | SYMCA PCCoE Pune
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "heart_disease_clean.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
METRICS_DIR = os.path.join(PROJECT_ROOT, "outputs", "metrics")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def generate_eda():
    print("--- Starting Exploratory Data Analysis ---")
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Dataset loaded for EDA. Shape: {df.shape}")

    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["target"].value_counts()
    percentages = (counts / len(df)) * 100
    bars = ax.bar(["No Disease (0)", "Disease Present (1)"], counts, color=["#2ecc71", "#e74c3c"], width=0.5, edgecolor="black")
    for bar, pct in zip(bars, percentages):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 3, f"{int(yval)} ({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(counts) + 25)
    ax.set_title("Target Distribution: Heart Disease Diagnosis (Cleveland Clinic)", pad=15)
    ax.set_ylabel("Patient Count")
    ax.set_xlabel("Diagnosis Category")
    plt.tight_layout()
    target_dist_path = os.path.join(FIGURES_DIR, "01_target_distribution.png")
    plt.savefig(target_dist_path, dpi=300)
    plt.close()
    print(f"[SAVED] {target_dist_path}")

    # 2. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, mask=mask, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix (Pearson r)", pad=15)
    plt.tight_layout()
    corr_path = os.path.join(FIGURES_DIR, "02_correlation_heatmap.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"[SAVED] {corr_path}")

    # 3. Age vs Maximum Heart Rate Achieved (thalach) by Target
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(
        data=df, x="age", y="thalach", hue="target", palette=["#2ecc71", "#e74c3c"],
        style="target", markers=["o", "X"], s=80, alpha=0.85, ax=ax
    )
    sns.regplot(data=df[df["target"] == 0], x="age", y="thalach", scatter=False, ax=ax, color="#27ae60", line_kws={"label": "Healthy Trend"})
    sns.regplot(data=df[df["target"] == 1], x="age", y="thalach", scatter=False, ax=ax, color="#c0392b", line_kws={"label": "Disease Trend"})
    ax.set_title("Age vs. Maximum Heart Rate (thalach) by Heart Disease Diagnosis", pad=15)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Maximum Heart Rate Achieved (bpm)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:4], ["Healthy (0)", "Disease (1)", "Trend (Healthy)", "Trend (Disease)"], loc="lower left")
    plt.tight_layout()
    scatter_path = os.path.join(FIGURES_DIR, "03_age_vs_thalach_by_target.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"[SAVED] {scatter_path}")

    # 4. Chest Pain Type vs Heart Disease Diagnosis
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cp_labels = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal", 4: "Asymptomatic"}
    df_cp = df.copy()
    df_cp["cp_label"] = df_cp["cp"].map(cp_labels)
    sns.countplot(data=df_cp, x="cp_label", hue="target", palette=["#2ecc71", "#e74c3c"], edgecolor="black", ax=ax)
    ax.set_title("Heart Disease Incidence by Chest Pain Type", pad=15)
    ax.set_xlabel("Chest Pain Type")
    ax.set_ylabel("Count")
    ax.legend(["Healthy (0)", "Disease (1)"], title="Diagnosis")
    plt.tight_layout()
    cp_path = os.path.join(FIGURES_DIR, "04_chest_pain_vs_target.png")
    plt.savefig(cp_path, dpi=300)
    plt.close()
    print(f"[SAVED] {cp_path}")

    # 5. Numerical Feature Distributions
    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for idx, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[idx], color="#3498db", bins=20, edgecolor="black")
        axes[idx].set_title(f"Distribution of {col.capitalize()}")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Frequency")
    
    # Use 6th subplot for oldpeak vs target boxplot
    sns.boxplot(data=df, x="target", y="oldpeak", palette=["#2ecc71", "#e74c3c"], ax=axes[5])
    axes[5].set_title("ST Depression (oldpeak) by Target")
    axes[5].set_xticklabels(["Healthy (0)", "Disease (1)"])
    axes[5].set_ylabel("ST Depression (oldpeak)")

    plt.tight_layout()
    num_dist_path = os.path.join(FIGURES_DIR, "05_numerical_distributions.png")
    plt.savefig(num_dist_path, dpi=300)
    plt.close()
    print(f"[SAVED] {num_dist_path}")

    # Summary Statistics Calculation
    summary_stats = {
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "target_balance": {
            "healthy_count": int(counts[0]),
            "disease_count": int(counts[1]),
            "healthy_pct": round(float(percentages[0]), 2),
            "disease_pct": round(float(percentages[1]), 2)
        },
        "correlations_with_target": df.corr()["target"].sort_values(ascending=False).to_dict(),
        "numerical_stats": df[num_cols].describe().to_dict()
    }

    stats_file = os.path.join(METRICS_DIR, "eda_summary.json")
    with open(stats_file, "w") as f:
        json.dump(summary_stats, f, indent=4)
    print(f"[SUCCESS] EDA statistical summary saved to: {stats_file}")
    print("--- EDA Complete ---")


if __name__ == "__main__":
    generate_eda()
