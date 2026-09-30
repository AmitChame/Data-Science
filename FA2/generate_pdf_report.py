# -*- coding: utf-8 -*-
"""
FA2 Case Study PDF Report Generator
Advanced Data Science [MCA33PE17]
Student: Amit Chame | SYMCA PCCoE Pune
"""

import os
import json
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

PROJECT_ROOT = r"C:\MCA\DataScience\FA2"
REPORT_PATH = os.path.join(PROJECT_ROOT, "report", "FA2_Data_Analysis_Report.pdf")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
METRICS_DIR = os.path.join(PROJECT_ROOT, "outputs", "metrics")


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))
        self.drawString(54, 750, "PCCoE MCA | Advanced Data Science [MCA33PE17] - FA2 Case Study Report")
        self.drawRightString(558, 750, "Student: Amit Chame")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.75)
        self.line(54, 742, 558, 742)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Cardiovascular Disease Prediction using ML & Streamlit App Deployment")
        self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_pdf():
    print("--- Generating Comprehensive Academic Report (PDF) ---")
    doc = SimpleDocTemplate(
        REPORT_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CoverTitle", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=18, leading=23,
        textColor=colors.HexColor("#1A365D"), alignment=1
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle", parent=styles["Normal"],
        fontName="Helvetica", fontSize=11, leading=15,
        textColor=colors.HexColor("#2B6CB0"), alignment=1
    )
    h1_style = ParagraphStyle(
        "ReportH1", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=12.5, leading=16,
        textColor=colors.HexColor("#1A365D"), spaceBefore=11, spaceAfter=5, keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "ReportH2", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=10, leading=14,
        textColor=colors.HexColor("#2B6CB0"), spaceBefore=8, spaceAfter=3, keepWithNext=True
    )
    body_style = ParagraphStyle(
        "ReportBody", parent=styles["Normal"],
        fontName="Helvetica", fontSize=8.5, leading=12.5,
        textColor=colors.HexColor("#2D3748"), spaceAfter=5
    )
    bullet_style = ParagraphStyle(
        "ReportBullet", parent=body_style,
        leftIndent=14, firstLineIndent=-10, spaceAfter=3
    )
    callout_style = ParagraphStyle(
        "ReportCallout", parent=body_style,
        fontName="Helvetica-Oblique", fontSize=8, leading=11.5,
        textColor=colors.HexColor("#1A365D"), backColor=colors.HexColor("#EDF2F7"),
        borderColor=colors.HexColor("#CBD5E0"), borderWidth=1, borderPadding=5,
        spaceBefore=4, spaceAfter=6
    )
    table_cell = ParagraphStyle(
        "TableCell", parent=styles["Normal"],
        fontName="Helvetica", fontSize=7.5, leading=10, textColor=colors.HexColor("#2D3748")
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold", parent=table_cell,
        fontName="Helvetica-Bold", textColor=colors.HexColor("#1A365D")
    )

    story = []

    # PAGE 1: COVER PAGE
    story.append(Spacer(1, 15))
    pccoe_header = """
    <font size="12" color="#1A365D"><b>PIMPRI CHINCHWAD EDUCATION TRUST'S</b></font><br/>
    <font size="14" color="#C53030"><b>PIMPRI CHINCHWAD COLLEGE OF ENGINEERING (PCCOE)</b></font><br/>
    <font size="8.5" color="#4A5568">(An Autonomous Institute Affiliated to Savitribai Phule Pune University, SPPU)<br/>
    Sector 26, Pradhikaran, Nigdi, Pune, Maharashtra 411044</font><br/><br/>
    <font size="11" color="#2B6CB0"><b>DEPARTMENT OF MASTER OF COMPUTER APPLICATIONS (MCA)</b></font>
    """
    story.append(Paragraph(pccoe_header, ParagraphStyle("InstHeader", alignment=1, leading=15)))
    story.append(Spacer(1, 15))
    story.append(HRFlowable(width="90%", thickness=2, color=colors.HexColor("#1A365D"), spaceAfter=15))

    story.append(Paragraph("<b>FORMATIVE ASSESSMENT – 02 (FA2)</b><br/><font size=12>DATA ANALYSIS CASE STUDY REPORT</font>", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>CARDIOVASCULAR DISEASE RISK PREDICTION USING MACHINE LEARNING & STREAMLIT APP DEPLOYMENT</b>", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Course:</b> Advanced Data Science [MCA33PE17] &nbsp;|&nbsp; <b>Syllabus:</b> Unit 1 to Unit 4", subtitle_style))
    story.append(Spacer(1, 18))

    meta_table_data = [
        [Paragraph("<b>Student Name:</b>", table_cell_bold), Paragraph("Amit Chame", table_cell)],
        [Paragraph("<b>Academic Year:</b>", table_cell_bold), Paragraph("2026 – 2027", table_cell)],
        [Paragraph("<b>Class & Semester:</b>", table_cell_bold), Paragraph("SYMCA, Semester I", table_cell)],
        [Paragraph("<b>Assignment Date:</b>", table_cell_bold), Paragraph("04/09/2026", table_cell)],
        [Paragraph("<b>Submission Date:</b>", table_cell_bold), Paragraph("01/10/2026", table_cell)],
        [Paragraph("<b>Course Teacher:</b>", table_cell_bold), Paragraph("Prof. Prakash Ukhalkar", table_cell)],
        [Paragraph("<b>Maximum Marks:</b>", table_cell_bold), Paragraph("20 Marks (Converted to 10 Marks)", table_cell)]
    ]
    t_meta = Table(meta_table_data, colWidths=[140, 290])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 4.5),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 25))
    story.append(Paragraph("<b>Department of Master of Computer Applications • PCCoE Pune</b>", ParagraphStyle("CoverFooter", alignment=1, fontName="Helvetica", fontSize=8.5, textColor=colors.HexColor("#718096"))))
    story.append(PageBreak())

    # PAGE 2: CERTIFICATE, DECLARATION, ABSTRACT
    story.append(Paragraph("CERTIFICATE OF AUTHENTICITY", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=10))
    cert_text = """
    This is to certify that the FA2 Case Study report entitled <b>'Cardiovascular Disease Risk Prediction Using Machine Learning & Streamlit App Deployment'</b> 
    has been successfully completed by <b>Amit Chame</b> (SYMCA, Semester I, Academic Year 2026–2027) 
    in partial fulfillment of the academic requirements for the course <b>Advanced Data Science [MCA33PE17]</b> 
    in the Department of Master of Computer Applications, Pimpri Chinchwad College of Engineering (PCCoE), Pune.
    <br/><br/>
    The work presented herein represents genuine code execution, statistical preprocessing, cross-validation, 
    hyperparameter tuning, model benchmarking, and Streamlit application development.
    """
    story.append(Paragraph(cert_text, body_style))
    story.append(Spacer(1, 25))

    sig_data = [
        [Paragraph("<b>Amit Chame</b><br/>Student, SYMCA", table_cell), 
         Paragraph("<b>Prof. Prakash Ukhalkar</b><br/>Course Teacher", table_cell), 
         Paragraph("<b>Head of Department</b><br/>Department of MCA, PCCoE", table_cell)]
    ]
    t_sig = Table(sig_data, colWidths=[160, 160, 160])
    t_sig.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_sig)
    story.append(Spacer(1, 15))

    story.append(Paragraph("STUDENT DECLARATION", h2_style))
    decl_text = """
    I, <b>Amit Chame</b>, hereby declare that this case study report is my authentic work. 
    All machine learning models, statistical computations, hyperparameter tuning steps, exploratory graphs, 
    and the interactive Streamlit deployment script were executed and verified locally. 
    No metrics, graphs, or predictions have been artificially fabricated.
    """
    story.append(Paragraph(decl_text, body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("EXECUTIVE ABSTRACT", h2_style))
    abstract_text = """
    Cardiovascular diseases (CVDs) remain the leading contributor to human mortality globally, necessitating 
    accurate non-invasive early screening systems. This case study develops an end-to-end 
    Machine Learning workflow using the benchmark Cleveland Heart Disease dataset (303 patient records, 14 features) 
    sourced from the UC Irvine Machine Learning Repository. 
    <br/><br/>
    Five supervised learning algorithms—<b>Logistic Regression, Decision Trees, Random Forest, Support Vector Machines (SVM), 
    and k-Nearest Neighbors (k-NN)</b>—were evaluated under 5-Fold Stratified Cross-Validation. Extensive hyperparameter 
    tuning via <b>GridSearchCV (1,440 fits)</b> was conducted on Random Forest, elevating test accuracy to <b>90.16%</b>, 
    with a recall of <b>92.86%</b> and ROC-AUC of <b>0.9643</b>. 
    The end-to-end model and preprocessor are serialized and operationalized inside an interactive <b>Streamlit</b> 
    web application supporting real-time risk assessment, confidence scoring, and multi-model consensus.
    """
    story.append(Paragraph(abstract_text, body_style))
    story.append(PageBreak())    # PAGE 3: INTRODUCTION, PROBLEM STATEMENT, OBJECTIVES & DATASET
    story.append(Paragraph("1. INTRODUCTION & PROBLEM STATEMENT", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=8))
    p1_text = """
    Cardiovascular diseases claim approximately 17.9 million lives each year according to the World Health Organization (WHO), 
    representing 32% of all global fatalities. Timely and accurate detection is critical to initiating lifestyle and therapeutic 
    interventions before irreversible cardiac damage occurs. Conventional gold-standard diagnostic techniques, specifically coronary 
    angiography, are invasive, resource-intensive, expensive, and entail procedural risks. 
    <br/><br/>
    <b>Problem Statement:</b> To develop an accurate, transparent, and reproducible Machine Learning classification framework 
    that analyzes routinely collected non-invasive clinical biomarkers to predict the presence of significant coronary artery disease, 
    and to deploy this diagnostic system via an accessible Streamlit web application.
    """
    story.append(Paragraph(p1_text, body_style))

    story.append(Paragraph("2. PROJECT OBJECTIVES", h2_style))
    story.append(Paragraph("• <b>Data Preprocessing & Integrity:</b> Acquire benchmark clinical data, impute missing values, binarize target diagnoses, and standardize features strictly on training splits without data leakage.", bullet_style))
    story.append(Paragraph("• <b>Exploratory Analysis (EDA):</b> Uncover diagnostic distributions, correlation patterns, physiological trends, and symptom associations with clinical interpretations.", bullet_style))
    story.append(Paragraph("• <b>Algorithm Benchmarking:</b> Implement five distinct scikit-learn classifiers spanning linear, tree-based, ensemble, kernel, and distance-based paradigms.", bullet_style))
    story.append(Paragraph("• <b>Hyperparameter Optimization:</b> Apply exhaustive GridSearchCV over 288 hyperparameter combinations (1,440 fits) to regularize and optimize tree ensembles.", bullet_style))
    story.append(Paragraph("• <b>Model Serialization & Deployment:</b> Save pipelines via joblib and engineer an interactive Streamlit clinical decision support dashboard.", bullet_style))

    story.append(Paragraph("3. DATASET DESCRIPTION & CLINICAL ATTRIBUTES", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=8))
    story.append(Paragraph("""
    The investigation utilizes the celebrated <b>Cleveland Heart Disease Dataset</b> curated at the Cleveland Clinic Foundation 
    by Dr. Robert Detrano, M.D., Ph.D., hosted by the <b>UC Irvine Machine Learning Repository</b> (Dataset ID: 45).
    """, body_style))

    dataset_table_rows = [
        [Paragraph("<b>Attribute</b>", table_cell_bold), Paragraph("<b>Type</b>", table_cell_bold), Paragraph("<b>Description & Values</b>", table_cell_bold)],
        [Paragraph("<code>age</code>", table_cell), Paragraph("Numeric", table_cell), Paragraph("Patient age in years (Range: 29 – 77)", table_cell)],
        [Paragraph("<code>sex</code>", table_cell), Paragraph("Binary", table_cell), Paragraph("1 = Male; 0 = Female", table_cell)],
        [Paragraph("<code>cp</code>", table_cell), Paragraph("Categorical", table_cell), Paragraph("Chest Pain: 1: Typical Angina, 2: Atypical, 3: Non-Anginal, 4: Asymptomatic", table_cell)],
        [Paragraph("<code>trestbps</code>", table_cell), Paragraph("Numeric", table_cell), Paragraph("Resting blood pressure in mm Hg on admission (94 – 200 mm Hg)", table_cell)],
        [Paragraph("<code>chol</code>", table_cell), Paragraph("Numeric", table_cell), Paragraph("Serum cholesterol level in mg/dl (126 – 564 mg/dl)", table_cell)],
        [Paragraph("<code>fbs</code>", table_cell), Paragraph("Binary", table_cell), Paragraph("Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)", table_cell)],
        [Paragraph("<code>restecg</code>", table_cell), Paragraph("Categorical", table_cell), Paragraph("Resting ECG: 0 = Normal, 1 = ST-T wave abnormality, 2 = LV Hypertrophy", table_cell)],
        [Paragraph("<code>thalach</code>", table_cell), Paragraph("Numeric", table_cell), Paragraph("Maximum heart rate achieved during treadmill stress test (71 – 202 bpm)", table_cell)],
        [Paragraph("<code>exang</code>", table_cell), Paragraph("Binary", table_cell), Paragraph("Exercise-induced angina (1 = Yes, 0 = No)", table_cell)],
        [Paragraph("<code>oldpeak</code>", table_cell), Paragraph("Numeric", table_cell), Paragraph("ST depression induced by exercise relative to rest (0.0 – 6.2 mm)", table_cell)],
        [Paragraph("<code>slope</code>", table_cell), Paragraph("Categorical", table_cell), Paragraph("Slope of peak exercise ST: 1 = Upsloping, 2 = Flat, 3 = Downsloping", table_cell)],
        [Paragraph("<code>ca</code>", table_cell), Paragraph("Numeric", table_cell), Paragraph("Number of major vessels (0–3) colored by fluoroscopy (Contains 4 missing values)", table_cell)],
        [Paragraph("<code>thal</code>", table_cell), Paragraph("Categorical", table_cell), Paragraph("Thalassemia status: 3 = Normal, 6 = Fixed defect, 7 = Reversible (2 missing values)", table_cell)],
        [Paragraph("<code>target</code>", table_cell), Paragraph("Binary Target", table_cell), Paragraph("Angiographic disease diagnosis: 0 = Healthy (<50% narrowing), 1 = Disease (>50% narrowing)", table_cell)]
    ]
    t_data = Table(dataset_table_rows, colWidths=[65, 65, 370])
    t_data.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EDF2F7")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_data)
    story.append(PageBreak())

    # PAGE 4: PREPROCESSING & EDA PART 1
    story.append(Paragraph("4. DATA ACQUISITION & PREPROCESSING PIPELINE", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=8))
    prep_narrative = """
    A rigorous preprocessing methodology was implemented in <code>src/data_preprocessing.py</code>:
    <br/><br/>
    <b>1. Automated Acquisition:</b> The raw dataset is fetched directly from UCI repository and verified against 14 attribute columns.
    <br/>
    <b>2. Missing Value Imputation:</b> Missing entries in <code>ca</code> (4 records) and <code>thal</code> (2 records) were imputed 
    using column statistical modes (<code>ca_mode = 0</code>, <code>thal_mode = 3</code>), preserving all 303 patient records.
    <br/>
    <b>3. Target Binarization:</b> Angiographic disease status (0 to 4) was mapped to binary diagnosis: <code>target = (target > 0).astype(int)</code>.
    <br/>
    <b>4. Stratified Split:</b> The dataset was partitioned into <b>80% training (242 samples)</b> and <b>20% testing (61 samples)</b> 
    with stratified sampling (<code>random_state=42</code>) to maintain identical class ratios.
    <br/>
    <b>5. Feature Scaling:</b> Continuous features were standardized using <code>StandardScaler</code> fitted strictly on <code>X_train</code> 
    to guarantee zero data leakage into the evaluation set.
    """
    story.append(Paragraph(prep_narrative, body_style))

    story.append(Paragraph("5. EXPLORATORY DATA ANALYSIS (EDA) & FINDINGS", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=8))

    f1_path = os.path.join(FIGURES_DIR, "01_target_distribution.png")
    f2_path = os.path.join(FIGURES_DIR, "02_correlation_heatmap.png")
    t_eda1 = Table([[Image(f1_path, width=240, height=160), Image(f2_path, width=255, height=160)]], colWidths=[250, 260])
    t_eda1.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(t_eda1)
    story.append(Spacer(1, 4))

    story.append(Paragraph("""
    <b>Clinical Interpretation of Figures 1 & 2:</b><br/>
    • <b>Target Distribution:</b> The cohort contains 164 healthy subjects (54.13%) and 139 heart disease subjects (45.87%). 
    The balanced distribution enables reliable evaluation using Accuracy and F1-score without synthetic over-sampling.<br/>
    • <b>Correlation Matrix:</b> Thalassemia defect (<code>thal</code>, r = 0.52), major colored fluoroscopy vessels (<code>ca</code>, r = 0.46), 
    exercise angina (<code>exang</code>, r = 0.43), and ST depression (<code>oldpeak</code>, r = 0.42) are the strongest positive correlates with disease. 
    Maximum heart rate (<code>thalach</code>, r = -0.42) exhibits a strong inverse protective relationship.
    """, callout_style))
    story.append(PageBreak())

    # PAGE 5: EDA CONTINUED (FIGURES 3, 4, 5)
    story.append(Paragraph("5. EXPLORATORY DATA ANALYSIS (CONTINUED)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=8))

    f3_path = os.path.join(FIGURES_DIR, "03_age_vs_thalach_by_target.png")
    f4_path = os.path.join(FIGURES_DIR, "04_chest_pain_vs_target.png")
    t_eda2 = Table([[Image(f3_path, width=250, height=165), Image(f4_path, width=250, height=165)]], colWidths=[255, 255])
    t_eda2.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(t_eda2)
    story.append(Spacer(1, 4))

    story.append(Paragraph("""
    <b>Clinical Interpretation of Figures 3 & 4:</b><br/>
    • <b>Age vs Maximum Heart Rate:</b> While peak heart rate naturally declines with age, cardiac patients systematically plateau 
    at lower peak heart rates during stress testing across all ages (gap of ~15–20 bpm).<br/>
    • <b>Chest Pain Presentation:</b> Patients presenting with <b>Type 4 (Asymptomatic Chest Pain)</b> account for the overwhelming majority 
    of diagnosed heart disease cases (over 75% of Type 4 are disease-positive). This confirms that lack of acute angina is common in severe coronary disease ('silent ischemia').
    """, callout_style))
    story.append(Spacer(1, 8))

    f5_path = os.path.join(FIGURES_DIR, "05_numerical_distributions.png")
    story.append(Paragraph("<b>Figure 5: Numerical Feature Distributions & ST Depression Boxplot</b>", h2_style))
    story.append(Image(f5_path, width=500, height=210))
    story.append(Paragraph("""
    <b>Distribution Analysis:</b> Resting blood pressure (<code>trestbps</code>) and cholesterol (<code>chol</code>) display unimodal distributions 
    with moderate right skewness representing clinically elevated cases (cholesterol > 350 mg/dl). ST depression (<code>oldpeak</code>) 
    exhibits a pronounced elevation among cardiac patients during exercise exertion.
    """, callout_style))
    story.append(PageBreak())    # PAGE 6: MACHINE LEARNING ALGORITHMS & TRAINING
    story.append(Paragraph("6. MACHINE LEARNING ALGORITHMS & MATHEMATICAL FOUNDATION", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=8))
    story.append(Paragraph("""
    In compliance with Part 2 of the rubric, we implemented <b>five distinct machine learning algorithms</b>:
    """, body_style))

    algo_desc = [
        [Paragraph("<b>Algorithm</b>", table_cell_bold), Paragraph("<b>Paradigm & Rationale</b>", table_cell_bold), Paragraph("<b>Mathematical Formulation & Mechanics</b>", table_cell_bold)],
        [
            Paragraph("<b>Logistic Regression</b>", table_cell_bold),
            Paragraph("Linear Probabilistic Model.<br/>Provides interpretable odds ratios and log-likelihood benchmark.", table_cell),
            Paragraph("Estimates posterior probability via the logistic sigmoid:<br/>P(y=1|X) = 1 / (1 + exp(-w^T X))<br/>Fitted with L2 Ridge regularization.", table_cell)
        ],
        [
            Paragraph("<b>Decision Tree</b>", table_cell_bold),
            Paragraph("Non-parametric Partitioning.<br/>Provides transparent clinical if-then decision rules.", table_cell),
            Paragraph("Recursively partitions feature space using Gini impurity reduction:<br/>Gini(D) = 1 - sum(p_i^2)<br/>Constrained with max_depth=5.", table_cell)
        ],
        [
            Paragraph("<b>Random Forest</b>", table_cell_bold),
            Paragraph("Bagging Ensemble.<br/>Averages randomized decision trees to minimize variance and overfitting.", table_cell),
            Paragraph("Trains B bootstrap trees on random feature subsets. Aggregates outputs via majority vote:<br/>y_hat = mode{T_1(X), ..., T_B(X)}", table_cell)
        ],
        [
            Paragraph("<b>Support Vector Machine (SVM)</b>", table_cell_bold),
            Paragraph("Maximum Margin Classifier.<br/>Separates complex non-linear clinical boundaries in Hilbert space.", table_cell),
            Paragraph("Maps features using Radial Basis Function (RBF) kernel:<br/>K(x, x') = exp(-gamma * ||x - x'||^2)<br/>Calibrated with Platt scaling.", table_cell)
        ],
        [
            Paragraph("<b>k-Nearest Neighbors (k-NN)</b>", table_cell_bold),
            Paragraph("Instance-Based Memory Learner.<br/>Classifies cases based on geometric similarity to known patients.", table_cell),
            Paragraph("Computes Euclidean distance across normalized predictors:<br/>d(x, x') = sqrt(sum((x_j - x'_j)^2))<br/>Assigns majority class among k=5 neighbors.", table_cell)
        ]
    ]
    t_algo = Table(algo_desc, colWidths=[90, 140, 270])
    t_algo.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EDF2F7")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t_algo)
    story.append(Spacer(1, 10))

    story.append(Paragraph("7. MODEL TRAINING & 5-FOLD CROSS-VALIDATION", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=8))
    cv_text = """
    To prevent evaluation bias arising from an arbitrary split, every model was first evaluated using 
    <b>5-Fold Stratified Cross-Validation</b> on the training set (242 samples). 
    Stratified K-Fold maintains the 54:46 healthy-to-disease ratio across all 5 validation splits, 
    measuring out-of-fold generalization stability and preventing variance inflation.
    """
    story.append(Paragraph(cv_text, body_style))
    story.append(PageBreak())

    # PAGE 7: GRIDSEARCHCV & FEATURE IMPORTANCE
    story.append(Paragraph("8. HYPERPARAMETER TUNING VIA GRIDSEARCHCV", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=8))
    story.append(Paragraph("""
    In direct fulfillment of the FA2 rubric, exhaustive hyperparameter optimization was conducted via 
    <b>GridSearchCV</b> on the <b>Random Forest Classifier</b> across 288 distinct hyperparameter 
    combinations using 5-Fold Stratified Cross-Validation (total of <b>1,440 fits</b>).
    """, body_style))

    grid_table_data = [
        [Paragraph("<b>Hyperparameter</b>", table_cell_bold), Paragraph("<b>Search Space Explored</b>", table_cell_bold), Paragraph("<b>Optimal Value Found</b>", table_cell_bold), Paragraph("<b>Rationale & Clinical Impact</b>", table_cell_bold)],
        [Paragraph("<code>n_estimators</code>", table_cell), Paragraph("[50, 100, 150, 200]", table_cell), Paragraph("<b>50</b>", table_cell_bold), Paragraph("Provides sufficient ensemble diversity without excessive computational overhead.", table_cell)],
        [Paragraph("<code>max_depth</code>", table_cell), Paragraph("[3, 5, 8, None]", table_cell), Paragraph("<b>3</b>", table_cell_bold), Paragraph("Shallow trees enforce strong regularization, preventing memorization of patient noise.", table_cell)],
        [Paragraph("<code>min_samples_split</code>", table_cell), Paragraph("[2, 5, 10]", table_cell), Paragraph("<b>10</b>", table_cell_bold), Paragraph("Requires at least 10 samples to split an internal node, avoiding erratic fine-grained splits.", table_cell)],
        [Paragraph("<code>min_samples_leaf</code>", table_cell), Paragraph("[1, 2, 4]", table_cell), Paragraph("<b>2</b>", table_cell_bold), Paragraph("Guarantees at least 2 patients per leaf, smoothing decision thresholds.", table_cell)],
        [Paragraph("<code>criterion</code>", table_cell), Paragraph("['gini', 'entropy']", table_cell), Paragraph("<b>'entropy'</b>", table_cell_bold), Paragraph("Information gain criterion provided superior entropy reduction in this clinical feature space.", table_cell)]
    ]
    t_grid = Table(grid_table_data, colWidths=[95, 110, 80, 215])
    t_grid.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EDF2F7")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t_grid)
    story.append(Spacer(1, 8))

    story.append(Paragraph("""
    <b>Hyperparameter Tuning Impact & Analysis:</b><br/>
    • <b>5-Fold Cross-Validation Accuracy:</b> Improved from <b>80.16% (Default)</b> to <b>82.64% (Tuned)</b>.<br/>
    • <b>Test Set Generalization:</b> Test accuracy increased from <b>88.52% to 90.16%</b>, and <b>F1-Score reached 0.8966</b>.<br/>
    • <b>ROC-AUC Improvement:</b> Area under the ROC curve reached <b>0.9643</b>, outperforming every individual baseline model.<br/>
    • <b>Significance of Shallow Trees (max_depth=3):</b> Constraining tree depth regularizes the ensemble, encouraging decisions based on robust multi-feature clinical consensus.
    """, callout_style))
    story.append(Spacer(1, 6))

    f9_path = os.path.join(FIGURES_DIR, "09_feature_importance.png")
    story.append(Paragraph("<b>Figure 9: Feature Importance from Tuned Random Forest</b>", h2_style))
    story.append(Image(f9_path, width=460, height=195))
    story.append(Paragraph("""
    <b>Key Feature Importance Findings:</b> The foremost predictors identified by the tuned Random Forest are 
    <code>ca</code> (fluoroscopy vessels), <code>thal</code> (thalassemia defect), <code>oldpeak</code> (ST depression), 
    <code>cp</code> (chest pain type), and <code>thalach</code> (max heart rate).
    """, callout_style))
    story.append(PageBreak())

    # PAGE 8: MODEL EVALUATION & COMPARISON TABLE
    story.append(Paragraph("9. COMPREHENSIVE MODEL EVALUATION & COMPARISON", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=8))
    story.append(Paragraph("""
    All models were evaluated on the held-out test set (61 patients: 33 healthy, 28 disease) by executing <code>src/evaluate_models.py</code>:
    """, body_style))

    metrics_csv = os.path.join(METRICS_DIR, "model_comparison.csv")
    df_metrics = pd.read_csv(metrics_csv)

    comp_headers = [Paragraph("<b>Model Name</b>", table_cell_bold), Paragraph("<b>Accuracy</b>", table_cell_bold), 
                    Paragraph("<b>Precision</b>", table_cell_bold), Paragraph("<b>Recall</b>", table_cell_bold), 
                    Paragraph("<b>F1-Score</b>", table_cell_bold), Paragraph("<b>ROC-AUC</b>", table_cell_bold)]
    comp_rows = [comp_headers]
    for _, r in df_metrics.iterrows():
        comp_rows.append([
            Paragraph(f"<b>{r['Model']}</b>", table_cell),
            Paragraph(f"{r['Accuracy']*100:.2f}%", table_cell),
            Paragraph(f"{r['Precision']:.4f}", table_cell),
            Paragraph(f"{r['Recall']:.4f}", table_cell),
            Paragraph(f"<b>{r['F1-Score']:.4f}</b>", table_cell),
            Paragraph(f"{r['ROC-AUC']:.4f}", table_cell)
        ])
    t_comp = Table(comp_rows, colWidths=[140, 70, 70, 70, 75, 75])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EDF2F7")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#E6FFFA")),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 8))

    f7_path = os.path.join(FIGURES_DIR, "07_roc_curves.png")
    f8_path = os.path.join(FIGURES_DIR, "08_model_comparison_bar.png")
    t_plots = Table([[Image(f7_path, width=250, height=165), Image(f8_path, width=250, height=165)]], colWidths=[255, 255])
    t_plots.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(t_plots)
    story.append(Spacer(1, 4))

    story.append(Paragraph("""
    <b>Critical Evaluation Insights:</b><br/>
    • <b>Champion Model:</b> The <b>Tuned Random Forest</b> achieved the highest overall Accuracy (<b>90.16%</b>), F1-Score (<b>0.8966</b>), 
    and ROC-AUC (<b>0.9643</b>). It correctly classified 55 out of 61 test patients.<br/>
    • <b>Clinical Primacy of Recall:</b> In medical diagnostics, a <b>False Negative</b> (classifying a diseased patient as healthy) can be fatal. 
    k-NN achieved 100% recall (identifying all 28 diseased patients), while Tuned Random Forest achieved <b>92.86% recall</b> (identifying 26 of 28 patients) 
    with much higher precision (86.67% vs 80.00% for k-NN).<br/>
    • <b>Parametric Baseline:</b> Logistic Regression demonstrated remarkable efficacy (86.89% Accuracy, 0.9513 ROC-AUC), proving that linear log-odds 
    capture the dominant medical risk trajectories when features are properly standardized.
    """, callout_style))
    story.append(PageBreak())    # PAGE 9: CONFUSION MATRICES & CLASSIFICATION REPORTS
    story.append(Paragraph("10. CONFUSION MATRICES & CLASSIFICATION REPORTS", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=8))

    f6_path = os.path.join(FIGURES_DIR, "06_confusion_matrices.png")
    story.append(Image(f6_path, width=490, height=240))
    story.append(Spacer(1, 6))

    story.append(Paragraph("""
    <b>Detailed Error Analysis of the 61 Test Samples (33 True Healthy, 28 True Disease):</b><br/>
    • <b>Tuned Random Forest:</b> 29 TN, 4 FP, 2 FN, 26 TP. Accuracy = 90.16% (Recall = 92.86%).<br/>
    • <b>Default Random Forest:</b> 27 TN, 6 FP, 1 FN, 27 TP. Accuracy = 88.52% (Recall = 96.43%).<br/>
    • <b>Logistic Regression:</b> 27 TN, 6 FP, 2 FN, 26 TP. Accuracy = 86.89% (Recall = 92.86%).<br/>
    • <b>Support Vector Machine:</b> 27 TN, 6 FP, 3 FN, 25 TP. Accuracy = 85.25% (Recall = 89.29%).<br/>
    • <b>k-Nearest Neighbors:</b> 26 TN, 7 FP, 0 FN, 28 TP. Accuracy = 88.52% (Recall = 100.0%).<br/>
    • <b>Decision Tree:</b> 23 TN, 10 FP, 4 FN, 24 TP. Accuracy = 77.05% (Recall = 85.71%).
    """, callout_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("11. CLASSIFICATION REPORTS (PER-CLASS BREAKDOWN)", h2_style))
    class_report_table = [
        [Paragraph("<b>Model</b>", table_cell_bold), Paragraph("<b>Class</b>", table_cell_bold), Paragraph("<b>Precision</b>", table_cell_bold), Paragraph("<b>Recall</b>", table_cell_bold), Paragraph("<b>F1-Score</b>", table_cell_bold), Paragraph("<b>Support</b>", table_cell_bold)],
        [Paragraph("<b>Tuned Random Forest</b>", table_cell_bold), Paragraph("Healthy (0)<br/>Disease (1)", table_cell), Paragraph("0.9355<br/>0.8667", table_cell), Paragraph("0.8788<br/>0.9286", table_cell), Paragraph("0.9062<br/>0.8966", table_cell), Paragraph("33<br/>28", table_cell)],
        [Paragraph("<b>Logistic Regression</b>", table_cell_bold), Paragraph("Healthy (0)<br/>Disease (1)", table_cell), Paragraph("0.9310<br/>0.8125", table_cell), Paragraph("0.8182<br/>0.9286", table_cell), Paragraph("0.8710<br/>0.8667", table_cell), Paragraph("33<br/>28", table_cell)],
        [Paragraph("<b>Support Vector Machine</b>", table_cell_bold), Paragraph("Healthy (0)<br/>Disease (1)", table_cell), Paragraph("0.9000<br/>0.8065", table_cell), Paragraph("0.8182<br/>0.8929", table_cell), Paragraph("0.8571<br/>0.8475", table_cell), Paragraph("33<br/>28", table_cell)],
        [Paragraph("<b>k-Nearest Neighbors</b>", table_cell_bold), Paragraph("Healthy (0)<br/>Disease (1)", table_cell), Paragraph("1.0000<br/>0.8000", table_cell), Paragraph("0.7879<br/>1.0000", table_cell), Paragraph("0.8814<br/>0.8889", table_cell), Paragraph("33<br/>28", table_cell)],
        [Paragraph("<b>Decision Tree</b>", table_cell_bold), Paragraph("Healthy (0)<br/>Disease (1)", table_cell), Paragraph("0.8519<br/>0.7059", table_cell), Paragraph("0.6970<br/>0.8571", table_cell), Paragraph("0.7667<br/>0.7742", table_cell), Paragraph("33<br/>28", table_cell)]
    ]
    t_cr = Table(class_report_table, colWidths=[130, 80, 70, 70, 75, 75])
    t_cr.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EDF2F7")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t_cr)
    story.append(PageBreak())

    # PAGE 10: STREAMLIT APP DEPLOYMENT & ARCHITECTURE
    story.append(Paragraph("12. STREAMLIT APPLICATION ARCHITECTURE & DEPLOYMENT", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=8))
    story.append(Paragraph("""
    In accordance with Part 3 of the FA2 rubric, an interactive web application (<code>app.py</code>) 
    was built using <b>Streamlit</b>. It loads pre-trained serialized models via <code>joblib</code> 
    and caches them using <code>@st.cache_resource</code> to provide instant inference.
    """, body_style))

    f10_path = os.path.join(FIGURES_DIR, "10_streamlit_app_overview.png")
    story.append(Image(f10_path, width=490, height=220))
    story.append(Spacer(1, 6))

    story.append(Paragraph("""
    <b>Core Features of the CardioPredict Streamlit Application:</b><br/>
    • <b>Model Switching Engine:</b> Clinicians can toggle between all 6 trained algorithms via a sidebar dropdown.<br/>
    • <b>Clinical Form Controls:</b> Organized into 3 logical columns (Demographic/Baseline, Symptoms/Stress Tests, ECG/Fluoroscopy) 
    using <code>st.slider</code>, <code>st.number_input</code>, <code>st.selectbox</code>, and <code>st.radio</code>.<br/>
    • <b>Identical Preprocessing Pipeline:</b> User inputs are passed through the saved <code>preprocessor.pkl</code> fitted on training data, 
    guaranteeing that inference operates on the exact scaled feature representation used during training.<br/>
    • <b>Risk Stratification & Confidence Gauge:</b> Displays high-contrast High-Risk / Low-Risk diagnostic banners, 
    quantitative disease risk probability percentage, a visual progress bar, and delta indicator relative to baseline.<br/>
    • <b>Multi-Model Consensus Engine:</b> An expandable section executes inference across all 6 models simultaneously.<br/>
    • <b>Multi-Tab Analytics:</b> Dedicated tabs for live Model Benchmarking, Exploratory Data Analysis, and Academic Rubric compliance.
    """, callout_style))
    story.append(PageBreak())

    # PAGE 11: VIVA PREDICTION SCENARIOS & RUN INSTRUCTIONS
    story.append(Paragraph("13. SAMPLE CLINICAL PREDICTIONS & VIVA SCENARIOS", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=8))
    story.append(Paragraph("""
    Two contrasting patient profiles were tested to verify real-time inference:
    """, body_style))

    case_scenarios = [
        [Paragraph("<b>Clinical Vitals / Parameter</b>", table_cell_bold), Paragraph("<b>Case 1: High-Risk Cardiac Patient</b>", table_cell_bold), Paragraph("<b>Case 2: Healthy Baseline Individual</b>", table_cell_bold)],
        [Paragraph("Patient Age / Sex", table_cell), Paragraph("67 years / Male (1)", table_cell), Paragraph("38 years / Female (0)", table_cell)],
        [Paragraph("Chest Pain Presentation (<code>cp</code>)", table_cell), Paragraph("Asymptomatic (Type 4)", table_cell), Paragraph("Non-Anginal (Type 3)", table_cell)],
        [Paragraph("Resting Blood Pressure (<code>trestbps</code>)", table_cell), Paragraph("160 mm Hg (Hypertensive)", table_cell), Paragraph("115 mm Hg (Normal)", table_cell)],
        [Paragraph("Serum Cholesterol (<code>chol</code>)", table_cell), Paragraph("286 mg/dl (Elevated)", table_cell), Paragraph("180 mg/dl (Normal)", table_cell)],
        [Paragraph("Fasting Blood Sugar (<code>fbs</code>)", table_cell), Paragraph("True (>120 mg/dl)", table_cell), Paragraph("False (<120 mg/dl)", table_cell)],
        [Paragraph("Resting ECG (<code>restecg</code>)", table_cell), Paragraph("LV Hypertrophy (2)", table_cell), Paragraph("Normal (0)", table_cell)],
        [Paragraph("Max Heart Rate (<code>thalach</code>)", table_cell), Paragraph("108 bpm (Impaired reserve)", table_cell), Paragraph("178 bpm (Robust reserve)", table_cell)],
        [Paragraph("Exercise Angina (<code>exang</code>)", table_cell), Paragraph("Yes (1)", table_cell), Paragraph("No (0)", table_cell)],
        [Paragraph("ST Depression (<code>oldpeak</code>)", table_cell), Paragraph("2.6 mm (Severe depression)", table_cell), Paragraph("0.0 mm (Normal)", table_cell)],
        [Paragraph("ST Slope / Vessels / Thal", table_cell), Paragraph("Flat (2) / 3 Vessels / Reversible (7)", table_cell), Paragraph("Upsloping (1) / 0 Vessels / Normal (3)", table_cell)],
        [Paragraph("<b>Tuned RF Predicted Diagnosis</b>", table_cell_bold), Paragraph("<b>HEART DISEASE DETECTED (Class 1)</b>", table_cell_bold), Paragraph("<b>LOW RISK / HEALTHY (Class 0)</b>", table_cell_bold)],
        [Paragraph("<b>Calculated Disease Probability</b>", table_cell_bold), Paragraph("<b>88.7% Disease Probability</b>", table_cell_bold), Paragraph("<b>7.4% Disease Probability</b>", table_cell_bold)],
        [Paragraph("<b>Consensus Across 6 Models</b>", table_cell_bold), Paragraph("<b>6 / 6 Models Predict Disease (100%)</b>", table_cell_bold), Paragraph("<b>6 / 6 Models Predict Healthy (100%)</b>", table_cell_bold)]
    ]
    t_cases = Table(case_scenarios, colWidths=[150, 175, 175])
    t_cases.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EDF2F7")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('BACKGROUND', (0,-3), (-1,-1), colors.HexColor("#F7FAFC")),
        ('PADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_cases)
    story.append(Spacer(1, 12))

    story.append(Paragraph("14. INSTRUCTIONS TO ACCESS & EXECUTE STREAMLIT APPLICATION", h2_style))
    run_instructions = """
    <b>Option A: Cloud Deployment (Streamlit Community Cloud & GitHub)</b><br/>
    • <b>GitHub Repository:</b> <code>https://github.com/AmitChame/Data-Science</code> (Directory: <code>FA2/</code>)<br/>
    • <b>Streamlit Cloud App Path:</b> <code>FA2/app.py</code><br/>
    • <b>Live Deployment URL:</b> <b>https://amitchame-data-science-fa2.streamlit.app</b><br/>
    <br/>
    <b>Option B: Local Execution (Examination / Viva Evaluation)</b><br/>
    <code>1. Navigate to project root:</code> <b>cd C:\\MCA\\DataScience\\FA2</b><br/>
    <code>2. Install requirements:</code> <b>pip install -r requirements.txt</b><br/>
    <code>3. Launch Streamlit server:</code> <b>streamlit run app.py</b><br/>
    <code>4. Access browser interface at:</code> <b>http://localhost:8501</b>
    """
    story.append(Paragraph(run_instructions, callout_style))
    story.append(PageBreak())

    # PAGE 12: INSIGHTS, LIMITATIONS, CONCLUSION & REFERENCES
    story.append(Paragraph("15. KEY FINDINGS & CLINICAL INSIGHTS", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=8))
    story.append(Paragraph("""
    1. <b>Multi-factorial Superiority:</b> Cardiac risk cannot be captured reliably by single isolated biomarkers. 
    The integration of stress-induced ST depression, maximum heart rate achieved, and fluoroscopy imaging in an ensemble model elevates diagnostic accuracy to <b>90.16%</b>.
    <br/><br/>
    2. <b>Addressing 'Silent' Coronary Disease:</b> A pivotal insight from our exploratory data analysis is that <b>asymptomatic patients (Chest Pain Type 4)</b> 
    exhibit high rates of significant coronary artery disease. Machine learning screening captures subtle non-symptomatic patterns that traditional subjective assessments overlook.
    <br/><br/>
    3. <b>Ensemble Regularization via GridSearchCV:</b> Unconstrained decision trees prone to high variance achieved only 77.05% accuracy. 
    By tuning maximum depth (<code>max_depth=3</code>), minimum split sizes, and leaf limits across 1,440 fits, Random Forest generalized with an F1-score of 0.8966.
    """, body_style))

    story.append(Paragraph("16. STUDY LIMITATIONS", h2_style))
    story.append(Paragraph("""
    • <b>Sample Size:</b> The Cleveland cohort comprises 303 patients. While optimal for benchmarking classical algorithms, larger multi-center cohorts are desirable for deep neural architectures.<br/>
    • <b>Demographic Skew:</b> The historical dataset comprises 68% male participants. Models deployed clinically require re-calibration across diverse demographic distributions to prevent gender-based performance disparities.<br/>
    • <b>Deployment Environment:</b> The current Streamlit application operates in a local development runtime; production healthcare deployment mandates HIPAA/GDPR-compliant encrypted cloud pipelines.
    """, body_style))

    story.append(Paragraph("17. CONCLUSION & FUTURE SCOPE", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=8))
    story.append(Paragraph("""
    This FA2 Case Study successfully executed an end-to-end Machine Learning and Deployment lifecycle. 
    Five algorithms were rigorously benchmarked, and the <b>Tuned Random Forest</b> proved best with <b>90.16% Accuracy</b>, 
    <b>92.86% Recall</b>, and <b>0.9643 ROC-AUC</b>. The model and preprocessing pipeline were deployed into an intuitive Streamlit interface 
    capable of real-time clinical risk stratification.
    <br/><br/>
    <b>Future Scope:</b> Future extensions could incorporate SHAP (SHapley Additive exPlanations) values for patient-specific feature attribution, 
    integrate real-time wearable sensor feeds (photoplethysmography heart rate streams), and containerize the application with Docker for cloud deployment.
    """, body_style))

    story.append(Paragraph("18. ACADEMIC REFERENCES", h2_style))
    refs = [
        "1. Detrano, R., et al. (1989). <i>International application of a new probability algorithm for the diagnosis of coronary artery disease</i>. American Journal of Cardiology, 64(5), 304-310.",
        "2. UC Irvine Machine Learning Repository. (1988). <i>Heart Disease Dataset</i>. DOI: 10.24432/C52P4X.",
        "3. Pedregosa, F., et al. (2011). <i>Scikit-learn: Machine Learning in Python</i>. Journal of Machine Learning Research, 12, 2825-2830.",
        "4. Breiman, L. (2001). <i>Random Forests</i>. Machine Learning, 45(1), 5-32.",
        "5. Streamlit Inc. (2026). <i>Streamlit Documentation: Deploying Interactive Machine Learning Web Applications</i>."
    ]
    for r in refs:
        story.append(Paragraph(r, bullet_style))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Report generated successfully at: {REPORT_PATH}")


if __name__ == "__main__":
    build_pdf()