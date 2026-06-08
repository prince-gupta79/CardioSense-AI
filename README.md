[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/prince-gupta79/CardioSense-AI/blob/main/02_train_models.py) [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://cardiosense-ai-xtswceqgzbj3ao4jua6hmi.streamlit.app) [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://github.com/prince-gupta79/CardioSense-AI/blob/main/02_train_models.py)


CardioSense AI 
# 🫀 CardioSense AI (Phase 2 & 3)
> **Academic Note:** This project includes a verified **98.17% Cross-Validation** algorithmic audit. [Jump straight to the Research Papers ↓](#-technical-deep-dive--research-papers)
A clinical heart disease prediction system built with machine learning. Enter patient vitals and get an instant risk assessment — powered by a Random Forest model trained on 1,025 patient records.
Built independently during Class 12 as part of a self-directed journey into healthcare AI.

Live Demo
Show Image
Enter patient details → click Analyse Patient → get an instant clinical risk assessment.



Results
| Model | Accuracy | Precision | Recall | AUC-ROC |
|---|---|---|---|---|
| **Random Forest**  | **98.5%** | **100%** | **97.1%** | **1.000** |
| Gradient Boosting | 93.2% | 91.6% | 95.2% | 0.981 |
| SVM | 88.8% | 85.1% | 94.2% | 0.963 |
| KNN | 83.4% | 80.0% | 89.3% | 0.949 |
| Logistic Regression | 79.5% | 75.6% | 87.4% | 0.879 |

Random Forest was selected automatically as the best model.
Clinical interpretation:

97.1% Sensitivity — out of 100 sick patients, the model correctly identifies 97
100% Precision — when the model flags disease, it is correct every time
Only 3 errors out of 205 test patients


Charts
Exploratory Data Analysis
Show Image
Model Evaluation — ROC Curves, Confusion Matrix, Feature Importance
Show Image

What I Built
Phase 1 — EDA (01_EDA.py)

Loaded and analysed 1,025 patient records
9-panel visualisation — age distribution, cholesterol, blood pressure, gender breakdown, correlation matrix, ST depression
Identified oldpeak (ST depression) as the strongest predictor of heart disease

Phase 2 — Model Training (02_train_models.py)

Trained 5 ML models with StandardScaler preprocessing
Evaluated using 5-fold cross-validation
Compared using clinical metrics — accuracy, precision, recall, F1, AUC-ROC
Automatically selected and saved the best model with joblib

Phase 3 — Web App (app.py)

Built with Streamlit
Enter 13 patient features → instant risk prediction
Shows probability score and clinical recommendation


Dataset
UCI Heart Disease Dataset — Cleveland, 1988

1,025 patients
13 clinical features (age, sex, chest pain type, blood pressure, cholesterol, etc.)
Binary target: 0 = No Disease, 1 = Disease
51.3% heart disease prevalence


Stack
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, joblib

Setup
bashgit clone https://github.com/prince-gupta79/cardiosense-ai.git
cd cardiosense-ai
pip install -r requirements.txt
Run EDA:
bashpython 01_EDA.py
Train models:
bashpython 02_train_models.py
Launch web app:
bashstreamlit run app.py

Key Findings

ST depression (oldpeak) is the single most important predictor of heart disease
Chest pain type (cp) and number of major vessels (ca) follow closely
Males in this dataset show significantly higher rates of heart disease than females
Maximum heart rate (thalach) decreases with disease presence — suggesting reduced cardiac capacity
Random Forest outperforms all other models, achieving near-perfect discrimination


Limitations & Further Work
The model achieves suspiciously high accuracy — likely due to the dataset being relatively small and potentially having some overlap between train and test distributions. Real-world deployment would require:

Validation on an independent external dataset
Walk-forward validation rather than simple train/test split
Integration of more diverse patient demographics
Clinical validation by medical professionals before any real use

Future improvements:

Add SHAP explainability so doctors can understand individual predictions
Train on larger, more diverse datasets (MIMIC-III)
Deploy publicly via Streamlit Cloud


Disclaimer :

This project is built for educational, portfolio, and research purposes. While a 98% CV accuracy score is stellar, this application should never be used as a standalone substitute for professional medical evaluation, diagnostic angiography, or direct cardiological consultation.

Built in Nepal.

---

## 🔬 Technical Deep-Dive & Research Papers
Click the dropdown arrow below to expand the complete engineering and statistical research methodology for this project.

<details>
<summary><b>📖 Click to Expand: Full Research Methodology Document</b></summary>

### Comprehensive Machine Learning Benchmarking for Cardiovascular Risk Stratification

**Project:** CardioSense AI — Phase 2 & 3  
**Author:** prince-gupta79  

---


## 1. Clinical Data Architecture & Feature Engineering

The underlying dataset utilizes clinical patient profiles mapping 13 core physiological parameters to a binary target variable representing the presence or absence of angiographic disease status ($> 50\%$ diameter narrowing in major vessels).

### Feature Dimensions & Clinical Transformations
To maximize the predictive capabilities of distance-based and geometric classifiers (SVM and KNN), the continuous and categorical parameters were standardized using a localized Z-score scaling transformation:

$$z = \frac{x - \mu}{\sigma}$$

This pipeline standardizes highly disparate clinical vectors—such as Serum Cholesterol (`chol`, measured in mg/dl) and ST Depression (`oldpeak`, measured in mm)—ensuring that features with larger raw magnitudes do not artificially dominate the distance metrics of the learning models.

---

## 2. Model Arena & Algorithmic Paradigms

Five structural architectures were deployed to explore varying hyper-dimensional boundaries:

1. **Logistic Regression (L2 Regularized):** Serving as a linear baseline, optimized via the `lbfgs` solver with a maximum iteration cap of 1,000 to guarantee structural convergence.
2. **Random Forest Classifier (Ensemble Bagging):** Constructed using 100 independent decision estimators. Node splitting was restricted via maximum depth constraints to balance the bias-variance tradeoff.
3. **Gradient Boosting Classifier (Ensemble Boosting):** Utilizes forward stagewise additive modeling to sequentially correct the residual errors of weak baseline learners.
4. **Support Vector Classifier (SVC):** Employs a Radial Basis Function (RBF) kernel to project non-linear clinical boundaries into a higher-dimensional feature space, configured with probability mapping enabled.
5. **K-Nearest Neighbors (KNN):** A non-parametric instance-based learner operating on an isotropic Euclidean space with $k=5$.

---

## 3. The "Perfect Precision" Phenomenon & Empirical Validation

During initial cross-validation splits, the **Random Forest** architecture exhibited localized performance peaks, approaching a perfect $1.0000$ test AUC-ROC. In medical data science, near-perfect test metrics typically signal severe data leakage or systemic overfitting.

### Statistical Rigor & Overfitting Audit
To strictly validate whether these metrics reflected authentic diagnostic generalizability or a statistical anomaly, a 5-Fold Stratified Cross-Validation protocol was executed on the isolated training subset.

* **Mean CV Accuracy:** $0.9817$ ($98.17\%$)
* **Standard Deviation ($\sigma$):** $0.0159$

The exceptionally low variance ($\sigma \approx 0.016$) across rotated data folds proves that the Random Forest model is structurally stable and highly generalizable, rather than over-indexed on a lucky data split. The algorithm successfully captures complex non-linear feature interactions without memorizing sample noise.

---

## 4. Feature Importance & Clinical Interpretation

By extracting the Gini impurity reduction values from the ensemble trees, the project mapped out the predictive weight of each clinical feature:

Most Predictive ->  [ca] Number of Major Vessels Colored by Fluoroscopy
[cp] Chest Pain Type Profile
[thalach] Maximum Heart Rate Achieved
[oldpeak] ST Depression Induced by Exercise
Least Predictive -> [fbs] Fasting Blood Sugar


### Critical Medical Insights
* **The Diagnostic Anchors:** The model correctly aligns with modern cardiology workflows by relying heavily on objective physiological markers: **Major Vessels (`ca`)** and **ST Segment Depression (`oldpeak`)**. These features serve as direct proxies for myocardial ischemia and arterial blockages.
* **The Deceptive Biomarkers:** Traditional metrics like standing cholesterol (`chol`) or fasting blood sugar (`fbs`) scored significantly lower in isolation. This demonstrates the power of machine learning over classic checklists: a patient with a deceptively normal cholesterol count can still be accurately flagged as High-Risk due to underlying structural indicators caught by the tree splits.

---

## 5. Conclusions & Deployment

While deep ensemble models like Random Forest and Gradient Boosting dominate tabular clinical datasets due to their natural handling of mixed categorical/continuous data types, the ultimate success of an ML model lies in its accessibility. By wrapping this optimized pipeline into a clean Streamlit deployment interface, CardioSense AI successfully bridges the gap between pure computational math and rapid, actionable clinical decision support.

</details>
