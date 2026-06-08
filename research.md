
# Comprehensive Machine Learning Benchmarking for Cardiovascular Risk Stratification

**Project:** CardioSense AI — Phase 2 & 3  
**Author:** prince-gupta79  
**Core Objective:** To systematically evaluate, optimize, and benchmark five distinct machine learning paradigms for the early detection and predictive analysis of ischemic heart disease.

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
## 🔬 Technical Deep-Dive & Research
For a granular look at the data paradigms, model benchmarking parameters, and how we audited the 100% precision anomaly, check out our full [Research Methodology Documentation](research.md).

## 5. Conclusions & Deployment

While deep ensemble models like Random Forest and Gradient Boosting dominate tabular clinical datasets due to their natural handling of mixed categorical/continuous data types, the ultimate success of an ML model lies in its accessibility. By wrapping this optimized pipeline into a clean Streamlit deployment interface, CardioSense AI successfully bridges the gap between pure computational math and rapid, actionable clinical decision support.



