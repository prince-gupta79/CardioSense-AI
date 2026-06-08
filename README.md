CardioSense AI ❤️
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
