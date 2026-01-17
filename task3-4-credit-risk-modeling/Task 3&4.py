import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')

# Missing values
print(df.isnull().sum())
print()

# Default distribution
default_counts = df['default'].value_counts()
default_pct = df['default'].value_counts(normalize=True) * 100
print(f"   No default (0) : {default_counts[0]} ({default_pct[0]:.1f}%)")
print(f"   Default (1)    : {default_counts[1]} ({default_pct[1]:.1f}%)")
print()

if default_pct[1] < 10:
    print("   Imbalanced dataset: few defaults. This is common in credit risk.")
    print("   We will rely on adapted metrics (Precision, Recall, AUC-ROC).")
print()

# Descriptive statistics by default status
print(" Comparison: Clients with default vs without default")
print()
print("Statistics for clients WITHOUT default (default=0):")
print(df[df['default'] == 0].describe())
print()
print("Statistics for clients WITH default (default=1):")
print(df[df['default'] == 1].describe())
print()

# Features / target
X = df.drop(['customer_id', 'default'], axis=1)
y = df['default']

print(f" Features (X): {X.shape[1]} columns")
print(f"   Column names: {list(X.columns)}")
print()
print(f" Target (y): {len(y)} observations")
print(f"   Unique values: {y.unique()}")
print()

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(" Train / test split:")
print(f"   Train: {len(X_train)} clients ({len(X_train)/len(df)*100:.1f}%)")
print(f"   Test : {len(X_test)} clients ({len(X_test)/len(df)*100:.1f}%)")
print()

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(" Standardization completed (StandardScaler).")
print("   Each feature now has: mean ≈ 0, std ≈ 1.")
print()

models = {}
predictions = {}

# -------------------------------------------------------------------------
# MODEL 1: LOGISTIC REGRESSION
# -------------------------------------------------------------------------
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

models['Logistic Regression'] = lr
predictions['Logistic Regression'] = (y_pred_lr, y_pred_proba_lr)

print(" Logistic Regression model trained.")
print()

print(" Coefficients (feature importance proxy):")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print(feature_importance)
print()
print(" Interpretation:")
print("   - Positive coefficient → Increases default risk.")
print("   - Negative coefficient → Decreases default risk.")
print("   - Larger absolute value → More important feature.")
print()

# -------------------------------------------------------------------------
# MODEL 2: RANDOM FOREST
# -------------------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=50,
    class_weight='balanced'
)
rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)
y_pred_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

models['Random Forest'] = rf
predictions['Random Forest'] = (y_pred_rf, y_pred_proba_rf)

print(" Random Forest feature importance:")
feature_importance_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance_rf)
print()

# -------------------------------------------------------------------------
# MODEL EVALUATION
# -------------------------------------------------------------------------
results = []

for model_name, (y_pred, y_pred_proba) in predictions.items():
    print(f" {model_name}")
    print("-" * 80)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"   Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall   : {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1-Score : {f1:.4f}")
    print(f"   AUC-ROC  : {auc:.4f}")
    print()
    
    print("   Confusion matrix:")
    print(f"      TN (True Negative) : {tn} | FP (False Positive): {fp}")
    print(f"      FN (False Negative): {fn} | TP (True Positive) : {tp}")
    print()
    print("   Interpretation:")
    print(f"      - True defaults detected     : {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)")
    print(f"      - False alarms (non-defaults flagged as default): {fp}/{fp+tn} ({fp/(fp+tn)*100:.1f}%)")
    print()
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc
    })

# Model comparison table
print("="*80)
print(" MODEL COMPARISON")
print("="*80)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print()

best_model = results_df.loc[results_df['AUC-ROC'].idxmax(), 'Model']
print(f" Best model: {best_model} (highest AUC-ROC).")
print()

# -------------------------------------------------------------------------
# FINAL PREDICTION FUNCTIONS
# -------------------------------------------------------------------------
print("="*80)
print("Final prediction function")
print("="*80)
print()

best_model_obj = models[best_model]

def predict_default_probability(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score
):
    """
    Predict the probability of default for a borrower.
    
    Parameters
    ----------
    credit_lines_outstanding : int
        Number of active credit lines.
    loan_amt_outstanding : float
        Current loan amount ($).
    total_debt_outstanding : float
        Total outstanding debt ($).
    income : float
        Annual income ($).
    years_employed : int
        Years in current employment.
    fico_score : int
        FICO score (300–850).
    
    Returns
    -------
    float
        Probability of default (between 0 and 1).
    """
    features = pd.DataFrame({
        'credit_lines_outstanding': [credit_lines_outstanding],
        'loan_amt_outstanding': [loan_amt_outstanding],
        'total_debt_outstanding': [total_debt_outstanding],
        'income': [income],
        'years_employed': [years_employed],
        'fico_score': [fico_score]
    })
    
    features_scaled = scaler.transform(features)
    probability = best_model_obj.predict_proba(features_scaled)[0, 1]
    
    return probability

def calculate_expected_loss(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score,
    recovery_rate=0.10
):
    """
    Compute expected loss on a loan.
    
    Formula
    -------
    Expected Loss = PD × Loan Amount × LGD
    
    where:
    - PD  = Probability of Default (from the model).
    - Loan Amount = Exposure at default.
    - LGD = Loss Given Default = 1 - Recovery Rate.
    
    Parameters
    ----------
    Same parameters as predict_default_probability()
    recovery_rate : float, default=0.10
        Recovery rate (10% means 10% recovered, 90% lost).
    
    Returns
    -------
    dict
        {
            'probability_of_default': float,
            'loan_amount': float,
            'loss_given_default': float,
            'expected_loss': float,
            'expected_loss_pct': float
        }
    """
    pd_value = predict_default_probability(
        credit_lines_outstanding,
        loan_amt_outstanding,
        total_debt_outstanding,
        income,
        years_employed,
        fico_score
    )
    
    lgd = 1 - recovery_rate
    expected_loss = pd_value * loan_amt_outstanding * lgd
    expected_loss_pct = (expected_loss / loan_amt_outstanding) * 100
    
    return {
        'probability_of_default': pd_value,
        'loan_amount': loan_amt_outstanding,
        'loss_given_default': lgd,
        'expected_loss': expected_loss,
        'expected_loss_pct': expected_loss_pct
    }

# -------------------------------------------------------------------------
# EXAMPLES
# -------------------------------------------------------------------------
print(" TEST 1: LOW-RISK CLIENT")
print("-" * 80)
print("Profile:")
print("   - High income      : $100,000")
print("   - Excellent FICO   : 750")
print("   - Low debt         : $2,000")
print("   - Stable employment: 8 years")
print()

result1 = calculate_expected_loss(
    credit_lines_outstanding=1,
    loan_amt_outstanding=5000,
    total_debt_outstanding=2000,
    income=100000,
    years_employed=8,
    fico_score=750
)

print("Result:")
print(f"   Probability of default : {result1['probability_of_default']:.2%}")
print(f"   Loan amount            : ${result1['loan_amount']:,.2f}")
print(f"   Expected loss          : ${result1['expected_loss']:,.2f}")
print(f"   Expected loss (% loan) : {result1['expected_loss_pct']:.2f}%")
print()

print(" TEST 2: HIGH-RISK CLIENT")
print("-" * 80)
print("Profile:")
print("   - Low income       : $25,000")
print("   - Poor FICO        : 500")
print("   - High debt        : $20,000")
print("   - Unstable job     : 1 year")
print()

result2 = calculate_expected_loss(
    credit_lines_outstanding=5,
    loan_amt_outstanding=5000,
    total_debt_outstanding=20000,
    income=25000,
    years_employed=1,
    fico_score=500
)

print("Result:")
print(f"   Probability of default : {result2['probability_of_default']:.2%}")
print(f"   Loan amount            : ${result2['loan_amount']:,.2f}")
print(f"   Expected loss          : ${result2['expected_loss']:,.2f}")
print(f"   Expected loss (% loan) : {result2['expected_loss_pct']:.2f}%")
print()

print(" TEST 3: AVERAGE-RISK CLIENT")
print("-" * 80)
print("Profile:")
print("   - Medium income    : $60,000")
print("   - Fair FICO        : 650")
print("   - Moderate debt    : $8,000")
print("   - Normal job tenure: 4 years")
print()

result3 = calculate_expected_loss(
    credit_lines_outstanding=2,
    loan_amt_outstanding=5000,
    total_debt_outstanding=8000,
    income=60000,
    years_employed=4,
    fico_score=650
)

print("Result:")
print(f"   Probability of default : {result3['probability_of_default']:.2%}")
print(f"   Loan amount            : ${result3['loan_amount']:,.2f}")
print(f"   Expected loss          : ${result3['expected_loss']:,.2f}")
print(f"   Expected loss (% loan) : {result3['expected_loss_pct']:.2f}%")
print()

# -------------------------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------------------------
print(f"FINAL MODEL : {best_model}")
print(f"   Performance (AUC-ROC): {results_df.loc[results_df['Model']==best_model, 'AUC-ROC'].values[0]:.4f}")
