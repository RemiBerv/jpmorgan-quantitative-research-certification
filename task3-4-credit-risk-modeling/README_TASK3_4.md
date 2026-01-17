# Tasks 3 & 4 : Credit Default Prediction & Expected Loss Calculation

##  Objectives

**Task 3:** Build a machine learning model to predict the probability of loan default (PD)  
**Task 4:** Calculate expected loss on loans using the predicted PD

---

##  Business Context

Banks face significant risk from loan defaults. Accurate prediction of default probability allows:
- **Capital Planning:** Set aside sufficient reserves
- **Risk Pricing:** Adjust interest rates based on risk
- **Lending Decisions:** Approve/reject loan applications
- **Portfolio Management:** Monitor overall risk exposure

---

##  Dataset

**File:** `Task_3_and_4_Loan_Data.csv`
- **Records:** 10,000 borrowers
- **Features:** 7 predictive variables
- **Target:** Binary default indicator (0 = no default, 1 = default)

### Feature Description

| Feature | Description | Type |
|---------|-------------|------|
| `credit_lines_outstanding` | Number of active credit lines | Integer |
| `loan_amt_outstanding` | Current loan amount | Float ($) |
| `total_debt_outstanding` | Total debt across all sources | Float ($) |
| `income` | Annual income | Float ($) |
| `years_employed` | Years in current employment | Integer |
| `fico_score` | Credit score (300-850) | Integer |
| `default` | Default indicator (TARGET) | Binary |

### Class Distribution
- **No Default (0):** 8,149 (81.5%)
- **Default (1):** 1,851 (18.5%)

 **Imbalanced dataset** - requires special handling

---

## 🔍 Exploratory Analysis

### Default Rate by FICO Range

| FICO Range | Count | Default Rate |
|------------|-------|--------------|
| 400-500 | 155 | **72.3%**  |
| 500-550 | 603 | **49.6%**  |
| 550-600 | 1,924 | **29.2%** |
| 600-650 | 3,125 | **17.7%** |
| 650-700 | 2,709 | **9.5%** |
| 700-750 | 1,211 | **5.0%** |
| 750-850 | 273 | **2.9%**  |

**Key Finding:** Strong negative correlation between FICO and default risk

---

##  Machine Learning Models

### Models Tested

1. **Logistic Regression**
   - Simple, interpretable
   - Shows feature coefficients
   - Fast training

2. **Random Forest**
   - Ensemble method (100 trees)
   - Captures non-linear relationships
   - Robust to overfitting

### Data Preprocessing

```python
# 1. Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# 2. Feature Normalization (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

##  Results

### Model Performance Comparison

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| **Accuracy** | 99.90% | 98.85% |
| **Precision** | 100.00% | 94.60% |
| **Recall** | 99.46% | 99.46% |
| **F1-Score** | 0.9973 | 0.9697 |
| **AUC-ROC** | **1.0000**  | 0.9997 |

 **Winner:** Logistic Regression (slightly better AUC)

### Confusion Matrix (Logistic Regression)

```
                Predicted
              No Default  Default
Actual  
No Default      1626        4
Default            2      368
```

**Interpretation:**
- True Negatives: 1,626 (correctly identified non-defaults)
- False Positives: 4 (false alarms)
- False Negatives: 2 (missed defaults) 
- True Positives: 368 (correctly caught defaults)

**Detection Rate:** 368/370 = 99.5% of defaults caught!

---

##  Feature Importance

### Logistic Regression Coefficients

| Feature | Coefficient | Impact |
|---------|------------|--------|
| `credit_lines_outstanding` | **+8.75** |  Highest risk factor |
| `total_debt_outstanding` | +3.84 | High risk |
| `years_employed` | -2.90 | Protective  |
| `income` | -2.35 | Protective  |
| `fico_score` | -1.14 | Protective  |
| `loan_amt_outstanding` | +0.12 | Minimal impact |

**Key Insights:**
- Having **many credit lines** is the #1 default predictor
- **Employment stability** and **income** are protective
- **FICO score** matters but less than credit line count

---

##  Expected Loss Calculation (Task 4)

### Formula

```
Expected Loss (EL) = PD × EAD × LGD

Where:
  PD  = Probability of Default (from ML model)
  EAD = Exposure at Default (loan amount)
  LGD = Loss Given Default (1 - recovery rate)
```

### Given Parameters
- **Recovery Rate:** 10% (we recover 10% of defaulted loans)
- **LGD:** 90% (we lose 90%)

### Example Calculations

**Low-Risk Client:**
```
FICO: 750, Income: $100K, Debt: $2K
→ PD = 0.00% (model prediction)
→ Loan Amount = $5,000
→ EL = 0.00 × $5,000 × 0.90 = $0
```

**High-Risk Client:**
```
FICO: 500, Income: $25K, Debt: $20K
→ PD = 100.00% (model prediction)
→ Loan Amount = $5,000
→ EL = 1.00 × $5,000 × 0.90 = $4,500
```

**Medium-Risk Client:**
```
FICO: 650, Income: $60K, Debt: $8K
→ PD = 0.01% (model prediction)
→ Loan Amount = $5,000
→ EL = 0.0001 × $5,000 × 0.90 = $0.65
```

---

##  Code Structure

### Prediction Functions

```python
def predict_default_probability(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score
):
    """Returns probability of default (0-1)"""
    
def calculate_expected_loss(
    ... same parameters ...,
    recovery_rate=0.10
):
    """Returns expected loss in dollars"""
```

---

##  Business Applications

### 1. Loan Approval Decisions

```python
PD = predict_default_probability(...)

if PD < 0.05:
    decision = "APPROVE - Low Risk"
elif PD < 0.20:
    decision = "APPROVE - Higher Rate"
else:
    decision = "DECLINE - Too Risky"
```

### 2. Risk-Based Pricing

```python
base_rate = 0.05  # 5%
risk_premium = PD * 10  # 10% premium per PD point

final_rate = base_rate + risk_premium
```

### 3. Portfolio Risk Assessment

```python
total_expected_loss = sum(
    calculate_expected_loss(...) 
    for each loan in portfolio
)

required_capital = total_expected_loss * 1.2  # 20% buffer
```

---

##  Model Validation

### Why AUC = 1.0?

The near-perfect AUC indicates:
-  Features are highly predictive
-  Clear separation between classes
-  Model captures the underlying pattern

In production:
- Real-world AUC typically 0.70-0.85
- Our 1.0 reflects clean, well-structured data
- Still demonstrates proper ML methodology

---

##  Files

```
task3-4-credit-risk-modeling/
├── README.md (this file)
├── task3_4_credit_risk.py
└── data/
    └── Task_3_and_4_Loan_Data.csv
```

---

##  Technologies Used

- **Python 3.9+**
- **pandas:** Data manipulation
- **scikit-learn:** ML models (LogisticRegression, RandomForestClassifier)
- **numpy:** Numerical operations
- **matplotlib/seaborn:** Visualization

---

##  Key Learnings

### Technical
- Handling imbalanced datasets (stratification, class weights)
- Model comparison methodology
- Feature importance interpretation
- Proper train-test splitting

### Business
- Credit risk = function of multiple factors
- Employment and income are protective
- Too many credit lines = red flag
- Default prediction enables proactive risk management

---

##  Future Enhancements

1. **Calibration:** Ensure predicted probabilities match actual rates
2. **More Features:** Payment history, debt-to-income ratio
3. **Temporal Validation:** Test on data from different time periods
4. **Ensemble Methods:** Combine multiple models
5. **Explainability:** SHAP values for individual predictions

---

*Completed as part of JP Morgan Quantitative Research Virtual Experience*
