# Task 1: Natural Gas Price Forecasting

##  Objective

Build a predictive model for natural gas prices that incorporates seasonal patterns and can predict prices for any given date.

---

##  Dataset

**File:** `Nat_Gas.csv`
- **Period:** October 2020 - September 2024
- **Frequency:** Monthly
- **Records:** 48 data points
- **Range:** $10.00 - $13.54 per MMBtu

---

##  Analysis & Findings

### Seasonal Pattern Discovery
- **Peak Season:** January (average $11.78/MMBtu)
- **Trough Season:** June (average $10.70/MMBtu)
- **Seasonal Variation:** ~10% price swing
- **Pattern:** Winter demand drives higher prices, summer sees lower prices

### Data Characteristics
- Clear upward trend over the 4-year period
- Strong seasonal cyclicality (12-month period)
- Relatively low noise/volatility

---

##  Methodology

### 1. Feature Engineering
Created seasonal features using Fourier transform:
```python
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
```

**Why sin/cos?**
- Captures cyclical nature of seasons
- Ensures December is close to January (continuity)
- Provides smooth seasonal transitions

### 2. Model Selection
**Linear Regression** with features:
- `days_since_start` (trend)
- `month_sin` (seasonal component 1)
- `month_cos` (seasonal component 2)

### 3. Dual Prediction Mode
- **Historical dates (before Sep 2024):** Cubic spline interpolation for smooth values
- **Future dates (after Sep 2024):** Regression model for extrapolation

---

##  Results

### Model Performance
- **R² Score:** 0.9290 (93% variance explained) 
- **RMSE:** Low residual error
- **Visual fit:** Excellent alignment with observed data

### Key Metrics
```
Mean Absolute Error: ~$0.30/MMBtu
Max Seasonal Range: $10.70 - $11.78
Prediction Confidence: High for 1-2 years ahead
```

---

##  Code Structure

```python
def predict_price(date_input):
    """
    Predicts natural gas price for any given date.
    
    Parameters:
    -----------
    date_input : str or datetime
        Target date for price prediction
    
    Returns:
    --------
    float : Predicted price in $/MMBtu
    
    Examples:
    ---------
    >>> predict_price('2025-01-15')
    13.54
    
    >>> predict_price('2024-06-01')
    12.11
    """
```

---

##  Visualizations

### 1. Time Series Plot
Complete historical data with trend line

### 2. Seasonal Decomposition
Monthly price patterns showing clear winter peaks

### 3. Final Predictions
Model predictions vs actual prices with future extrapolation

---

##  Key Insights

1. **Seasonality is strong:** Winter months consistently 10% higher than summer
2. **Upward trend:** Overall prices increasing over time
3. **Predictability:** High R² indicates reliable forecasting
4. **Feature importance:** Seasonal features crucial for accuracy

---

##  Business Applications

### Use Cases
1. **Contract Pricing:** Inform storage contract valuations (Task 2)
2. **Hedging Strategy:** Plan purchases during low-price periods
3. **Revenue Forecasting:** Predict future cash flows
4. **Risk Management:** Estimate price volatility

### Recommendations
- Buy gas during summer months (June-August)
- Sell during winter months (December-February)
- Account for ~10% seasonal spread in contracts
- Monitor long-term trend for strategic planning

---

##  Files

```
task1-gas-price-forecasting/
├── README.md (this file)
├── gas_price_complete.py
├── data/
│   └── Nat_Gas.csv
└── outputs/
    ├── 1_serie_temporelle.png
    ├── 2_saisonnalite.png
    └── 3_predictions_finales.png
```

---

##  Technologies Used

- **Python 3.9+**
- **pandas:** Data manipulation
- **numpy:** Numerical computing
- **scikit-learn:** Linear regression
- **scipy:** Cubic spline interpolation
- **matplotlib:** Visualization

---

##  Notes

- Model assumes continuation of historical trends
- Extreme events (geopolitical, weather) not modeled
- Confidence decreases for predictions >2 years out
- Should be recalibrated annually with new data

---

##  Related Tasks

This price prediction function is used in:
- **Task 2:** Storage Contract Pricing (provides price inputs)

---

*Completed as part of JP Morgan Quantitative Research Virtual Experience*
