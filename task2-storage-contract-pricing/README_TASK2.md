# Task 2: Storage Contract Valuation

##  Objective

Build a pricing model for natural gas storage contracts that accounts for operational constraints, variable costs, and market dynamics.

---

##  Business Context

A storage contract allows clients to:
1. **Inject** gas during low-price periods (typically summer)
2. **Store** gas in a facility (with associated costs)
3. **Withdraw** gas during high-price periods (typically winter)
4. **Profit** from the price spread minus costs

---

##  Contract Parameters

### Required Inputs
1. **Injection dates:** When gas can be purchased and injected
2. **Withdrawal dates:** When gas can be extracted and sold
3. **Injection rate:** Maximum volume per day (MMBtu/day)
4. **Withdrawal rate:** Maximum volume per day (MMBtu/day)
5. **Maximum storage volume:** Total capacity (MMBtu)
6. **Storage cost:** Daily cost per MMBtu stored ($/MMBtu/day)

---

##  Methodology

### Day-by-Day Simulation

Rather than simple point calculations, this model simulates **every single day** of the contract:

```
For each day from start to end:
    1. Check if it's an injection date
       → Inject min(injection_rate, remaining_capacity)
    
    2. Check if it's a withdrawal date
       → Withdraw min(withdrawal_rate, current_volume)
    
    3. Calculate storage cost for current volume
       → Cost = current_volume × storage_cost_per_day
    
    4. Track all cash flows
```

### Operational Constraints

**Rate Limits:** Can't inject/withdraw everything instantly
- Realistic flow rates (e.g., 100,000 MMBtu/day)
- Spreads operations over multiple days

**Capacity Limits:** Storage facility has maximum volume
- Prevents over-injection
- Enforces physical constraints

**Variable Costs:** Storage costs depend on actual volume stored
- Daily calculation: `volume_stored × cost_per_day`
- Accumulates over entire contract duration

---

##  Valuation Formula

```
Net Contract Value = Total Revenue - Total Costs

Where:
  Total Revenue = Σ (withdrawal_volume × withdrawal_price)
  Total Costs = Σ (injection_volume × injection_price)
                + Σ (daily_storage_costs)
                + injection_fees
                + withdrawal_fees
```

---

##  Test Results

### Test Case 1: Simple Scenario
```
Parameters:
- Injection: 7 days in June 2025
- Withdrawal: 7 days in December 2025
- Rate: 100K MMBtu/day
- Capacity: 1M MMBtu
- Storage cost: $0.01/MMBtu/day

Results:
- Volume: 700K MMBtu
- Buy price avg: $12.07/MMBtu
- Sell price avg: $13.50/MMBtu
- Storage costs: $1.28M
- Net Value: -$280K 

Insight: Storage costs exceeded price spread profit
```

### Test Case 2: Capacity Constrained
```
Parameters:
- Injection: 30 days in June 2025
- Withdrawal: 15 days in December 2025
- Injection rate: 50K MMBtu/day
- Withdrawal rate: 100K MMBtu/day
- Capacity: 800K MMBtu (LIMITED!)
- Storage cost: $0.015/MMBtu/day

Results:
- Volume: 800K MMBtu (100% capacity used)
- Storage costs: $2.15M
- Net Value: -$1.01M 

Insight: Hit capacity limit, higher costs due to longer duration
```

---

##  Key Insights

### Profitability Drivers

**Positive Factors:**
-  Large price spreads (>$2/MMBtu)
-  Short storage duration
-  Low storage costs
-  Optimal timing (summer buy → winter sell)

**Negative Factors:**
-  Small price spreads (<$1.50/MMBtu)
-  Long storage periods (>6 months)
-  High daily storage costs
-  Large volumes (more daily costs)

### Break-Even Analysis

```
For profitability:
Price Spread × Volume > Storage Costs

Example:
$1.43 × 700K = $1.00M revenue
$1.28M storage costs
→ LOSS of $280K
```

**Minimum profitable spread:**
```
Required Spread = Storage Costs / Volume
                = $1.28M / 700K
                = $1.83/MMBtu
```

---

##  Code Structure

```python
def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    injection_rate,
    withdrawal_rate,
    max_volume,
    storage_cost_per_day
):
    """
    Simulates and prices a gas storage contract.
    
    Returns:
    --------
    dict with:
        - net_value: Total contract value
        - total_injected: Total volume injected
        - total_withdrawn: Total volume withdrawn
        - total_storage_cost: Accumulated storage costs
        - daily_data: DataFrame with daily breakdown
    """
```

---

##  Visualizations

### Volume Evolution Over Time
Shows how storage fills up (injection phase) and empties (withdrawal phase)

**Test 1 Graph:**
- Rapid fill to 700K MMBtu
- 6-month plateau (costs accumulating!)
- Rapid withdrawal

**Test 2 Graph:**
- Slower fill (50K vs 100K rate)
- Hits 800K capacity limit
- Longer plateau = higher costs

---

##  Business Applications

### Decision Support
1. **Contract Evaluation:** Should we accept this contract?
2. **Pricing:** What premium to charge for this service?
3. **Optimization:** What injection/withdrawal schedule maximizes profit?
4. **Risk Assessment:** What if prices don't move as expected?

### Recommendations

**For Profitable Contracts:**
- Target spreads >$2/MMBtu
- Minimize storage duration (3-4 months max)
- Inject during summer low (May-July)
- Withdraw during winter high (Dec-Feb)

**Red Flags:**
- Spreads <$1.50/MMBtu
- Duration >6 months
- High storage costs (>$0.015/MMBtu/day)
- Small volumes (fixed costs don't amortize well)

---

##  Advanced Features

### Integration with Task 1
Prices are automatically predicted using the ML model from Task 1:
```python
price = predict_price(date)  # Uses Task 1 model
```

### Flexibility
- Handles any number of injection/withdrawal dates
- Works with variable rates and costs
- Adapts to any contract duration

---

##  Files

```
task2-storage-contract-pricing/
├── README.md (this file)
├── task2_advanced_pricing.py
└── outputs/
    └── contract_simulation.png
```

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **pandas:** Time series and data manipulation
- **numpy:** Numerical calculations
- **matplotlib:** Visualization
- **datetime:** Date handling

---

##  Dependencies

- **Task 1:** Uses `predict_price()` function for price forecasting

---

##  Future Enhancements

1. **Optimization:** Find optimal injection/withdrawal schedule
2. **Uncertainty:** Add price volatility and confidence intervals
3. **Transport costs:** Include variable transport fees
4. **Multiple facilities:** Compare different storage locations
5. **Real options:** Value flexibility as a real option

---

*Completed as part of JP Morgan Quantitative Research Virtual Experience*
