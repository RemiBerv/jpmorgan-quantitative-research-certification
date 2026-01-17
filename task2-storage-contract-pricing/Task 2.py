# Task 2

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('Nat_Gas.csv')
df['Dates'] = pd.to_datetime(df['Dates'])
df = df.sort_values('Dates')

# Time features
first_date = df['Dates'].min()
df['days_since_start'] = (df['Dates'] - first_date).dt.days
df['month'] = df['Dates'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Interpolator on historical prices
interpolator = interp1d(
    df['days_since_start'],
    df['Prices'],
    kind='cubic',
    fill_value='extrapolate'
)

# Linear regression model for extrapolation
X = df[['days_since_start', 'month_sin', 'month_cos']].values
y = df['Prices'].values
model = LinearRegression()
model.fit(X, y)

def predict_price(date_input):
    """Predict natural gas price for a given date."""
    if isinstance(date_input, str):
        try:
            target_date = pd.to_datetime(date_input, dayfirst=True)
        except Exception:
            target_date = pd.to_datetime(date_input)
    else:
        target_date = date_input
    
    days = (target_date - first_date).days
    month = target_date.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    last_date = df['Dates'].max()
    
    if target_date <= last_date:
        price = float(interpolator(days))
    else:
        X_new = np.array([[days, month_sin, month_cos]])
        price = float(model.predict(X_new)[0])
    
    return price

def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    injection_rate,
    withdrawal_rate,
    max_volume,
    storage_cost_per_day
):
    """Simulate a gas storage contract and compute its net value."""
    
    injection_dates = [pd.to_datetime(d) for d in injection_dates]
    withdrawal_dates = [pd.to_datetime(d) for d in withdrawal_dates]
    
    injection_dates = sorted(injection_dates)
    withdrawal_dates = sorted(withdrawal_dates)
    
    print(f"Injection rate        : {injection_rate:,.0f} MMBtu/day")
    print(f"Withdrawal rate       : {withdrawal_rate:,.0f} MMBtu/day")
    print(f"Maximum capacity      : {max_volume:,.0f} MMBtu")
    print(f"Storage cost          : ${storage_cost_per_day:.4f}/MMBtu/day")
    print(f"Injection dates       : {len(injection_dates)} dates")
    print(f"Withdrawal dates      : {len(withdrawal_dates)} dates")
    print()
    
    start_date = min(injection_dates)
    end_date = max(withdrawal_dates)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    daily_data = []
    current_volume = 0
    total_injected = 0
    total_withdrawn = 0
    
    print()
    
    for current_date in date_range:
        
        injection = 0
        withdrawal = 0
        price = predict_price(current_date)
        
        # Injection day
        if current_date in injection_dates:
            can_inject = min(
                injection_rate,
                max_volume - current_volume
            )
            
            injection = can_inject
            total_injected += injection
            current_volume += injection
        
        # Withdrawal day
        if current_date in withdrawal_dates:
            can_withdraw = min(
                withdrawal_rate,
                current_volume
            )
            
            withdrawal = can_withdraw
            total_withdrawn += withdrawal
            current_volume -= withdrawal
        
        purchase_cost = injection * price if injection > 0 else 0
        sale_revenue = withdrawal * price if withdrawal > 0 else 0
        storage_cost = current_volume * storage_cost_per_day
        
        daily_cash_flow = sale_revenue - purchase_cost - storage_cost
        
        daily_data.append({
            'date': current_date,
            'price': price,
            'injection': injection,
            'withdrawal': withdrawal,
            'volume_stored': current_volume,
            'purchase_cost': purchase_cost,
            'sale_revenue': sale_revenue,
            'storage_cost': storage_cost,
            'daily_cash_flow': daily_cash_flow
        })
        
        if injection > 0 or withdrawal > 0:
            action = ""
            if injection > 0:
                action += f"INJECTION {injection:,.0f} MMBtu at ${price:.2f}"
            if withdrawal > 0:
                if action:
                    action += " | "
                action += f"WITHDRAWAL {withdrawal:,.0f} MMBtu at ${price:.2f}"
            
            print(f"{current_date.strftime('%Y-%m-%d')} : {action}")
            print(f"   → Volume stored: {current_volume:,.0f} MMBtu")
    
    print()
    
    df_sim = pd.DataFrame(daily_data)
    
    total_purchase = df_sim['purchase_cost'].sum()
    total_sales = df_sim['sale_revenue'].sum()
    total_storage_cost = df_sim['storage_cost'].sum()
    net_value = total_sales - total_purchase - total_storage_cost
    
    print()
    print(" INJECTIONS:")
    print(f"   Total injected volume   : {total_injected:,.0f} MMBtu")
    print(f"   Total purchase cost     : ${total_purchase:,.2f}")
    print(f"   Average purchase price  : ${total_purchase/total_injected:.2f}/MMBtu")
    print()
    print(" WITHDRAWALS:")
    print(f"   Total withdrawn volume  : {total_withdrawn:,.0f} MMBtu")
    print(f"   Total sales revenue     : ${total_sales:,.2f}")
    print(f"   Average selling price   : ${total_sales/total_withdrawn:.2f}/MMBtu")
    print()
    print(" COSTS:")
    print(f"   Storage cost            : ${total_storage_cost:,.2f}")
    print()
    print(f" NET CONTRACT VALUE        : ${net_value:,.2f}")
    print()
    
    if net_value > 0:
        print(f" PROFITABLE CONTRACT (+${net_value:,.2f})")
        roi = (net_value / (total_purchase + total_storage_cost)) * 100
        print(f"   ROI: {roi:.2f}%")
    else:
        print(f" NON-PROFITABLE CONTRACT (${net_value:,.2f})")
        print(f"   LOSS: ${abs(net_value):,.2f}")
    
    print()
    
    return {
        'net_value': net_value,
        'total_injected': total_injected,
        'total_withdrawn': total_withdrawn,
        'total_purchase': total_purchase,
        'total_sales': total_sales,
        'total_storage_cost': total_storage_cost,
        'daily_data': df_sim,
        'utilization': (total_injected / max_volume) * 100
    }

# Scenario 1
injection_dates_1 = pd.date_range('2025-06-01', periods=7, freq='D')
withdrawal_dates_1 = pd.date_range('2025-12-01', periods=7, freq='D')

result1 = price_storage_contract(
    injection_dates=injection_dates_1,
    withdrawal_dates=withdrawal_dates_1,
    injection_rate=100_000,
    withdrawal_rate=100_000,
    max_volume=1_000_000,
    storage_cost_per_day=0.01
)

# Scenario 2
injection_dates_2 = pd.date_range('2025-06-01', periods=30, freq='D')
withdrawal_dates_2 = pd.date_range('2025-12-01', periods=15, freq='D')

result2 = price_storage_contract(
    injection_dates=injection_dates_2,
    withdrawal_dates=withdrawal_dates_2,
    injection_rate=50_000,
    withdrawal_rate=100_000,
    max_volume=800_000,
    storage_cost_per_day=0.015
)

# Plots
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

ax1 = axes[0]
df1 = result1['daily_data']
ax1.fill_between(df1['date'], 0, df1['volume_stored'], alpha=0.3, color='blue')
ax1.plot(df1['date'], df1['volume_stored'], 'b-', linewidth=2)
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Stored volume (MMBtu)', fontsize=11)
ax1.set_title('TEST 1: Stored volume evolution', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1_000_000, color='red', linestyle='--', label='Max capacity (1M)')
ax1.legend()

ax2 = axes[1]
df2 = result2['daily_data']
ax2.fill_between(df2['date'], 0, df2['volume_stored'], alpha=0.3, color='green')
ax2.plot(df2['date'], df2['volume_stored'], 'g-', linewidth=2)
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('Stored volume (MMBtu)', fontsize=11)
ax2.set_title('TEST 2: Evolution with capacity constraint', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=800_000, color='red', linestyle='--', label='Max capacity (800K)')
ax2.legend()

plt.tight_layout()
plt.savefig('contract_simulation.png', dpi=150)

print(f"   Test 1: ${result1['net_value']:,.2f} (Utilization: {result1['utilization']:.1f}%)")
print(f"   Test 2: ${result2['net_value']:,.2f} (Utilization: {result2['utilization']:.1f}%)")
