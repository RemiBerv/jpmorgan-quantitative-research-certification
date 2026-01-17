import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('Nat_Gas.csv')

# Parse and sort dates
df['Dates'] = pd.to_datetime(df['Dates'])
df = df.sort_values('Dates')

# Time series plot
plt.figure(figsize=(14, 5))
plt.plot(df['Dates'], df['Prices'], 'b-o', linewidth=2, markersize=4)
plt.xlabel('Date', fontsize=11)
plt.ylabel('Price ($/MMBtu)', fontsize=11)
plt.title('Price Evolution', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('time_series.png', dpi=150)

# Monthly average prices
df['month'] = df['Dates'].dt.month
monthly_avg = df.groupby('month')['Prices'].mean()

# Best and worst months
max_month = monthly_avg.idxmax()  # idxmax = index of maximum
min_month = monthly_avg.idxmin()  # idxmin = index of minimum

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

print(f"   Highest month: {month_names[max_month-1]} (${monthly_avg[max_month]:.2f})")
print(f"   Cheapest month: {month_names[min_month-1]} (${monthly_avg[min_month]:.2f})")
print(f"   Difference: ${monthly_avg[max_month] - monthly_avg[min_month]:.2f}")

# Cyclicity bar chart
plt.figure(figsize=(12, 5))
plt.bar(range(1, 13), monthly_avg.values, color='steelblue', alpha=0.7, edgecolor='black')
plt.axhline(
    y=df['Prices'].mean(),
    color='red',
    linestyle='--',
    linewidth=2,
    label=f'Average: ${df["Prices"].mean():.2f}'
)
plt.xlabel('Month', fontsize=11)
plt.ylabel('Avg Price ($/MMBtu)', fontsize=11)
plt.title('Seasonality: Average Price per Month', fontsize=13, fontweight='bold')
plt.xticks(range(1, 13), month_names)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('2_seasonality.png', dpi=150)

# Days since start
first_date = df['Dates'].min()
df['days_since_start'] = (df['Dates'] - first_date).dt.days

print(f"   First date = {first_date.strftime('%d/%m/%Y')} → day 0")
print(f"   Last date = {df['Dates'].max().strftime('%d/%m/%Y')} → day {df['days_since_start'].max()}")

# Cyclical features
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Interpolator
interpolator = interp1d(
    df['days_since_start'],
    df['Prices'],
    kind='cubic',
    fill_value='extrapolate'
)

# Linear regression model
X = df[['days_since_start', 'month_sin', 'month_cos']].values
y = df['Prices'].values

model = LinearRegression()
model.fit(X, y)

r2_score = model.score(X, y)

print(f"   → Model quality (R²): {r2_score:.4f}")
if r2_score > 0.7:
    print("   → Excellent model!")
elif r2_score > 0.5:
    print("   → Good model")
else:
    print("   → Average model (can be improved)")
print()

# Prediction function
def predict_price(date_input):
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
        method = "interpolation"
    else:
        X_new = np.array([[days, month_sin, month_cos]])
        price = float(model.predict(X_new)[0])
        method = "extrapolation"

    return price, method

print("    Function predict_price() created")
print("   → It can predict the price for ANY date!")
print()

# Test 1: Historical dates (interpolation)
print("📅 TEST 1: Historical dates (interpolation)")
test_dates_historical = [
    '15/11/2020',
    '15/06/2021',
    '25/12/2023',
]

for date in test_dates_historical:
    price, method = predict_price(date)
    print(f"   {date} → ${price:.2f} ({method})")
print()

# Test 2: Future dates (extrapolation)
print(" TEST 2: Future dates (extrapolation)")
test_dates_future = [
    '31/03/2025',
    '30/06/2025',
    '31/12/2025',
    '30/09/2026',
]

for date in test_dates_future:
    price, method = predict_price(date)
    print(f"   {date} → ${price:.2f} ({method})")
print()

# Future monthly predictions
last_date = df['Dates'].max()
future_dates = pd.date_range(
    start=last_date + timedelta(days=30),
    periods=12,
    freq='MS'
)

future_prices = []
for date in future_dates:
    price, _ = predict_price(date)
    future_prices.append(price)

plt.figure(figsize=(15, 6))

# Historical data
plt.plot(
    df['Dates'],
    df['Prices'],
    'b-o',
    linewidth=2,
    markersize=5,
    label='Historical data',
    alpha=0.7
)

# Future predictions
plt.plot(
    future_dates,
    future_prices,
    'r--o',
    linewidth=2,
    markersize=5,
    label='Predictions (extrapolation)',
    alpha=0.7
)

# Last observed date marker
plt.axvline(
    x=last_date,
    color='green',
    linestyle=':',
    linewidth=2,
    label=f'Last observation ({last_date.strftime("%d/%m/%Y")})'
)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Price ($/MMBtu)', fontsize=12)
plt.title('Natural Gas Price: Historical Data and Future Predictions', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('3_final_predictions.png', dpi=150)
