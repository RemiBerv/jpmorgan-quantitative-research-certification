import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')
fico_data = df[['fico_score', 'default']].copy()

print(f" {len(fico_data)} clients")

print(" FICO statistics:")
print(f"   Min: {fico_data['fico_score'].min()} | Max: {fico_data['fico_score'].max()}")
print(f"   Mean: {fico_data['fico_score'].mean():.0f} | Median: {fico_data['fico_score'].median():.0f}")
print()

print(" Default rate by FICO range:")
bins_analyze = [400, 500, 550, 600, 650, 700, 750, 850]
fico_data['range'] = pd.cut(fico_data['fico_score'], bins=bins_analyze)
analysis = fico_data.groupby('range')['default'].agg(['count', 'sum', 'mean'])
analysis.columns = ['Count', 'Defaults', 'Default_Rate']
print(analysis)

# -------------------------------------------------------------------------
# METHOD 1: QUANTILES
# -------------------------------------------------------------------------
print("\n" + "="*80)
print(" METHOD 1: QUANTILE BUCKETING")
print("="*80)

def create_quantile_buckets(scores: np.ndarray, n_buckets: int = 10) -> List[float]:
    """Create buckets based on quantiles (uniform distribution)."""
    quantiles = np.linspace(0, 1, n_buckets + 1)
    boundaries = [scores.min()]
    for q in quantiles[1:-1]:
        boundaries.append(np.quantile(scores, q))
    boundaries.append(scores.max() + 1)
    return boundaries

scores = fico_data['fico_score'].values
defaults = fico_data['default'].values

quantile_boundaries = create_quantile_buckets(scores, n_buckets=10)

print(" Quantile boundaries:")
for i in range(len(quantile_boundaries) - 1):
    print(f"   Bucket {i+1}: [{quantile_boundaries[i]:.0f}, {quantile_boundaries[i+1]:.0f})")
print()

# -------------------------------------------------------------------------
# METHOD 2: MSE OPTIMIZATION
# -------------------------------------------------------------------------
print("="*80)
print(" METHOD 2: MSE OPTIMIZATION")
print("="*80)
print()

def create_mse_buckets_greedy(scores: np.ndarray, n_buckets: int = 10) -> List[float]:
    """
    Create buckets minimizing MSE using a fast greedy approach.
    
    Algorithm:
    1. Start with n_buckets equal segments
    2. Iteratively improve boundaries
    3. Converge to local optimum
    """
    boundaries = create_quantile_buckets(scores, n_buckets)
    
    def calc_mse(bounds):
        mse = 0
        for i in range(len(bounds) - 1):
            mask = (scores >= bounds[i]) & (scores < bounds[i+1])
            bucket_scores = scores[mask]
            if len(bucket_scores) > 0:
                mse += np.sum((bucket_scores - bucket_scores.mean()) ** 2)
        return mse
    
    for iteration in range(5):
        for i in range(1, len(boundaries) - 1):
            original = boundaries[i]
            best_mse = calc_mse(boundaries)
            best_val = original
            
            test_range = np.linspace(boundaries[i-1] + 1, boundaries[i+1] - 1, 10)
            for test_val in test_range:
                boundaries[i] = test_val
                new_mse = calc_mse(boundaries)
                if new_mse < best_mse:
                    best_mse = new_mse
                    best_val = test_val
            
            boundaries[i] = best_val
    
    return boundaries

mse_boundaries = create_mse_buckets_greedy(scores, n_buckets=10)

print(" Optimal MSE boundaries:")
for i in range(len(mse_boundaries) - 1):
    print(f"   Bucket {i+1}: [{mse_boundaries[i]:.0f}, {mse_boundaries[i+1]:.0f})")
print()

# -------------------------------------------------------------------------
# METHOD 3: DEFAULT RATE OPTIMIZATION
# -------------------------------------------------------------------------
print("="*80)
print(" METHOD 3: DEFAULT-RATE BASED BUCKETING")
print("="*80)
print()

def create_default_rate_buckets(scores: np.ndarray, defaults: np.ndarray, 
                               n_buckets: int = 10) -> List[float]:
    """
    Create buckets finding homogeneous default rate zones.
    
    Algorithm:
    1. Sort scores
    2. Compute rolling default rate
    3. Identify significant change points
    4. Create buckets around these points
    """
    data = pd.DataFrame({'score': scores, 'default': defaults})
    data = data.sort_values('score').reset_index(drop=True)
    
    window = len(data) // 20
    data['default_rate_rolling'] = data['default'].rolling(window=window, center=True).mean()
    
    data['rate_change'] = data['default_rate_rolling'].diff().abs()
    
    split_points = data.nlargest(n_buckets - 1, 'rate_change')['score'].values
    split_points = np.sort(split_points)
    
    boundaries = [scores.min()]
    boundaries.extend(split_points)
    boundaries.append(scores.max() + 1)
    
    boundaries = sorted(list(set(boundaries)))
    
    while len(boundaries) < n_buckets + 1:
        max_gap_idx = np.argmax(np.diff(boundaries))
        new_boundary = (boundaries[max_gap_idx] + boundaries[max_gap_idx + 1]) / 2
        boundaries.insert(max_gap_idx + 1, new_boundary)
        boundaries = sorted(boundaries)
    
    return boundaries[:n_buckets + 1]

default_rate_boundaries = create_default_rate_buckets(scores, defaults, n_buckets=10)

print(" Default-rate based boundaries:")
for i in range(len(default_rate_boundaries) - 1):
    print(f"   Bucket {i+1}: [{default_rate_boundaries[i]:.0f}, {default_rate_boundaries[i+1]:.0f})")
print()

# -------------------------------------------------------------------------
# COMPARISON OF 3 METHODS
# -------------------------------------------------------------------------
print("="*80)
print(" COMPARISON OF 3 METHODS")
print("="*80)
print()

def analyze_buckets(scores, defaults, boundaries, method_name):
    """Detailed bucket analysis."""
    print(f"\n🔍 {method_name}")
    print("-" * 80)
    
    results = []
    for i in range(len(boundaries) - 1):
        mask = (scores >= boundaries[i]) & (scores < boundaries[i+1])
        n = mask.sum()
        k = defaults[mask].sum()
        p = k / n if n > 0 else 0
        avg_fico = scores[mask].mean() if n > 0 else 0
        
        results.append({
            'Bucket': i + 1,
            'Range': f"[{boundaries[i]:.0f}, {boundaries[i+1]:.0f})",
            'Count': int(n),
            'Defaults': int(k),
            'Default_Rate%': f"{p*100:.2f}",
            'Avg_FICO': f"{avg_fico:.0f}"
        })
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    mse = 0
    for i in range(len(boundaries) - 1):
        mask = (scores >= boundaries[i]) & (scores < boundaries[i+1])
        bucket_scores = scores[mask]
        if len(bucket_scores) > 0:
            mse += np.sum((bucket_scores - bucket_scores.mean()) ** 2)
    
    print(f"\nTotal MSE: {mse:,.2f}")
    return df_results

results_quantile = analyze_buckets(scores, defaults, quantile_boundaries, "QUANTILES")
results_mse = analyze_buckets(scores, defaults, mse_boundaries, "MSE OPTIMIZED")
results_default = analyze_buckets(scores, defaults, default_rate_boundaries, "DEFAULT-BASED")

# -------------------------------------------------------------------------
# VISUALIZATION
# -------------------------------------------------------------------------


fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Histogram + Quantile boundaries
ax1 = axes[0, 0]
ax1.hist(scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
for b in quantile_boundaries[1:-1]:
    ax1.axvline(x=b, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax1.set_title('Method 1: Quantiles', fontsize=13, fontweight='bold')
ax1.set_xlabel('FICO Score')
ax1.set_ylabel('Frequency')
ax1.grid(True, alpha=0.3)

# Histogram + MSE boundaries
ax2 = axes[0, 1]
ax2.hist(scores, bins=50, alpha=0.7, color='green', edgecolor='black')
for b in mse_boundaries[1:-1]:
    ax2.axvline(x=b, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax2.set_title('Method 2: MSE Optimized', fontsize=13, fontweight='bold')
ax2.set_xlabel('FICO Score')
ax2.set_ylabel('Frequency')
ax2.grid(True, alpha=0.3)

# Histogram + Default-rate boundaries
ax3 = axes[1, 0]
ax3.hist(scores, bins=50, alpha=0.7, color='orange', edgecolor='black')
for b in default_rate_boundaries[1:-1]:
    ax3.axvline(x=b, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax3.set_title('Method 3: Default-rate based', fontsize=13, fontweight='bold')
ax3.set_xlabel('FICO Score')
ax3.set_ylabel('Frequency')
ax3.grid(True, alpha=0.3)

# Default rates comparison
ax4 = axes[1, 1]

def get_default_rates(boundaries):
    rates = []
    for i in range(len(boundaries) - 1):
        mask = (scores >= boundaries[i]) & (scores < boundaries[i+1])
        rate = defaults[mask].mean() * 100 if mask.sum() > 0 else 0
        rates.append(rate)
    return rates

x = range(1, 11)
rates_q = get_default_rates(quantile_boundaries)
rates_mse = get_default_rates(mse_boundaries)
rates_def = get_default_rates(default_rate_boundaries)

ax4.plot(x, rates_q, 'o-', label='Quantiles', linewidth=2, markersize=8)
ax4.plot(x, rates_mse, 's-', label='MSE', linewidth=2, markersize=8)
ax4.plot(x, rates_def, '^-', label='Default-based', linewidth=2, markersize=8)
ax4.set_title('Comparison: Default rate per bucket', fontsize=13, fontweight='bold')
ax4.set_xlabel('Bucket')
ax4.set_ylabel('Default rate (%)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fico_buckets_final.png', dpi=150)

# -------------------------------------------------------------------------
# FINAL FUNCTION
# -------------------------------------------------------------------------


def create_fico_rating_map(fico_scores: np.ndarray, 
                          defaults: np.ndarray = None,
                          n_buckets: int = 10,
                          method: str = 'default_based') -> Dict:
    """
    Create an optimized FICO rating map.
    
    Parameters
    ----------
    fico_scores : np.ndarray
        FICO scores array.
    defaults : np.ndarray, optional
        Default flags (required for 'default_based' method).
    n_buckets : int, default=10
        Number of rating buckets.
    method : str, default='default_based'
        'quantile', 'mse', or 'default_based'.
    
    Returns
    -------
    Dict
        Contains boundaries, rating_map, and ratings array.
    """
    if method == 'quantile':
        boundaries = create_quantile_buckets(fico_scores, n_buckets)
    elif method == 'mse':
        boundaries = create_mse_buckets_greedy(fico_scores, n_buckets)
    elif method == 'default_based':
        if defaults is None:
            raise ValueError("defaults required for method='default_based'")
        boundaries = create_default_rate_buckets(fico_scores, defaults, n_buckets)
    else:
        raise ValueError("method must be 'quantile', 'mse', or 'default_based'")
    
    rating_map = {}
    for i in range(len(boundaries) - 1):
        bucket_label = n_buckets - i  # Higher rating for higher FICO
        rating_map[i] = {
            'rating': bucket_label,
            'lower_bound': boundaries[i],
            'upper_bound': boundaries[i+1]
        }
    
    ratings = np.zeros(len(fico_scores), dtype=int)
    for score_idx, score in enumerate(fico_scores):
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= score < boundaries[i+1]:
                ratings[score_idx] = n_buckets - i
                break
    
    return {
        'boundaries': boundaries,
        'rating_map': rating_map,
        'ratings': ratings
    }

# Test
result = create_fico_rating_map(scores, defaults, n_buckets=10, method='default_based')

print(" Function create_fico_rating_map() created.\n")
