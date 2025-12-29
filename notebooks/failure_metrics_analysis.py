# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Multiuser Webchat Prometheus Data Analysis
#
# ## Test Phases Overview
# - **Phase 1 (Baseline)**: Normal operation with 100 users, ~0.3 msg/sec/user
# - **Phase 2 (Throughput Stress)**: 4 levels of message rate escalation (moderate → insane)
# - **Phase 3 (Bandwidth Stress)**: 5 levels of message size escalation (10KB → 1000KB)
# - **Phase 4 (Connection Stress)**: 6 levels of concurrent users (100 → 600)

# ## 1. Library Setup and Configuration
#

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Visualization style settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Font settings for Korean characters (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # Prevent minus sign rendering issues

# Default figure size
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 100
# -

# ## 2. Data Loading and Preprocessing
#

# +
# Data path configuration
BASE_PATH = Path('load_test_results')
PHASE_PATHS = {
    'phase1_baseline': BASE_PATH / 'phase1_baseline_20251228_194309',
    'phase2_throughput': BASE_PATH / 'phase2_throughput_stress_20251228_200708',
    'phase3_bandwidth': BASE_PATH / 'phase3_bandwidth_stress_20251228_201715',
    'phase4_connection': BASE_PATH / 'phase4_connection_stress_20251228_204327'
}

def load_phase_data(phase_path):
    """
    Load metrics data for a single test phase.

    Args:
        phase_path: Path to the phase directory containing metrics.csv

    Returns:
        DataFrame with parsed timestamps and elapsed_seconds column
    """
    metrics_file = phase_path / 'metrics.csv'
    if not metrics_file.exists():
        print(f"Warning: {metrics_file} not found")
        return None

    # Load CSV with pandas
    df = pd.read_csv(metrics_file)

    # Parse timestamp column to datetime objects
    # Vector operation: applies pd.to_datetime to entire column at once
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Calculate elapsed time in seconds from start
    # Vector operation: (df['timestamp'] - df['timestamp'].iloc[0]) computes time delta for all rows
    # .dt.total_seconds() converts timedelta to float seconds
    df['elapsed_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

    return df

def identify_violations(df, metric, safe_zone_upper):
    """
    Identify when a metric violates its safe zone threshold.

    Args:
        df: DataFrame containing the metric data
        metric: Name of the metric column to check
        safe_zone_upper: Upper bound threshold for the safe zone

    Returns:
        DataFrame with violation records (timestamp, metric value, elapsed time)
    """
    # Boolean vector: True where metric exceeds safe zone
    # This is a vectorized comparison across entire column
    violations = df[df[metric] > safe_zone_upper]

    if len(violations) > 0:
        # Select only relevant columns using fancy indexing
        return violations[['timestamp', metric, 'elapsed_seconds']]
    return pd.DataFrame()

# Load all phase data
print("Loading phase data...")
phase_data = {}
for phase_name, phase_path in PHASE_PATHS.items():
    df = load_phase_data(phase_path)
    if df is not None:
        phase_data[phase_name] = df
        print(f"  {phase_name}: {len(df)} data points")

# Create convenient references for individual phases
baseline_df = phase_data.get('phase1_baseline')
throughput_df = phase_data.get('phase2_throughput')
bandwidth_df = phase_data.get('phase3_bandwidth')
connection_df = phase_data.get('phase4_connection')

print(f"\nLoaded {len(phase_data)} phases successfully")
# -

# ### 2.1 Collected Metrics Description
#
# Documentation of Prometheus metrics collected from the system.

baseline_df.info()

# ### Metric descriptions
# **Timestamp**
# timestamp: Metric collection time (UTC)
#
# **Connection metrics**
# connected_users: Number of currently connected WebSocket clients
# conn_attempt_rate: Connection attempt rate per second
# conn_success_rate: Connection success rate (successful/total attempts)
# conn_fail_rate: Connection failure rate per second
# disconn_rate: Disconnection rate per second
#
# **Message processing**
# message_rate: Messages processed per second (msg/s)
#
# **Latency metrics (milliseconds)**
# e2e_latency_p95: End-to-end latency 95th percentile - client send to all clients receive (ms)
# e2e_latency_p99: End-to-end latency 99th percentile (ms)
#
# **Event loop and Redis metrics**
# eventloop_lag_p95: Event loop lag 95th percentile - blocking operation indicator (ms)
# eventloop_lag_p99: Event loop lag 99th percentile (ms)
# redis_stream_lag: Redis Stream consumer lag - delay in message consumption
# redis_op_latency_p95: Redis operation latency 95th percentile (ms)
#
# **System resources**
# memory_bytes: Memory usage in bytes
#
# **Error metrics**
# error_rate: Error occurrence rate per second
#
# **Derived metrics**
# elapsed_seconds: Seconds elapsed since test start

# ### 2.2 Exploratory Data Analysis (EDA)
#
# Examine missing values and basic statistics for each phase's collected metrics.

# +
print("=" * 80)
print("Phase-wise Metrics EDA")
print("=" * 80)

for phase_name, df in phase_data.items():
    print(f"\n{'='*80}")
    print(f"{phase_name.upper()}")
    print(f"{'='*80}")

    # Basic information
    print(f"\n1. Basic Info:")
    print(f"   - Data points: {len(df)}")
    print(f"   - Columns: {len(df.columns)}")
    print(f"   - Duration: {df['elapsed_seconds'].max():.1f}s ({df['elapsed_seconds'].max()/60:.1f}min)")

    # Missing value analysis
    # Vector operation: isnull() returns boolean DataFrame, sum() aggregates along columns
    missing = df.isnull().sum()
    # Vector operation: divide entire Series by scalar (len(df)) and multiply by 100
    missing_pct = (missing / len(df)) * 100

    # Boolean indexing: select only columns where percentage > 0
    missing_cols = missing_pct[missing_pct > 0]

    print(f"\n2. Missing Values:")
    if len(missing_cols) > 0:
        for col, pct in missing_cols.items():
            print(f"   - {col}: {missing[col]} ({pct:.1f}%)")
    else:
        print("   - No missing values")

    # Numeric columns summary statistics
    # Vector operation: select_dtypes returns subset of columns by data type
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    print(f"\n3. Numeric Metrics:")
    print(f"   - Count: {len(numeric_cols)}")
    print(f"   - Columns: {', '.join(list(numeric_cols)[:5])}...")

    # Display summary statistics for key metrics
    # Vector operation: describe() computes count, mean, std, min, quartiles, max for all numeric columns
    key_metrics = ['connected_users', 'message_rate', 'e2e_latency_p95', 'error_rate']
    # List comprehension with conditional: filter metrics that exist in the DataFrame
    available_metrics = [m for m in key_metrics if m in df.columns]

    if available_metrics:
        print(f"\n4. Key Metrics Summary:")
        # Select multiple columns at once and compute statistics
        summary = df[available_metrics].describe()
        print(summary.round(2))


# -

# ## 3. Cross-Phase Comparison Overview
#

# ### Leading Indicator Analysis via Lagged Correlation
#
# Define a function to find metrics that change before errors/failures occur (leading indicators).

# +
def calculate_lagged_correlations(df, target_col, candidate_metrics, lags=[0, 5, 10, 15, 20, 30]):
    """
    Calculate lagged correlations to identify leading indicators (optimized with vector operations).

    This function computes Pearson correlation between a target metric (e.g., error_rate) and
    candidate metrics at various time lags. A positive lag means the candidate metric changes
    *before* the target metric, making it a potential leading indicator.

    Args:
        df: DataFrame containing time series data
        target_col: Target metric column name (e.g., 'error_rate', 'failure_count')
        candidate_metrics: List of candidate metric column names to test
        lags: List of time lags in seconds to test (default: [0, 5, 10, 15, 20, 30])

    Returns:
        corr_matrix: DataFrame with metrics as rows and lags as columns, values are correlations

    Vector Operations Used:
        - df[metric].shift(lag): Shifts entire column backward by 'lag' positions
        - df[target_col].corr(shifted): Computes Pearson correlation between two Series
        - pd.DataFrame(data, index, columns): Constructs DataFrame from nested dict
    """
    # Dictionary to store correlation values: {metric_name: {lag: correlation}}
    corr_data = {}

    # Iterate through each candidate metric
    for metric in candidate_metrics:
        if metric not in df.columns:
            continue

        # Store correlations at different lags for this metric
        corr_data[metric] = {}
        """
        # Test each time lag
        for lag in lags:
            # Shift metric backward by 'lag' positions
            # This makes the metric's earlier values align with target's later values
            # Vector operation: shift() operates on entire column without loops
            shifted_metric = df[metric].shift(lag)

            # Calculate Pearson correlation between shifted metric and target
            # Vector operation: corr() computes correlation using vectorized numpy operations
            # Only uses rows where both values are non-null
            correlation = df[target_col].corr(shifted_metric)

            # Store the correlation value
            corr_data[metric][lag] = correlation
        """
    # Convert nested dictionary to DataFrame
    # Rows = metrics, Columns = lags, Values = correlations
    corr_matrix = pd.DataFrame(corr_data).T

    # Sort by absolute correlation at lag=0 (descending)
    # Vector operation: abs() applied to entire column, sort_values() reorders rows
    if 0 in lags:
        corr_matrix = corr_matrix.sort_values(by=0, key=abs, ascending=False) # ===>.abs df에 바로 map key 말고

    return corr_matrix

def find_top_leading_indicators(corr_matrix, top_n=3):
    """
    Identify top leading indicators from lagged correlation matrix.

    Args:
        corr_matrix: DataFrame from calculate_lagged_correlations()
        top_n: Number of top indicators to return

    Returns:
        List of tuples: (metric_name, optimal_lag, correlation_value)
    """
    results = []
    for metric in corr_matrix.index:
        # Vector operation: abs() on entire row, idxmax() finds column with max value
        row = corr_matrix.loc[metric]
        if row.isna().all():
            continue

        # skipna=True ensures we skip NaN values
        optimal_lag = row.abs().idxmax(skipna=True)
        if pd.isna(optimal_lag):
            continue

        max_corr = corr_matrix.loc[metric, optimal_lag]
        results.append((metric, optimal_lag, max_corr))

    # Sort by absolute correlation value (descending)
    results.sort(key=lambda x: abs(x[2]), reverse=True)

    return results[:top_n]

def find_top_predictors(corr_matrix, top_n=3):
    """
    최고 예측 지표 찾기

    Args:
        corr_matrix: DataFrame from calculate_lagged_correlations()
                    (rows=metrics, columns=lags)
        top_n: Number of top indicators to return

    Returns:
        List of dicts: [{'metric': name, 'lag': lag, 'correlation': value}, ...]

    벡터 연산 사용:
    - corr_matrix.abs().max(axis=1) : 각 메트릭(행)의 최대 절대값 상관계수
    - corr_matrix.loc[metric].abs().idxmax() : 최대값의 인덱스(lag) 찾기
    """
    if corr_matrix is None or corr_matrix.empty:
        return []

    # 각 메트릭별 최대 상관계수 (벡터 연산: axis=1로 각 행의 최대값)
    max_corrs = corr_matrix.abs().max(axis=1)

    # NaN 제거
    max_corrs = max_corrs.dropna()

    if max_corrs.empty:
        return []

    # 상위 N개 메트릭 선택 (벡터 연산: sort_values)
    top_metrics = max_corrs.sort_values(ascending=False).head(top_n)

    # 각 메트릭의 최적 lag 찾기
    predictors = []
    for metric in top_metrics.index:
        # Skip rows where all values are NaN
        row = corr_matrix.loc[metric]
        if row.isna().all():
            continue

        # 벡터 연산: idxmax()로 최대 상관계수의 lag 찾기
        best_lag = row.abs().idxmax(skipna=True)

        # Check if best_lag is NaN
        if pd.isna(best_lag):
            continue

        best_corr = corr_matrix.loc[metric, best_lag]

        predictors.append({
            'metric': metric,
            'lag': best_lag,
            'correlation': best_corr
        })

    return predictors

def plot_correlation_heatmap(corr_matrix, title="Lagged Correlation Heatmap", figsize=(12, 8)):
    """
    Visualize lagged correlation matrix as a heatmap.

    Args:
        corr_matrix: DataFrame from calculate_lagged_correlations()
        title: Plot title
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)

    # Create heatmap with diverging colormap (red-white-blue)
    # vmin=-1, vmax=1 ensures proper scaling for correlation values
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlation Coefficient'})

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Lag (seconds)', fontsize=12)
    plt.ylabel('Metrics', fontsize=12)
    plt.tight_layout()
    plt.show()

print("Leading indicator analysis functions defined successfully")

# +
# Define core metrics for analysis
CORE_METRICS = [
    ('e2e_latency_p95', 'E2E Latency P95 (ms)'),
    ('eventloop_lag_p95', 'Event Loop Lag P95 (ms)'),
    ('memory_bytes', 'Memory Usage (MB)'),
    ('error_rate', 'Error Rate (%)')
]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
# Flatten 2D array of axes to 1D for easy iteration
axes = axes.flatten()

colors = {
    'phase1_baseline': '#2ecc71',
    'phase2_throughput': '#3498db',
    'phase3_bandwidth': '#f39c12',
    'phase4_connection': '#e74c3c'
}

# Iterate through core metrics and plot on subplots
for idx, (metric, label) in enumerate(CORE_METRICS):
    ax = axes[idx]

    for phase_name, df in phase_data.items():
        if metric in df.columns:
            # Convert memory from bytes to MB for readability
            # Vector operation: divide entire column by scalar (1024*1024)
            y_data = df[metric] / (1024*1024) if metric == 'memory_bytes' else df[metric]

            # Plot time series with phase-specific color
            ax.plot(df['elapsed_seconds'], y_data,
                   label=phase_name.replace('_', ' ').title(),
                   color=colors[phase_name], alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Elapsed Time (seconds)', fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(f'{label} Across All Phases', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Core Metrics Comparison Across Test Phases',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()
# -

# ## 4. Phase 1: Baseline Analysis
#
# ### Objective
# Establish normal operating conditions and define **Safe Zones** for key metrics.
#
# ### Safe Zone Definition
# For each metric, the safe zone upper bound is calculated as:
# ```
# Safe Zone Upper = Mean + (2 × Standard Deviation)
# ```
# This represents approximately 95% of normal operation data points, assuming normal distribution.

# Calculate Safe Zones from baseline data
safe_zones = {}
for metric, _ in CORE_METRICS:
    if metric in baseline_df.columns:
        # Vector operations: mean() and std() compute statistics across entire column
        # These operations process all data points at once (no loops)
        mean_val = baseline_df[metric].mean()
        std_val = baseline_df[metric].std()

        # Safe zone: mean + 2 standard deviations (covers ~95% of normal data)
        safe_zones[metric] = mean_val + (2 * std_val)

        print(f"{metric:20s}: mean={mean_val:8.2f}, std={std_val:8.2f}, "
              f"safe_zone_upper={safe_zones[metric]:8.2f}")


# ### Load Level Mapping Functions
#
# Map elapsed time to actual load levels (message rate, message size, user count) for each phase.

# +
def get_message_rate_phase2(elapsed_sec):
    """
    Map elapsed time to message rate for Phase 2 (Throughput Stress).

    Phase 2 escalates message rate every 2 minutes:
    - 0-120s: 70 msg/s (moderate)
    - 120-240s: 200 msg/s (high)
    - 240-360s: 500 msg/s (very high)
    - 360+s: 1000 msg/s (extreme)

    Args:
        elapsed_sec: Seconds elapsed since test start (scalar or array)

    Returns:
        Message rate (msg/s)
    """
    # Use numpy vectorized conditionals for efficient array operations
    # np.where works element-wise on arrays without explicit loops
    if isinstance(elapsed_sec, (int, float)):
        if elapsed_sec < 120:
            return 70
        elif elapsed_sec < 240:
            return 200
        elif elapsed_sec < 360:
            return 500
        else:
            return 1000
    else:
        # Vectorized version for pandas Series or numpy array
        # Nested np.where evaluates conditions element-wise
        return np.where(elapsed_sec < 120, 70,
               np.where(elapsed_sec < 240, 200,
               np.where(elapsed_sec < 360, 500, 1000)))

def get_message_size_phase3(elapsed_sec):
    """
    Map elapsed time to message size for Phase 3 (Bandwidth Stress).

    Phase 3 escalates message size every 2 minutes:
    - 0-120s: 10 KB
    - 120-240s: 50 KB
    - 240-360s: 100 KB
    - 360-480s: 500 KB
    - 480+s: 1000 KB (1 MB)
    """
    if isinstance(elapsed_sec, (int, float)):
        if elapsed_sec < 120:
            return 10
        elif elapsed_sec < 240:
            return 50
        elif elapsed_sec < 360:
            return 100
        elif elapsed_sec < 480:
            return 500
        else:
            return 1000
    else:
        return np.where(elapsed_sec < 120, 10,
               np.where(elapsed_sec < 240, 50,
               np.where(elapsed_sec < 360, 100,
               np.where(elapsed_sec < 480, 500, 1000))))

def get_user_count_phase4(elapsed_sec):
    """
    Map elapsed time to user count for Phase 4 (Connection Stress).

    Phase 4 escalates concurrent users every 2 minutes:
    - 0-120s: 100 users
    - 120-240s: 200 users
    - 240-360s: 300 users
    - 360-480s: 400 users
    - 480-600s: 500 users
    - 600+s: 600 users
    """
    if isinstance(elapsed_sec, (int, float)):
        if elapsed_sec < 120:
            return 100
        elif elapsed_sec < 240:
            return 200
        elif elapsed_sec < 360:
            return 300
        elif elapsed_sec < 480:
            return 400
        elif elapsed_sec < 600:
            return 500
        else:
            return 600
    else:
        return np.where(elapsed_sec < 120, 100,
               np.where(elapsed_sec < 240, 200,
               np.where(elapsed_sec < 360, 300,
               np.where(elapsed_sec < 480, 400,
               np.where(elapsed_sec < 600, 500, 600)))))

print("Load level mapping functions defined")
# -

# ## 5. Phase 2: Throughput Stress Analysis
#
# ### Scenario
# - Message rate escalation every 2 minutes: 70 → 200 → 500 → 1000 msg/s
# - Objective: Identify message rate threshold where system performance degrades

# +
phase2_df = phase_data['phase2_throughput']

# Visualize safe zone violations
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, (metric, label) in enumerate(CORE_METRICS):
    if metric not in phase2_df.columns:
        continue

    ax = axes[idx]

    # Convert memory to MB if needed
    # Vector operation: conditional division applied to entire column
    y_data = phase2_df[metric] / (1024*1024) if metric == 'memory_bytes' else phase2_df[metric]
    safe_zone = safe_zones[metric] / (1024*1024) if metric == 'memory_bytes' else safe_zones[metric]

    # Plot metric over time
    ax.plot(phase2_df['elapsed_seconds'], y_data,
           label=f'{label}', color='#3498db', linewidth=2)

    # Draw safe zone threshold line
    ax.axhline(y=safe_zone, color='red', linestyle='--', linewidth=2,
              label=f'Safe Zone Upper ({safe_zone:.1f})')

    # Highlight violation area
    # Vector operation: boolean masking to select rows where metric exceeds safe zone
    violations = y_data > safe_zone
    if violations.any():
        # Fill area where violations occur
        ax.fill_between(phase2_df['elapsed_seconds'], y_data, safe_zone,
                        where=violations, alpha=0.3, color='red',
                        label='Violation Zone')

    ax.set_xlabel('Elapsed Time (seconds)', fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(f'Phase 2: {label} vs Safe Zone', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Phase 2 (Throughput Stress): Safe Zone Violations',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()
# -

# ### Phase 2: Message Rate Analysis at Violation Points
#
# Analyze the message rate when E2E Latency P95 first violates the safe zone.

# +
# Analyze E2E Latency P95 violations
metric = 'e2e_latency_p95'
violations_p2 = identify_violations(phase2_df, metric, safe_zones[metric])

if len(violations_p2) > 0:
    # Get first violation
    first_violation = violations_p2.iloc[0]
    first_violation_time = first_violation['elapsed_seconds']
    first_violation_value = first_violation[metric]

    # Calculate message rate at violation time
    msg_rate_at_violation = get_message_rate_phase2(first_violation_time)

    print("E2E Latency P95 Safe Zone Violation Analysis")
    print("=" * 60)
    print(f"Safe Zone Upper Bound: {safe_zones[metric]:.2f} ms")
    print(f"First Violation Time: {first_violation_time:.1f}s")
    print(f"First Violation Value: {first_violation_value:.2f} ms")
    print(f"Message Rate at Violation: {msg_rate_at_violation} msg/s")
    print(f"Total Violations: {len(violations_p2)}")
    print(f"\nConclusion: System performance degrades when message rate exceeds {msg_rate_at_violation} msg/s")
else:
    print("No safe zone violations detected for E2E Latency P95")
# -

# ### Phase 2: Leading Indicator Lagged Correlation Analysis
#
# Find metrics that change before error_rate increases, enabling predictive monitoring.

# +
# Phase 2 leading indicator analysis
target_metric_p2 = 'error_rate'

# Candidate metrics (exclude error_rate itself)
# List comprehension: filter numeric columns, exclude target and timestamp columns
candidate_metrics_p2 = [
    col for col in phase2_df.select_dtypes(include=[np.number]).columns
    if col not in ['error_rate', 'failure_count', 'elapsed_seconds', 'timestamp']
]

# Calculate lagged correlations
corr_matrix_p2 = calculate_lagged_correlations(
    phase2_df, target_metric_p2, candidate_metrics_p2,
    lags=[0, 5, 10, 15, 20, 30]
)

# Display top 10 metrics with strongest correlation at any lag
print("Phase 2: Top 10 Metrics with Lagged Correlation to Error Rate")
print("=" * 80)
print(corr_matrix_p2.head(10).round(3))

# Find top leading indicators
top_indicators_p2 = find_top_leading_indicators(corr_matrix_p2, top_n=3)

print(f"\n\nTop 3 Leading Indicators for Error Rate:")
print("=" * 80)
for i, (metric, lag, corr) in enumerate(top_indicators_p2, 1):
    print(f"{i}. {metric:30s} | Optimal Lag: {lag:2d}s | Correlation: {corr:+.3f}")

# Visualize correlation heatmap
plot_correlation_heatmap(corr_matrix_p2.head(15),
                        title="Phase 2: Lagged Correlation with Error Rate (Top 15 Metrics)")
# -

# ### Phase 2: Correlation Analysis
#

# +
# Correlation analysis between message rate and core metrics
correlation_metrics = ['message_rate', 'e2e_latency_p95', 'eventloop_lag_p95',
                      'memory_bytes', 'error_rate']

# Filter to metrics that exist in the DataFrame
available_metrics = [m for m in correlation_metrics if m in phase2_df.columns]

# Calculate correlation matrix
# Vector operation: corr() computes pairwise correlations for all column combinations
corr_matrix = phase2_df[available_metrics].corr()

print("Phase 2: Correlation Matrix")
print("=" * 80)
print(corr_matrix.round(3))

# Visualize correlation matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
           center=0, vmin=-1, vmax=1, square=True,
           cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Phase 2: Metric Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
# -

# ### Phase 2: Strong Correlation Visualization
#
# Visualize metric pairs with correlation coefficient >= 0.7 to identify metrics that move together.

# +
# Find strongly correlated metric pairs (|correlation| >= 0.7)
strong_corrs_p2 = []
for i in range(len(available_metrics)):
    for j in range(i+1, len(available_metrics)):
        metric1 = available_metrics[i]
        metric2 = available_metrics[j]
        corr_val = corr_matrix.loc[metric1, metric2]

        # Check if absolute correlation exceeds threshold
        if abs(corr_val) >= 0.7:
            strong_corrs_p2.append((metric1, metric2, corr_val))

if strong_corrs_p2:
    print(f"Found {len(strong_corrs_p2)} strongly correlated pairs (|r| >= 0.7)")
    print("=" * 80)

    # Plot all strong correlations on same figure
    fig, axes = plt.subplots(len(strong_corrs_p2), 1,
                            figsize=(14, 5*len(strong_corrs_p2)))

    # Ensure axes is always iterable
    if len(strong_corrs_p2) == 1:
        axes = [axes]

    for idx, (m1, m2, corr) in enumerate(strong_corrs_p2):
        ax = axes[idx]

        # Create dual y-axis plot
        ax2 = ax.twinx()

        # Plot both metrics
        line1 = ax.plot(phase2_df['elapsed_seconds'], phase2_df[m1],
                       'b-', label=m1, linewidth=2)
        line2 = ax2.plot(phase2_df['elapsed_seconds'], phase2_df[m2],
                        'r-', label=m2, linewidth=2)

        # Labels and title
        ax.set_xlabel('Elapsed Time (seconds)', fontsize=11)
        ax.set_ylabel(m1, color='b', fontsize=11)
        ax2.set_ylabel(m2, color='r', fontsize=11)
        ax.set_title(f'{m1} vs {m2} (correlation: {corr:+.3f})',
                    fontsize=12, fontweight='bold')

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')

        ax.grid(True, alpha=0.3)

        print(f"{idx+1}. {m1:30s} <-> {m2:30s} | r = {corr:+.3f}")

    plt.tight_layout()
    plt.show()
else:
    print("No strongly correlated pairs found (threshold: |r| >= 0.7)")
# -

# ## 6. Phase 3: Bandwidth Stress Analysis
#
# ### Scenario
# - Message size escalation every 2 minutes: 10 → 50 → 100 → 500 → 1000 KB
# - Objective: Identify message size threshold where bandwidth becomes bottleneck

# +
phase3_df = phase_data['phase3_bandwidth']

# Visualize safe zone violations
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, (metric, label) in enumerate(CORE_METRICS):
    if metric not in phase3_df.columns:
        continue

    ax = axes[idx]

    # Convert memory to MB if needed
    y_data = phase3_df[metric] / (1024*1024) if metric == 'memory_bytes' else phase3_df[metric]
    safe_zone = safe_zones[metric] / (1024*1024) if metric == 'memory_bytes' else safe_zones[metric]

    # Plot metric over time
    ax.plot(phase3_df['elapsed_seconds'], y_data,
           label=f'{label}', color='#f39c12', linewidth=2)

    # Draw safe zone threshold
    ax.axhline(y=safe_zone, color='red', linestyle='--', linewidth=2,
              label=f'Safe Zone Upper ({safe_zone:.1f})')

    # Highlight violations
    violations = y_data > safe_zone
    if violations.any():
        ax.fill_between(phase3_df['elapsed_seconds'], y_data, safe_zone,
                        where=violations, alpha=0.3, color='red',
                        label='Violation Zone')

    ax.set_xlabel('Elapsed Time (seconds)', fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(f'Phase 3: {label} vs Safe Zone', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Phase 3 (Bandwidth Stress): Safe Zone Violations',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()
# -

# ### Phase 3: Message Size Analysis at Violation Points
#
# Analyze message size when E2E Latency P95, Memory Usage, and Error Rate violate safe zones.

# +
# Analyze multiple metric violations for Phase 3
metrics_to_check = [
    ('e2e_latency_p95', 'E2E Latency P95'),
    ('memory_bytes', 'Memory Usage'),
    ('error_rate', 'Error Rate')
]

print("Phase 3: Multi-Metric Violation Analysis")
print("=" * 80)

for metric, label in metrics_to_check:
    if metric not in phase3_df.columns or metric not in safe_zones:
        continue

    violations = identify_violations(phase3_df, metric, safe_zones[metric])

    if len(violations) > 0:
        first_violation = violations.iloc[0]
        violation_time = first_violation['elapsed_seconds']
        violation_value = first_violation[metric]
        msg_size = get_message_size_phase3(violation_time)

        # Convert memory to MB for display
        if metric == 'memory_bytes':
            violation_value_display = violation_value / (1024*1024)
            safe_zone_display = safe_zones[metric] / (1024*1024)
            unit = 'MB'
        else:
            violation_value_display = violation_value
            safe_zone_display = safe_zones[metric]
            unit = 'ms' if 'latency' in metric else '%' if 'rate' in metric else ''

        print(f"\n{label}:")
        print(f"  Safe Zone Upper: {safe_zone_display:.2f} {unit}")
        print(f"  First Violation: {violation_time:.1f}s at {violation_value_display:.2f} {unit}")
        print(f"  Message Size: {msg_size} KB")
        print(f"  Total Violations: {len(violations)}")
    else:
        print(f"\n{label}: No violations detected")
# -

# ### Phase 3: Leading Indicator Lagged Correlation Analysis
#
# Find metrics that change before error_rate increases under bandwidth stress.

# +
# Phase 3 leading indicator analysis
target_metric_p3 = 'error_rate'

candidate_metrics_p3 = [
    col for col in phase3_df.select_dtypes(include=[np.number]).columns
    if col not in ['error_rate', 'failure_count', 'elapsed_seconds', 'timestamp']
]

# Calculate lagged correlations
corr_matrix_p3 = calculate_lagged_correlations(
    phase3_df, target_metric_p3, candidate_metrics_p3,
    lags=[0, 5, 10, 15, 20, 30]
)

print("Phase 3: Top 10 Metrics with Lagged Correlation to Error Rate")
print("=" * 80)
print(corr_matrix_p3.head(10).round(3))

# Find top leading indicators
top_indicators_p3 = find_top_leading_indicators(corr_matrix_p3, top_n=3)

print(f"\n\nTop 3 Leading Indicators for Error Rate:")
print("=" * 80)
for i, (metric, lag, corr) in enumerate(top_indicators_p3, 1):
    print(f"{i}. {metric:30s} | Optimal Lag: {lag:2d}s | Correlation: {corr:+.3f}")

# Visualize correlation heatmap
plot_correlation_heatmap(corr_matrix_p3.head(15),
                        title="Phase 3: Lagged Correlation with Error Rate (Top 15 Metrics)")

# +
# Load Phase 3 Locust stats_history data
import glob
from datetime import datetime

phase3_stats_files = glob.glob(str(PHASE_PATHS['phase3_bandwidth'] / '*_stats_history.csv'))

if phase3_stats_files:
    print(f"Found {len(phase3_stats_files)} stats files")

    # Load and combine all stats files
    stats_dfs = []
    for f in phase3_stats_files:
        df = pd.read_csv(f)
        stats_dfs.append(df)

    # Concatenate all DataFrames
    # Vector operation: concat combines multiple DataFrames efficiently
    phase3_stats = pd.concat(stats_dfs, ignore_index=True)

    # Parse timestamp
    phase3_stats['Timestamp'] = pd.to_datetime(phase3_stats['Timestamp'], unit='s')

    print(f"Loaded {len(phase3_stats)} stats records")
    print(f"Columns: {list(phase3_stats.columns)}")
else:
    print("No stats_history files found")

# +
# failure로 그래프 그리기
# Phase 3 Locust stats_history 데이터 로드
import glob
from datetime import datetime

phase3_stats_files = glob.glob(str(PHASE_PATHS['phase3_bandwidth'] / '*_stats_history.csv'))

# 모든 stats_history 파일 로드 및 병합 (벡터 연산)
stats_dfs = []
for file in phase3_stats_files:
  df = pd.read_csv(file)
  # Timestamp를 datetime으로 변환 (벡터 연산)
  df['timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
  stats_dfs.append(df)

# 모든 DataFrame 병합 (벡터 연산: concat)
stats_combined = pd.concat(stats_dfs, ignore_index=True)

# 시간순 정렬 (벡터 연산: sort_values)
stats_combined = stats_combined.sort_values('timestamp').reset_index(drop=True)

# 경과 시간 계산 (벡터 연산)
stats_combined['elapsed_seconds'] = (stats_combined['timestamp'] -
                                   stats_combined['timestamp'].min()).dt.total_seconds()

print(f"✓ 통합 데이터: {len(stats_combined)}개 샘플")
print(f"  • 수집 기간: {stats_combined['elapsed_seconds'].max():.1f}초")
print(f"  • Failures/s 범위: [{stats_combined['Failures/s'].min():.2f}, "
    f"{stats_combined['Failures/s'].max():.2f}]")
print(f"  • Total Failure Count: {stats_combined['Total Failure Count'].max():.0f}건")

# Failures/s가 0보다 큰 샘플만 확인
failure_samples = stats_combined[stats_combined['Failures/s'] > 0]
print(f"  • 실패 발생 샘플: {len(failure_samples)}개 ({len(failure_samples)/len(stats_combined)*100:.1f}%)")

# 디버깅: 타임스탬프 범위 확인
print(f"\n🔍 타임스탬프 범위 확인:")
print(f"  Stats History:")
print(f"    시작: {stats_combined['timestamp'].min()}")
print(f"    종료: {stats_combined['timestamp'].max()}")
print(f"  Metrics (phase3_df):")
print(f"    시작: {phase3_df['timestamp'].min()}")
print(f"    종료: {phase3_df['timestamp'].max()}")

# 방법 1: elapsed_seconds 기반 병합 시도 (더 안정적)
print(f"\n🔄 elapsed_seconds 기반 병합 시도...")

# Phase3_df의 elapsed_seconds 재계산
phase3_df_copy = phase3_df.copy()
phase3_df_copy['elapsed_seconds'] = (phase3_df_copy['timestamp'] -
                                   phase3_df_copy['timestamp'].min()).dt.total_seconds()

# elapsed_seconds 기반 병합 (벡터 연산: merge_asof)
phase3_with_failures = pd.merge_asof(
  stats_combined[['elapsed_seconds', 'Failures/s', 'Total Failure Count', 'User Count']].sort_values('elapsed_seconds'),
  phase3_df_copy.sort_values('elapsed_seconds'),
  on='elapsed_seconds',
  direction='nearest',
  tolerance=5  # 30초 tolerance
)

# Failures/s와 최소 하나의 메트릭이 있는 행만 유지
initial_len = len(phase3_with_failures)
phase3_with_failures = phase3_with_failures[
  phase3_with_failures['Failures/s'].notna()
]

print(f"✓ Metrics와 병합 완료: {len(phase3_with_failures)}개 샘플 (초기: {initial_len}개)")

# Failures/s를 타겟으로 선행 지표 분석
if len(phase3_with_failures) > 0 and phase3_with_failures['Failures/s'].sum() > 0:
  target_metric_failures = 'Failures/s'

  # 후보 메트릭 (Failures/s 제외)
  candidate_metrics_failures = [
      'e2e_latency_p95', 'e2e_latency_p99', 'eventloop_lag_p95', 'eventloop_lag_p99',
      'redis_latency_p95', 'redis_backlog', 'broadcast_queue', 'memory_bytes',
      'cpu', 'message_rate', 'connected_users', 'error_rate'
  ]

  # 존재하는 메트릭만 선택
  candidate_metrics_failures = [m for m in candidate_metrics_failures
                                if m in phase3_with_failures.columns]

  print(f"\n" + "=" * 80)
  print(f"🔍 Failures/s 예측을 위한 선행 지표 분석")
  print("=" * 80)
  print(f"사용 가능한 후보 메트릭: {len(candidate_metrics_failures)}개")

  # 시차 상관관계 계산
  lags = [0, 5, 10, 15, 20, 30]
  corr_matrix_failures = calculate_lagged_correlations(
      phase3_with_failures, target_metric_failures, candidate_metrics_failures, lags
  )

  if corr_matrix_failures is not None and not corr_matrix_failures.empty:
      # 1. 상관관계 히트맵
      plt.figure(figsize=(14, 8))
      sns.heatmap(corr_matrix_failures.T, annot=True, fmt='.3f', cmap='RdYlGn',
                 center=0, vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
      plt.title(f'Phase 3: Failures/s 시차 상관관계 히트맵\n(Locust Stats History)',
               fontsize=14, fontweight='bold', pad=20)
      plt.xlabel('Lag Time (초)', fontsize=12)
      plt.ylabel('Metrics', fontsize=12)
      plt.tight_layout()
      plt.show()

      # 2. Top 3 예측 지표
      top_predictors_failures = find_top_predictors(corr_matrix_failures, top_n=3)

      print(f"\n✨ Top 3 선행 지표 (Failures/s 예측):")
      for i, pred in enumerate(top_predictors_failures, 1):
          direction = "양의" if pred['correlation'] > 0 else "음의"
          print(f"{i}. {pred['metric']}")
          print(f"   • 최적 선행 시간: {pred['lag']}초")
          print(f"   • 상관계수: {pred['correlation']:.3f} ({direction} 상관)")
          print(f"   💡 인사이트: {pred['metric']}이(가) {pred['lag']}초 전에 변화하면 "
                f"테스트 실패 가능성 높음\n")

      # 3. Top 1 예측 지표 + Failures/s 시각화
      if len(top_predictors_failures) > 0:
          top1 = top_predictors_failures[0]

          fig, ax1 = plt.subplots(figsize=(16, 6))
          ax2 = ax1.twinx()

          # 선행 지표 (시프트 적용)
          shifted_predictor = phase3_with_failures[top1['metric']].shift(top1['lag'])

          values1 = shifted_predictor / 1_000_000 if 'memory' in top1['metric'] else shifted_predictor
          line1 = ax1.plot(phase3_with_failures['elapsed_seconds'], values1,
                         color='#f39c12', linewidth=2,
                         label=f"{top1['metric']} (shifted -{top1['lag']}s)",
                         marker='o', markersize=4, alpha=0.7)
          ylabel1 = f"{top1['metric']} (MB)" if 'memory' in top1['metric'] else top1['metric']
          ax1.set_ylabel(ylabel1, fontsize=12, color='#f39c12')
          ax1.tick_params(axis='y', labelcolor='#f39c12')

          # Failures/s
          line2 = ax2.plot(phase3_with_failures['elapsed_seconds'],
                         phase3_with_failures[target_metric_failures],
                         color='#e74c3c', linewidth=2, label='Failures/s',
                         marker='s', markersize=4, alpha=0.7)
          ax2.set_ylabel('Failures/s', fontsize=12, color='#e74c3c')
          ax2.tick_params(axis='y', labelcolor='#e74c3c')

          ax1.set_xlabel('경과 시간 (초)', fontsize=12)
          ax1.set_title(f'Phase 3: Top 1 선행 지표 ({top1["metric"]}) vs Failures/s\n'
                       f'선행 시간: {top1["lag"]}초, 상관계수: {top1["correlation"]:.3f}',
                       fontsize=14, fontweight='bold', pad=20)

          lines = line1 + line2
          labels = [l.get_label() for l in lines]
          ax1.legend(lines, labels, loc='upper left', fontsize=10)
          ax1.grid(True, alpha=0.3)

          plt.tight_layout()
          plt.show()

      print("\n" + "=" * 80)
      print("💡 벡터 연산 사용:")
      print("   • pd.concat() - 다중 DataFrame 병합 (O(n))")
      print("   • pd.merge_asof() - 시계열 기반 nearest join (O(n log n))")
      print("   • elapsed_seconds 기반 병합으로 timestamp 형식 차이 해결")
      print("   • df[col].shift() - 시계열 시프트 (O(n))")
      print("=" * 80)
  else:
      print("⚠️  상관관계 계산 실패")
else:
  print("\n⚠️  Failures/s 데이터 없음 또는 병합 실패")

# -

# ### Phase 3: Correlation Analysis
#

# +
correlation_metrics_p3 = ['message_rate', 'e2e_latency_p95', 'eventloop_lag_p95',
                          'memory_bytes', 'error_rate']

available_metrics_p3 = [m for m in correlation_metrics_p3 if m in phase3_df.columns]

# Calculate correlation matrix
corr_matrix_metrics_p3 = phase3_df[available_metrics_p3].corr()

print("Phase 3: Correlation Matrix")
print("=" * 80)
print(corr_matrix_metrics_p3.round(3))

# Visualize
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_metrics_p3, annot=True, fmt='.2f', cmap='RdBu_r',
           center=0, vmin=-1, vmax=1, square=True,
           cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Phase 3: Metric Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
# -

# ### Phase 3: Strong Correlation Visualization
#

# +
# Find strongly correlated pairs
strong_corrs_p3 = []
for i in range(len(available_metrics_p3)):
    for j in range(i+1, len(available_metrics_p3)):
        metric1 = available_metrics_p3[i]
        metric2 = available_metrics_p3[j]
        corr_val = corr_matrix_metrics_p3.loc[metric1, metric2]

        if abs(corr_val) >= 0.7:
            strong_corrs_p3.append((metric1, metric2, corr_val))

if strong_corrs_p3:
    print(f"Found {len(strong_corrs_p3)} strongly correlated pairs (|r| >= 0.7)")
    print("=" * 80)

    fig, axes = plt.subplots(len(strong_corrs_p3), 1,
                            figsize=(14, 5*len(strong_corrs_p3)))

    if len(strong_corrs_p3) == 1:
        axes = [axes]

    for idx, (m1, m2, corr) in enumerate(strong_corrs_p3):
        ax = axes[idx]
        ax2 = ax.twinx()

        line1 = ax.plot(phase3_df['elapsed_seconds'], phase3_df[m1],
                       'b-', label=m1, linewidth=2)
        line2 = ax2.plot(phase3_df['elapsed_seconds'], phase3_df[m2],
                        'r-', label=m2, linewidth=2)

        ax.set_xlabel('Elapsed Time (seconds)', fontsize=11)
        ax.set_ylabel(m1, color='b', fontsize=11)
        ax2.set_ylabel(m2, color='r', fontsize=11)
        ax.set_title(f'{m1} vs {m2} (correlation: {corr:+.3f})',
                    fontsize=12, fontweight='bold')

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')

        ax.grid(True, alpha=0.3)

        print(f"{idx+1}. {m1:30s} <-> {m2:30s} | r = {corr:+.3f}")

    plt.tight_layout()
    plt.show()
else:
    print("No strongly correlated pairs found")
# -

# ## 7. Phase 4: Connection Stress Analysis
#
# ### Scenario
# - Concurrent user escalation every 2 minutes: 100 → 200 → 300 → 400 → 500 → 600
# - Objective: Identify connection limit where system scalability degrades

# +
phase4_df = phase_data['phase4_connection']

# Visualize safe zone violations
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, (metric, label) in enumerate(CORE_METRICS):
    if metric not in phase4_df.columns:
        continue

    ax = axes[idx]

    # Convert memory to MB if needed
    y_data = phase4_df[metric] / (1024*1024) if metric == 'memory_bytes' else phase4_df[metric]
    safe_zone = safe_zones[metric] / (1024*1024) if metric == 'memory_bytes' else safe_zones[metric]

    # Plot metric
    ax.plot(phase4_df['elapsed_seconds'], y_data,
           label=f'{label}', color='#e74c3c', linewidth=2)

    # Safe zone threshold
    ax.axhline(y=safe_zone, color='red', linestyle='--', linewidth=2,
              label=f'Safe Zone Upper ({safe_zone:.1f})')

    # Highlight violations
    violations = y_data > safe_zone
    if violations.any():
        ax.fill_between(phase4_df['elapsed_seconds'], y_data, safe_zone,
                        where=violations, alpha=0.3, color='red',
                        label='Violation Zone')

    ax.set_xlabel('Elapsed Time (seconds)', fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(f'Phase 4: {label} vs Safe Zone', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Phase 4 (Connection Stress): Safe Zone Violations',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()
# -

# ### Phase 4: User Count Analysis at Violation Points
#
# Analyze concurrent user count when E2E Latency P95 and Event Loop Lag P95 violate safe zones.

# +
# Analyze multiple metric violations for Phase 4
metrics_to_check_p4 = [
    ('e2e_latency_p95', 'E2E Latency P95'),
    ('eventloop_lag_p95', 'Event Loop Lag P95')
]

print("Phase 4: Multi-Metric Violation Analysis")
print("=" * 80)

for metric, label in metrics_to_check_p4:
    if metric not in phase4_df.columns or metric not in safe_zones:
        continue

    violations = identify_violations(phase4_df, metric, safe_zones[metric])

    if len(violations) > 0:
        first_violation = violations.iloc[0]
        violation_time = first_violation['elapsed_seconds']
        violation_value = first_violation[metric]
        user_count = get_user_count_phase4(violation_time)

        print(f"\n{label}:")
        print(f"  Safe Zone Upper: {safe_zones[metric]:.2f} ms")
        print(f"  First Violation: {violation_time:.1f}s at {violation_value:.2f} ms")
        print(f"  Concurrent Users: {user_count}")
        print(f"  Total Violations: {len(violations)}")
    else:
        print(f"\n{label}: No violations detected")
# -

# ### Phase 4: Leading Indicator Lagged Correlation Analysis
#
# Find leading indicators for connection stress. Since Phase 4 has minimal errors,
# we use e2e_latency_p95 as the target metric instead of error_rate.

# +
# Phase 4 leading indicator analysis
# Use e2e_latency_p95 as target since error_rate is minimal
target_metric_p4 = 'e2e_latency_p95'

candidate_metrics_p4 = [
    col for col in phase4_df.select_dtypes(include=[np.number]).columns
    if col not in ['e2e_latency_p95', 'elapsed_seconds', 'timestamp']
]

# Calculate lagged correlations
corr_matrix_p4 = calculate_lagged_correlations(
    phase4_df, target_metric_p4, candidate_metrics_p4,
    lags=[0, 5, 10, 15, 20, 30]
)

print("Phase 4: Top 10 Metrics with Lagged Correlation to E2E Latency P95")
print("=" * 80)
print(corr_matrix_p4.head(10).round(3))

# Find top leading indicators
top_indicators_p4 = find_top_leading_indicators(corr_matrix_p4, top_n=3)

print(f"\n\nTop 3 Leading Indicators for E2E Latency P95:")
print("=" * 80)
for i, (metric, lag, corr) in enumerate(top_indicators_p4, 1):
    print(f"{i}. {metric:30s} | Optimal Lag: {lag:2d}s | Correlation: {corr:+.3f}")

# Visualize correlation heatmap
plot_correlation_heatmap(corr_matrix_p4.head(15),
                        title="Phase 4: Lagged Correlation with E2E Latency P95 (Top 15 Metrics)")
# -

# ### Phase 4: Correlation Analysis
#

# +
correlation_metrics_p4 = ['connected_users', 'e2e_latency_p95', 'eventloop_lag_p95',
                          'memory_bytes', 'ws_send_latency_p95']

available_metrics_p4 = [m for m in correlation_metrics_p4 if m in phase4_df.columns]

# Calculate correlation matrix
corr_matrix_metrics_p4 = phase4_df[available_metrics_p4].corr()

# Visualize
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_metrics_p4, annot=True, fmt='.2f', cmap='RdBu_r',
           center=0, vmin=-1, vmax=1, square=True,
           cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Phase 4: Metric Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
# -

# ### Phase 4: Strong Correlation Visualization
#

# +
# Find strongly correlated pairs
strong_corrs_p4 = []
for i in range(len(available_metrics_p4)):
    for j in range(i+1, len(available_metrics_p4)):
        metric1 = available_metrics_p4[i]
        metric2 = available_metrics_p4[j]
        corr_val = corr_matrix_metrics_p4.loc[metric1, metric2]

        if abs(corr_val) >= 0.7:
            strong_corrs_p4.append((metric1, metric2, corr_val))

if strong_corrs_p4:
    print(f"Found {len(strong_corrs_p4)} strongly correlated pairs (|r| >= 0.7)")
    print("=" * 80)

    fig, axes = plt.subplots(len(strong_corrs_p4), 1,
                            figsize=(14, 5*len(strong_corrs_p4)))

    if len(strong_corrs_p4) == 1:
        axes = [axes]

    for idx, (m1, m2, corr) in enumerate(strong_corrs_p4):
        ax = axes[idx]
        ax2 = ax.twinx()

        line1 = ax.plot(phase4_df['elapsed_seconds'], phase4_df[m1],
                       'b-', label=m1, linewidth=2)
        line2 = ax2.plot(phase4_df['elapsed_seconds'], phase4_df[m2],
                        'r-', label=m2, linewidth=2)

        ax.set_xlabel('Elapsed Time (seconds)', fontsize=11)
        ax.set_ylabel(m1, color='b', fontsize=11)
        ax2.set_ylabel(m2, color='r', fontsize=11)
        ax.set_title(f'{m1} vs {m2} (correlation: {corr:+.3f})',
                    fontsize=12, fontweight='bold')

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')

        ax.grid(True, alpha=0.3)

        print(f"{idx+1}. {m1:30s} <-> {m2:30s} | r = {corr:+.3f}")

    plt.tight_layout()
    plt.show()
else:
    print("No strongly correlated pairs found")
