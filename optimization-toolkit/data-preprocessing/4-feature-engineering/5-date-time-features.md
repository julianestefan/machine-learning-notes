# Date and Time Feature Engineering

Date and time features can provide valuable information for many predictive modeling tasks. This guide covers techniques for extracting and transforming temporal information into useful features.

## Initial Date/Time Conversion

First, ensure date and time data is in the proper format:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataframe with date strings
df = pd.DataFrame({
    'date_string': ['2022-01-15', '2022-02-20', '2022-03-25', '2022-04-30'],
    'datetime_string': ['2022-01-15 08:30:00', '2022-02-20 14:45:00', 
                         '2022-03-25 23:15:00', '2022-04-30 10:00:00']
})

# Convert string columns to datetime
df['date'] = pd.to_datetime(df['date_string'])
df['datetime'] = pd.to_datetime(df['datetime_string'])

print(df[['date', 'datetime']].dtypes)
```

## Handling Different Date Formats

For data with mixed date formats:

```python
# Mixed date formats
mixed_dates = pd.DataFrame({
    'date_mixed': ['Jan 15, 2022', '02/20/2022', '2022-03-25', '30-04-2022']
})

# Convert with format inference
mixed_dates['date_parsed'] = pd.to_datetime(mixed_dates['date_mixed'], 
                                            infer_datetime_format=True)
print(mixed_dates)

# For specific format
specific_format = pd.DataFrame({
    'date_specific': ['15/01/2022', '20/02/2022']  # DD/MM/YYYY
})

# Convert with explicit format
specific_format['date_parsed'] = pd.to_datetime(specific_format['date_specific'], 
                                               format='%d/%m/%Y')
print(specific_format)
```

## Extracting Component Features

Extract standard components from datetime:

```python
# Create a more extensive dataset
date_range = pd.date_range(start='2021-01-01', end='2022-12-31', freq='D')
ts_df = pd.DataFrame({'date': date_range})

# Basic components
ts_df['year'] = ts_df['date'].dt.year
ts_df['quarter'] = ts_df['date'].dt.quarter
ts_df['month'] = ts_df['date'].dt.month
ts_df['day'] = ts_df['date'].dt.day
ts_df['dayofweek'] = ts_df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
ts_df['dayofyear'] = ts_df['date'].dt.dayofyear  # 1 to 365/366
ts_df['week'] = ts_df['date'].dt.isocalendar().week
ts_df['is_month_end'] = ts_df['date'].dt.is_month_end.astype(int)
ts_df['is_month_start'] = ts_df['date'].dt.is_month_start.astype(int)
ts_df['is_quarter_end'] = ts_df['date'].dt.is_quarter_end.astype(int)
ts_df['is_quarter_start'] = ts_df['date'].dt.is_quarter_start.astype(int)
ts_df['is_year_end'] = ts_df['date'].dt.is_year_end.astype(int)
ts_df['is_year_start'] = ts_df['date'].dt.is_year_start.astype(int)

# Sample of the dataframe
print(ts_df.head())

# For time components in datetime objects
time_df = pd.DataFrame({
    'datetime': pd.date_range(start='2022-01-01', end='2022-01-02', freq='H')
})

time_df['hour'] = time_df['datetime'].dt.hour
time_df['minute'] = time_df['datetime'].dt.minute
time_df['second'] = time_df['datetime'].dt.second
time_df['is_am'] = (time_df['hour'] < 12).astype(int)
time_df['is_business_hour'] = ((time_df['hour'] >= 9) & (time_df['hour'] < 17)).astype(int)

print(time_df.head())
```

## Creating Cyclical Features

For features with cyclical patterns (day of week, month, hour, etc.):

```python
# Create cyclical features for month
ts_df['month_sin'] = np.sin(2 * np.pi * ts_df['month'] / 12)
ts_df['month_cos'] = np.cos(2 * np.pi * ts_df['month'] / 12)

# Create cyclical features for day of week
ts_df['dayofweek_sin'] = np.sin(2 * np.pi * ts_df['dayofweek'] / 7)
ts_df['dayofweek_cos'] = np.cos(2 * np.pi * ts_df['dayofweek'] / 7)

# Create cyclical features for hour (for datetime data)
if 'hour' in time_df.columns:
    time_df['hour_sin'] = np.sin(2 * np.pi * time_df['hour'] / 24)
    time_df['hour_cos'] = np.cos(2 * np.pi * time_df['hour'] / 24)

# Visualize the cyclical nature of month feature
plt.figure(figsize=(10, 8))
plt.scatter(ts_df['month_cos'], ts_df['month_sin'], c=ts_df['month'], cmap='viridis')
plt.title('Cyclical Encoding of Month')
plt.xlabel('Cosine Component')
plt.ylabel('Sine Component')
plt.colorbar(label='Month')
plt.grid(True)
plt.axis('equal')
plt.show()

# Compare standard vs cyclical features in modeling
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random

# Create synthetic target influenced by month (with peak in summer)
np.random.seed(42)
baseline = 10  # Baseline value
seasonal_effect = 5 * np.sin(2 * np.pi * (ts_df['month'] - 6) / 12)  # Peak at month 6 (June)
noise = np.random.normal(0, 1, len(ts_df))  # Random noise
ts_df['target'] = baseline + seasonal_effect + noise

# Split into train/test
train_idx = ts_df['date'] < '2022-07-01'
test_idx = ~train_idx
X_train_standard = ts_df.loc[train_idx, ['month']]
X_train_cyclical = ts_df.loc[train_idx, ['month_sin', 'month_cos']]
y_train = ts_df.loc[train_idx, 'target']
X_test_standard = ts_df.loc[test_idx, ['month']]
X_test_cyclical = ts_df.loc[test_idx, ['month_sin', 'month_cos']]
y_test = ts_df.loc[test_idx, 'target']

# Train models
model_standard = LinearRegression().fit(X_train_standard, y_train)
model_cyclical = LinearRegression().fit(X_train_cyclical, y_train)

# Evaluate
mse_standard = mean_squared_error(y_test, model_standard.predict(X_test_standard))
mse_cyclical = mean_squared_error(y_test, model_cyclical.predict(X_test_cyclical))

print(f"MSE with standard month feature: {mse_standard:.4f}")
print(f"MSE with cyclical month features: {mse_cyclical:.4f}")
```

## Business Day Features

Features based on business calendar:

```python
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# Create business day features
ts_df['is_weekend'] = ts_df['dayofweek'].isin([5, 6]).astype(int)
ts_df['is_weekday'] = (~ts_df['dayofweek'].isin([5, 6])).astype(int)

# Create U.S. holiday calendar
us_cal = USFederalHolidayCalendar()
holidays = us_cal.holidays(start=ts_df['date'].min(), end=ts_df['date'].max())
ts_df['is_holiday'] = ts_df['date'].isin(holidays).astype(int)

# Business day of month/week
bday = CustomBusinessDay(calendar=us_cal)
ts_df['biz_day_of_month'] = ts_df.apply(
    lambda row: (
        row['date'].replace(day=1).bday.onOffset and 1 or
        np.busday_count(
            row['date'].replace(day=1).date().strftime('%Y-%m-%d'),
            row['date'].date().strftime('%Y-%m-%d'),
            holidays=holidays.date
        ) + 1
    ),
    axis=1
)

# Sample of business day features
print(ts_df[['date', 'is_weekend', 'is_holiday', 'biz_day_of_month']].head(10))
```

## Time-Based Features

For time series with irregular intervals:

```python
# Create sample time series with irregular intervals
timestamps = pd.to_datetime([
    '2022-01-01 08:30:00', '2022-01-01 09:15:00', '2022-01-01 11:45:00',
    '2022-01-02 14:20:00', '2022-01-03 10:00:00', '2022-01-05 16:30:00'
])
irregular_ts = pd.DataFrame({'timestamp': timestamps, 'value': range(len(timestamps))})

# Sort by timestamp to ensure chronological order
irregular_ts = irregular_ts.sort_values('timestamp').reset_index(drop=True)

# Time since start
irregular_ts['seconds_since_start'] = (
    irregular_ts['timestamp'] - irregular_ts['timestamp'].min()
).dt.total_seconds()

# Time since previous event
irregular_ts['seconds_since_prev'] = (
    irregular_ts['timestamp'] - irregular_ts['timestamp'].shift(1)
).dt.total_seconds()

# Fill NaN for first row
irregular_ts['seconds_since_prev'] = irregular_ts['seconds_since_prev'].fillna(0)

# Time until next event
irregular_ts['seconds_until_next'] = (
    irregular_ts['timestamp'].shift(-1) - irregular_ts['timestamp']
).dt.total_seconds()

# Fill NaN for last row
irregular_ts['seconds_until_next'] = irregular_ts['seconds_until_next'].fillna(
    irregular_ts['seconds_until_next'].mean()
)

print(irregular_ts)
```

## Lag and Rolling Window Features

For time series analysis:

```python
# Create a sample daily time series
ts = pd.DataFrame({
    'date': pd.date_range(start='2022-01-01', end='2022-03-31', freq='D'),
    'value': np.random.normal(10, 2, 90) + np.sin(np.linspace(0, 6*np.pi, 90))
})

# Set date as index for easier manipulation
ts.set_index('date', inplace=True)

# Create lag features (previous days)
for lag in [1, 7, 14, 28]:
    ts[f'lag_{lag}d'] = ts['value'].shift(lag)

# Create rolling window statistics
for window in [7, 14, 30]:
    ts[f'rolling_{window}d_mean'] = ts['value'].rolling(window=window).mean()
    ts[f'rolling_{window}d_std'] = ts['value'].rolling(window=window).std()
    ts[f'rolling_{window}d_min'] = ts['value'].rolling(window=window).min()
    ts[f'rolling_{window}d_max'] = ts['value'].rolling(window=window).max()

# Difference features (change from previous periods)
ts['diff_1d'] = ts['value'].diff(1)  # 1-day difference
ts['diff_7d'] = ts['value'].diff(7)  # 7-day difference

# Percentage change
ts['pct_change_1d'] = ts['value'].pct_change(1)  # 1-day percentage change
ts['pct_change_7d'] = ts['value'].pct_change(7)  # 7-day percentage change

# Sample of lag and rolling features
print(ts.head(10))

# Visualize the original series and a rolling mean
plt.figure(figsize=(12, 6))
plt.plot(ts.index, ts['value'], label='Original')
plt.plot(ts.index, ts['rolling_7d_mean'], label='7-day Rolling Mean')
plt.plot(ts.index, ts['rolling_30d_mean'], label='30-day Rolling Mean')
plt.title('Time Series with Rolling Means')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

## Seasonality Features

Extracting seasonal patterns:

```python
# Create a multi-year dataset to demonstrate seasonality
long_ts = pd.DataFrame({
    'date': pd.date_range(start='2018-01-01', end='2022-12-31', freq='D')
})

# Add seasonal patterns
long_ts['year'] = long_ts['date'].dt.year
long_ts['month'] = long_ts['date'].dt.month
long_ts['day'] = long_ts['date'].dt.day
long_ts['dayofyear'] = long_ts['date'].dt.dayofyear

# Create seasonal indicators
long_ts['is_spring'] = long_ts['month'].isin([3, 4, 5]).astype(int)
long_ts['is_summer'] = long_ts['month'].isin([6, 7, 8]).astype(int)
long_ts['is_fall'] = long_ts['month'].isin([9, 10, 11]).astype(int)
long_ts['is_winter'] = long_ts['month'].isin([12, 1, 2]).astype(int)

# Month of quarter
long_ts['month_of_quarter'] = (long_ts['month'] - 1) % 3 + 1

# Week of month (approximate)
long_ts['week_of_month'] = (long_ts['day'] - 1) // 7 + 1

# Create synthetic seasonal data
base = 10
# Yearly cycle
yearly_pattern = 5 * np.sin(2 * np.pi * long_ts['dayofyear'] / 365)
# Weekly cycle
weekly_pattern = 2 * np.sin(2 * np.pi * long_ts['date'].dt.dayofweek / 7)
# Time trend
time_trend = 0.001 * np.arange(len(long_ts))
# Random noise
np.random.seed(42)
noise = np.random.normal(0, 1, len(long_ts))

# Combine components
long_ts['value'] = base + yearly_pattern + weekly_pattern + time_trend + noise

# Visualize the components
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(long_ts['date'], yearly_pattern)
plt.title('Yearly Seasonal Pattern')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(long_ts['date'][:28], weekly_pattern[:28])
plt.title('Weekly Seasonal Pattern (First 4 Weeks)')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(long_ts['date'], long_ts['value'])
plt.title('Combined Time Series')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Holiday Features

Creating features for holidays and special events:

```python
from pandas.tseries.holiday import USFederalHolidayCalendar, AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import Day, Easter

# Create a custom holiday calendar
class CustomHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1, observance=nearest_workday),
        Holiday('SuperBowl', month=2, day=12, year=2022),  # Fixed date for demo
        Holiday('Valentine', month=2, day=14),
        Holiday('Easter', month=1, day=1, offset=Easter()),
        Holiday('Mother Day', month=5, day=8, year=2022),  # 2nd Sunday in May for 2022
        Holiday('Father Day', month=6, day=19, year=2022),  # 3rd Sunday in June for 2022
        Holiday('Independence Day', month=7, day=4, observance=nearest_workday),
        Holiday('Thanksgiving', month=11, day=24, year=2022),  # 4th Thursday in November for 2022
        Holiday('Black Friday', month=11, day=25, year=2022),  # Day after Thanksgiving
        Holiday('Christmas', month=12, day=25, observance=nearest_workday),
        Holiday('New Year Eve', month=12, day=31)
    ]

# Create a calendar instance
custom_calendar = CustomHolidayCalendar()

# Get all holidays in date range
custom_holidays = custom_calendar.holidays(
    start=long_ts['date'].min(),
    end=long_ts['date'].max()
)

# Create features for days before and after holidays
date_range = pd.Series(long_ts['date'])
is_holiday = date_range.isin(custom_holidays).astype(int)
is_day_before_holiday = date_range.isin(custom_holidays - Day(1)).astype(int)
is_day_after_holiday = date_range.isin(custom_holidays + Day(1)).astype(int)

# Days until next holiday and days since last holiday
days_until_next_holiday = np.zeros(len(date_range))
days_since_last_holiday = np.zeros(len(date_range))

sorted_holidays = sorted(custom_holidays)
for i, date in enumerate(date_range):
    # Find days until next holiday
    next_holidays = [h for h in sorted_holidays if h > date.date()]
    if next_holidays:
        days_until_next_holiday[i] = (next_holidays[0] - date.date()).days
    else:
        days_until_next_holiday[i] = 365  # Default if no future holidays
    
    # Find days since last holiday
    prev_holidays = [h for h in sorted_holidays if h < date.date()]
    if prev_holidays:
        days_since_last_holiday[i] = (date.date() - prev_holidays[-1]).days
    else:
        days_since_last_holiday[i] = 365  # Default if no past holidays

# Add holiday features to dataframe
long_ts['is_holiday'] = is_holiday
long_ts['is_day_before_holiday'] = is_day_before_holiday
long_ts['is_day_after_holiday'] = is_day_after_holiday
long_ts['days_until_next_holiday'] = days_until_next_holiday
long_ts['days_since_last_holiday'] = days_since_last_holiday

# Sample of holiday features
print(long_ts.loc[long_ts['is_holiday'] == 1, 
                 ['date', 'is_holiday', 'is_day_before_holiday', 'is_day_after_holiday', 
                  'days_until_next_holiday', 'days_since_last_holiday']].head(10))
```

## Date Difference Features

Calculate differences between dates:

```python
# Create sample data with multiple date columns
customers = pd.DataFrame({
    'customer_id': range(1, 6),
    'registration_date': pd.to_datetime(['2021-01-15', '2021-03-20', '2021-06-10', 
                                         '2021-09-05', '2021-12-25']),
    'first_purchase_date': pd.to_datetime(['2021-01-18', '2021-03-25', '2021-07-01', 
                                           '2021-10-10', '2022-01-15']),
    'last_activity_date': pd.to_datetime(['2022-02-10', '2022-01-15', '2022-03-01', 
                                          '2022-02-20', '2022-03-10'])
})

# Calculate days between dates
customers['days_to_first_purchase'] = (
    customers['first_purchase_date'] - customers['registration_date']
).dt.days

customers['days_since_last_activity'] = (
    pd.Timestamp.today() - customers['last_activity_date']
).dt.days

customers['total_customer_days'] = (
    customers['last_activity_date'] - customers['registration_date']
).dt.days

# Calculate age of customer in months
customers['customer_age_months'] = (
    (pd.Timestamp.today() - customers['registration_date']) / np.timedelta64(1, 'M')
).round().astype(int)

print(customers)
```

## Feature Engineering Evaluation

Test the impact of temporal features on model performance:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Filter to complete data
complete_ts = long_ts.dropna().copy()

# Feature sets to compare
basic_features = ['year', 'month', 'day', 'dayofweek']
advanced_features = basic_features + [
    'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos',
    'is_holiday', 'is_weekend', 'days_until_next_holiday', 'days_since_last_holiday'
]

# Set up time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Storage for results
results = {'basic': [], 'advanced': []}

# Perform time series cross-validation
for train_idx, test_idx in tscv.split(complete_ts):
    # Split the data
    train_data = complete_ts.iloc[train_idx]
    test_data = complete_ts.iloc[test_idx]
    
    # Basic features model
    rf_basic = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_basic.fit(train_data[basic_features], train_data['value'])
    basic_pred = rf_basic.predict(test_data[basic_features])
    basic_rmse = np.sqrt(mean_squared_error(test_data['value'], basic_pred))
    basic_r2 = r2_score(test_data['value'], basic_pred)
    results['basic'].append((basic_rmse, basic_r2))
    
    # Advanced features model
    rf_advanced = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_advanced.fit(train_data[advanced_features], train_data['value'])
    advanced_pred = rf_advanced.predict(test_data[advanced_features])
    advanced_rmse = np.sqrt(mean_squared_error(test_data['value'], advanced_pred))
    advanced_r2 = r2_score(test_data['value'], advanced_pred)
    results['advanced'].append((advanced_rmse, advanced_r2))

# Calculate average performance
basic_avg_rmse = np.mean([r[0] for r in results['basic']])
basic_avg_r2 = np.mean([r[1] for r in results['basic']])
advanced_avg_rmse = np.mean([r[0] for r in results['advanced']])
advanced_avg_r2 = np.mean([r[1] for r in results['advanced']])

print(f"Basic Features - Avg RMSE: {basic_avg_rmse:.4f}, Avg R²: {basic_avg_r2:.4f}")
print(f"Advanced Features - Avg RMSE: {advanced_avg_rmse:.4f}, Avg R²: {advanced_avg_r2:.4f}")
print(f"Improvement: {(basic_avg_rmse - advanced_avg_rmse) / basic_avg_rmse * 100:.2f}% reduction in RMSE")

# Feature importance from the advanced model
final_model = RandomForestRegressor(n_estimators=100, random_state=42)
final_model.fit(complete_ts[advanced_features], complete_ts['value'])

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': advanced_features,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Date/Time Feature Importance')
plt.tight_layout()
plt.show()
```

## Best Practices for Date/Time Feature Engineering

1. **Start with basic components**: Extract year, month, day, dayofweek as a baseline
2. **Add cyclical encoding**: For features with periodicity like hour, day of week, month
3. **Consider domain context**: Different applications have different important time features
4. **Create holiday features**: For retail, finance, travel, and other holiday-sensitive domains
5. **Add lag features**: For time series forecasting tasks
6. **Use rolling windows**: For capturing trends and variability over time
7. **Encode special events**: Festivals, sports events, weather events as needed
8. **Test feature relevance**: Evaluate the impact of different feature sets
9. **Handle missing timestamps**: For irregular time series, consider time-since-last-event features
10. **Be wary of leakage**: Ensure temporal features don't leak future information 