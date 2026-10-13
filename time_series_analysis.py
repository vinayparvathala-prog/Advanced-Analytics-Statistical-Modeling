from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "daily_sales.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["Date"]).set_index("Date").sort_index()
daily = df["Sales"].asfreq("D").interpolate()
weekly = daily.resample("W").sum()
monthly = daily.resample("MS").sum()
weekly.rename("WeeklySales").to_csv(OUT / "weekly_sales.csv")
monthly.rename("MonthlySales").to_csv(OUT / "monthly_sales.csv")

# Plot daily and monthly sales
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
daily.plot(ax=axes[0], title="Daily Sales")
monthly.plot(ax=axes[1], title="Monthly Sales Total", marker="o")
plt.tight_layout()
fig.savefig(OUT / "sales_trends.png", dpi=150)
plt.close(fig)

# Seasonal decomposition; daily data has weekly seasonality (period 7)
decomp = seasonal_decompose(daily, model="additive", period=7, extrapolate_trend="freq")
fig = decomp.plot()
fig.set_size_inches(12, 8)
fig.tight_layout()
fig.savefig(OUT / "seasonal_decomposition.png", dpi=150)
plt.close(fig)

# Stationarity test
try:
    adf = adfuller(daily.dropna())
    adf_text = f"ADF statistic: {adf[0]:.4f}\np-value: {adf[1]:.6f}\n"
    adf_text += "Interpretation: likely stationary at 5% significance." if adf[1] < 0.05 else "Interpretation: evidence of non-stationarity at 5% significance."
except Exception as exc:
    adf_text = f"ADF test could not be completed: {exc}"
(OUT / "adf_test.txt").write_text(adf_text, encoding="utf-8")

# Hold out the last 30 days for evaluation
train, test = daily.iloc[:-30], daily.iloc[-30:]
forecasts = {}
# Naive moving average baseline
ma_value = train.tail(7).mean()
forecasts["7-day moving average"] = np.repeat(ma_value, len(test))
# Exponential smoothing
try:
    fit_es = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7).fit(optimized=True)
    forecasts["Exponential smoothing"] = fit_es.forecast(len(test)).to_numpy()
except Exception as exc:
    warnings.warn(f"Exponential smoothing failed; using moving average fallback: {exc}")
    forecasts["Exponential smoothing"] = np.repeat(ma_value, len(test))
# ARIMA model, with safe fallback
try:
    fit_arima = ARIMA(train, order=(2, 1, 2)).fit()
    forecasts["ARIMA(2,1,2)"] = fit_arima.forecast(steps=len(test)).to_numpy()
except Exception as exc:
    warnings.warn(f"ARIMA failed; using moving average fallback: {exc}")
    forecasts["ARIMA(2,1,2)"] = np.repeat(ma_value, len(test))

rows = []
for name, pred in forecasts.items():
    mae = mean_absolute_error(test, pred)
    rmse = np.sqrt(mean_squared_error(test, pred))
    denom = np.maximum(np.abs(test.to_numpy()), 1e-8)
    mape = np.mean(np.abs((test.to_numpy() - pred) / denom)) * 100
    rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "MAPE_percent": mape})
metrics = pd.DataFrame(rows).sort_values("MAE")
metrics.to_csv(OUT / "forecast_metrics.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 5))
test.plot(ax=ax, label="Actual", linewidth=2)
for name, pred in forecasts.items():
    ax.plot(test.index, pred, label=name)
ax.set_title("Forecast comparison (last 30 days)")
ax.set_ylabel("Sales")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "forecast_comparison.png", dpi=150)
plt.close(fig)

print("Time series analysis completed.")
print(f"Outputs saved to: {OUT}")
print("\nADF test:\n", adf_text)
print("\nForecast metrics:\n", metrics.to_string(index=False))
