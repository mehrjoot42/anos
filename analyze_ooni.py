"""Analyze multi-month OONI Iran data."""
import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # برای محیط‌های بدون نمایشگر
import matplotlib.pyplot as plt

CSV = pathlib.Path("ooni_data/IR/iran_multi_month.csv")

print("Loading data (this may take 20-30 seconds)...")
df = pd.read_csv(CSV)
df["timestamp"] = pd.to_datetime(df["measurement_start_time"])
df["date"] = df["timestamp"].dt.date

print("=" * 60)
print(f"Total measurements: {len(df):,}")
print(f"Date range: {df['date'].min()}  →  {df['date'].max()}")
print(f"Days covered: {df['date'].nunique()}")
print("=" * 60)

print(f"\n[Overall censorship statistics]")
print(f"  Anomaly rate:   {df['anomaly'].mean():.2%}")
print(f"  Confirmed rate: {df['confirmed'].mean():.2%}")
print(f"  Failure rate:   {df['failure'].mean():.2%}")

# خلاصه روزانه
daily = df.groupby("date").agg(
    n=("measurement_uid", "count"),
    anomaly_rate=("anomaly", "mean"),
    confirmed_rate=("confirmed", "mean"),
    failure_rate=("failure", "mean"),
).round(4)

print(f"\n[Daily statistics - first 10 days]")
print(daily.head(10))

print(f"\n[Daily statistics - last 5 days]")
print(daily.tail(5))

print(f"\n[Anomaly rate distribution across days]")
print(f"  Min:    {daily['anomaly_rate'].min():.2%}")
print(f"  Median: {daily['anomaly_rate'].median():.2%}")
print(f"  Mean:   {daily['anomaly_rate'].mean():.2%}")
print(f"  Max:    {daily['anomaly_rate'].max():.2%}")
print(f"  Std:    {daily['anomaly_rate'].std():.4f}")

# ذخیره
daily.to_csv("ooni_data/IR/daily_censorship_summary.csv")
print("\n✅ Saved: ooni_data/IR/daily_censorship_summary.csv")

# ---- نمودارها ----
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# رنگ‌آمیزی روزها بر اساس شدت
colors = ["red" if x > 0.6 else "orange" if x > 0.4 else "green"
          for x in daily["anomaly_rate"]]

axes[0].bar(daily.index, daily["anomaly_rate"], color=colors, alpha=0.7)
axes[0].axhline(daily["anomaly_rate"].mean(), color="black",
                linestyle="--", label=f"Mean = {daily['anomaly_rate'].mean():.2%}")
axes[0].set_ylabel("Anomaly rate")
axes[0].set_title("Iran censorship (real OONI data, Jan-Feb 2024)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(daily.index, daily["confirmed_rate"], color="darkred", alpha=0.7)
axes[1].set_ylabel("Confirmed rate")
axes[1].grid(True, alpha=0.3)

axes[2].bar(daily.index, daily["n"], color="steelblue", alpha=0.7)
axes[2].set_ylabel("Measurements/day")
axes[2].set_xlabel("Date")
axes[2].grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("ooni_data/IR/censorship_timeseries.png", dpi=100)
print("✅ Saved plot: ooni_data/IR/censorship_timeseries.png")

# ---- محاسبه آمار برای شبیه‌ساز ----
print("\n" + "=" * 60)
print("STATISTICS FOR SIMULATOR CALIBRATION")
print("=" * 60)
print(f"  Mean anomaly rate (event probability):  {daily['anomaly_rate'].mean():.4f}")
print(f"  Std of anomaly rate (burstiness):       {daily['anomaly_rate'].std():.4f}")
print(f"  Max/min ratio:                          {daily['anomaly_rate'].max() / max(daily['anomaly_rate'].min(), 0.001):.2f}")
print(f"  Days with >50% anomaly:                 {(daily['anomaly_rate'] > 0.5).sum()} / {len(daily)}")
print(f"  Days with <30% anomaly:                 {(daily['anomaly_rate'] < 0.3).sum()} / {len(daily)}")
