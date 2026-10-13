"""Load real OONI-derived censorship patterns for the simulator."""
import pathlib
import pandas as pd
from typing import List, Tuple


def load_censorship_schedule(
    summary_csv: str = "ooni_data/IR/daily_censorship_summary.csv",
    horizon: int = 500,
    scale: float = 1.0,
    min_intensity: float = 0.35,
) -> List[Tuple[int, int, set]]:
    """
    Convert daily real censorship rates into simulator events.

    Args:
        summary_csv: path to daily summary CSV
        horizon: number of simulator steps
        scale: multiplier for anomaly rates (to tune severity)
        min_intensity: below this threshold, no event is created

    Returns:
        List of (start_step, end_step, blocked_protocols_set)
    """
    df = pd.read_csv(summary_csv)
    n_days = len(df)
    steps_per_day = max(1, horizon // n_days)

    events = []
    protocols = ["psiphon", "v2ray", "shadowsocks", "hysteria2"]

    for i, row in df.iterrows():
        anomaly = float(row["anomaly_rate"]) * scale
        confirmed = float(row["confirmed_rate"]) * scale
        intensity = min(1.0, (anomaly + confirmed) / 2.0)

        if intensity < min_intensity:
            continue

        # تعداد پروتکل‌های مسدود بر اساس شدت
        k = min(len(protocols), max(1, int(round(intensity * len(protocols)))))
        blocked = set(protocols[:k])

        start = i * steps_per_day
        # طول رویداد متناسب با شدت
        duration = max(1, int(steps_per_day * intensity))
        end = start + duration
        events.append((start, end, blocked))

    return events


def load_summary_stats(summary_csv: str = "ooni_data/IR/daily_censorship_summary.csv"):
    """Return aggregate statistics for reporting."""
    df = pd.read_csv(summary_csv)
    return {
        "n_days": len(df),
        "mean_anomaly": df["anomaly_rate"].mean(),
        "std_anomaly": df["anomaly_rate"].std(),
        "mean_confirmed": df["confirmed_rate"].mean(),
        "mean_failure": df["failure_rate"].mean(),
        "max_anomaly": df["anomaly_rate"].max(),
        "min_anomaly": df["anomaly_rate"].min(),
    }


if __name__ == "__main__":
    events = load_censorship_schedule(horizon=500)
    stats = load_summary_stats()
    print(f"Loaded {len(events)} censorship events from {stats['n_days']} real days")
    print(f"Mean anomaly rate: {stats['mean_anomaly']:.2%}")
    print(f"First 5 events:")
    for e in events[:5]:
        print(f"  steps [{e[0]:>3d}, {e[1]:>3d}]  blocked: {sorted(e[2])}")