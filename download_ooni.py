"""Download OONI measurements for Iran over multiple months (day by day)."""
import json
import csv
import pathlib
import time
import requests
from datetime import date, timedelta

OUT_DIR = pathlib.Path("ooni_data/IR")
OUT_DIR.mkdir(parents=True, exist_ok=True)

API = "https://api.ooni.io/api/v1/measurements"
CSV_PATH = OUT_DIR / "iran_multi_month.csv"


def fetch_one_day(country, day, test_name="web_connectivity", limit=5000):
    """Fetch measurements for a single day."""
    start = day.isoformat()
    end = (day + timedelta(days=1)).isoformat()
    params = {
        "probe_cc": country,
        "since": start,
        "until": end,
        "test_name": test_name,
        "limit": limit,
    }
    r = requests.get(API, params=params, timeout=120)
    r.raise_for_status()
    return r.json().get("results", [])


def append_to_csv(rows, csv_path):
    """Append new rows to CSV, creating header if file doesn't exist."""
    if not rows:
        return
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                "measurement_uid", "measurement_start_time", "probe_cc",
                "test_name", "input", "anomaly", "confirmed", "failure",
                "report_id",
            ])
        for m in rows:
            w.writerow([
                m.get("measurement_uid", ""),
                m.get("measurement_start_time", ""),
                m.get("probe_cc", ""),
                m.get("test_name", ""),
                m.get("input", ""),
                m.get("anomaly", ""),
                m.get("confirmed", ""),
                m.get("failure", ""),
                m.get("report_id", ""),
            ])


if __name__ == "__main__":
    # ۶۰ روز: از ۲۰۲۴-۰۱-۰۱ تا ۲۰۲۴-۰۳-۰۱
    start_day = date(2024, 1, 1)
    end_day   = date(2024, 3, 1)
    d = start_day
    total = 0
    while d < end_day:
        try:
            rows = fetch_one_day("IR", d)
            append_to_csv(rows, CSV_PATH)
            total += len(rows)
            print(f"[{d}] fetched {len(rows)} rows   (total: {total})")
        except Exception as e:
            print(f"[{d}] ERROR: {e}")
        d += timedelta(days=1)
        time.sleep(0.5)  # polite rate limit
    print(f"\n✅ Done. Total rows: {total}")
    print(f"   Saved to: {CSV_PATH}")
