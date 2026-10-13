"""Entry point: run experiments and print report."""
import csv
import pathlib
from anos.experiments import run_all, report

if __name__ == "__main__":
    results = run_all(
        seeds=list(range(50)),   # ۵۰ seed
        horizon=500,             # افق طولانی‌تر
        n_proxies=5,
        include_dqn=False,
    )
    report(results, baseline="ANOS")

    # ذخیرهٔ نتایج خام در CSV
    out = pathlib.Path("results/raw_results.csv")
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "seed_idx", "metric", "value"])
        for name, metrics in results.items():
            for metric, arr in metrics.items():
                for i, v in enumerate(arr):
                    w.writerow([name, i, metric, float(v)])
    print(f"\nSaved: {out}")
