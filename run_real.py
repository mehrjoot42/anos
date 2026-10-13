"""Run ANOS experiments with REAL OONI censorship patterns (Iran 2024)."""
import numpy as np
from anos.simulator import CensoredNetworkSimulator
from anos.data_loader import load_censorship_schedule, load_summary_stats
from anos.selectors import (
    ANOSSelector, RandomSelector, RoundRobinSelector, BestSingleSelector,
    LinUCBSelector, ThompsonSelector, OracleSelector,
)
from anos.statistics import paired_wilcoxon, bootstrap_ci, holm_bonferroni

SWITCH_COST = 5.0
SEEDS = list(range(30))
HORIZON = 500
N_PROXIES = 5

# --- بارگذاری الگوهای سانسور واقعی ایران ---
real_schedule = load_censorship_schedule(horizon=HORIZON, scale=1.0)
stats = load_summary_stats()

print("=" * 68)
print("EXPERIMENT WITH REAL OONI DATA — IRAN (JAN-FEB 2024)")
print("=" * 68)
print(f"  Days of real data:         {stats['n_days']}")
print(f"  Mean anomaly rate:         {stats['mean_anomaly']:.2%}")
print(f"  Mean confirmed rate:       {stats['mean_confirmed']:.2%}")
print(f"  Std of daily anomaly:      {stats['std_anomaly']:.4f}")
print(f"  Min / Max anomaly:         {stats['min_anomaly']:.2%} / {stats['max_anomaly']:.2%}")
print(f"  Censorship events loaded:  {len(real_schedule)}")
print(f"  Horizon:                   {HORIZON} steps")
print(f"  Seeds:                     {len(SEEDS)}")
print("=" * 68)


def run_episode(selector, seed):
    """Run one episode with real censorship schedule injected."""
    sim = CensoredNetworkSimulator(
        seed=seed, n_proxies=N_PROXIES, horizon=HORIZON,
    )
    # جایگزینی جدول سانسور تصادفی با جدول واقعی
    sim.censorship_events = list(real_schedule)

    total_q = 0.0
    switches = 0
    last = None
    for t in range(HORIZON):
        feats = {i: sim.sample(i) for i in range(N_PROXIES)}
        choice = selector.select(t, feats)
        m = feats[choice]
        util = (0.5 * m["speed"]
                + 0.3 * max(0.0, 100.0 - m["latency"] / 10.0)
                + 0.2 * max(0.0, 100.0 - m["loss"] * 5.0))
        if m["fail"] or m["blocked"]:
            util -= 50.0
        util = float(np.clip(util, 0.0, 100.0))
        if last is not None and choice != last:
            switches += 1
            util -= SWITCH_COST
        last = choice
        total_q += util
        selector.update(choice, m, util)
        sim.step()
    return total_q / HORIZON


factories = {
    "ANOS":        lambda s: ANOSSelector(N_PROXIES, switch_cost=SWITCH_COST),
    "Random":      lambda s: RandomSelector(N_PROXIES, seed=s),
    "Round-Robin": lambda s: RoundRobinSelector(N_PROXIES),
    "Best-Single": lambda s: BestSingleSelector(N_PROXIES),
    "LinUCB":      lambda s: LinUCBSelector(N_PROXIES, dim=8, alpha=0.5, seed=s),
    "Thompson":    lambda s: ThompsonSelector(N_PROXIES, seed=s),
    "Oracle":      lambda s: OracleSelector(N_PROXIES, switch_cost=SWITCH_COST),
}

print("\nRunning experiments (30 seeds × 7 strategies)...")
results = {name: [] for name in factories}
for seed in SEEDS:
    print(f"  seed {seed:>2d}...", end="", flush=True)
    for name, factory in factories.items():
        sel = factory(seed)
        q = run_episode(sel, seed)
        results[name].append(q)
    print(" done")

# --- خلاصه ---
results_np = {k: np.array(v) for k, v in results.items()}

print("\n" + "=" * 68)
print("RESULTS WITH REAL IRANIAN CENSORSHIP PATTERNS")
print("=" * 68)
print(f"\n{'Strategy':>15s}  {'Mean':>8s}  {'95% CI':>22s}")
print("-" * 68)
for name, arr in results_np.items():
    lo, hi = bootstrap_ci(arr)
    print(f"{name:>15s}  {arr.mean():8.2f}  [{lo:6.2f}, {hi:6.2f}]")

# --- آزمون آماری ---
print("\n" + "=" * 68)
print("PAIRED WILCOXON vs ANOS  (Holm-Bonferroni corrected)")
print("=" * 68)
pvals, names, effects = [], [], []
for name, arr in results_np.items():
    if name == "ANOS":
        continue
    w = paired_wilcoxon(arr, results_np["ANOS"])
    pvals.append(w["p"])
    names.append(name)
    effects.append(w["effect"])

reject = holm_bonferroni(pvals)
for name, p, eff, rej in zip(names, pvals, effects, reject):
    sig = "✓ SIGNIFICANT" if rej else "✗ not significant"
    print(f"  {name:>15s}  p={p:.4f}  effect={eff:+.3f}  {sig}")

# --- ذخیره در CSV ---
import csv, pathlib
out = pathlib.Path("results/real_data_results.csv")
out.parent.mkdir(exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["strategy", "seed_idx", "mean_quality"])
    for name, arr in results_np.items():
        for i, v in enumerate(arr):
            w.writerow([name, i, float(v)])
print(f"\n✅ Saved raw results: {out}")
print("\n✅ Real-data experiment complete.")
