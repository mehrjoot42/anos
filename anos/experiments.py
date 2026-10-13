"""Experiment runner: paired seeds, multiple metrics, proper stats."""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Type

from .simulator import CensoredNetworkSimulator
from .selectors import (
    ANOSSelector, RandomSelector, RoundRobinSelector, BestSingleSelector,
    LinUCBSelector, ThompsonSelector, DQNSelector, OracleSelector, BaseSelector,
)
from .statistics import summarize, bootstrap_ci

SWITCH_COST = 5.0

def run_episode(selector: BaseSelector, seed: int, horizon: int = 200,
                n_proxies: int = 5) -> Dict[str, float]:
    sim = CensoredNetworkSimulator(seed=seed, n_proxies=n_proxies, horizon=horizon)
    total_q = 0.0; switches = 0; bytes_ok = 0.0
    ttfb_sum = 0.0; disconnected = 0; last = None

    for t in range(horizon):
        features = {i: sim.sample(i) for i in range(n_proxies)}
        choice = selector.select(t, features)
        m = features[choice]
        util = (0.5 * m["speed"]
                + 0.3 * max(0.0, 100.0 - m["latency"] / 10.0)
                + 0.2 * max(0.0, 100.0 - m["loss"] * 5.0))
        if m["fail"] or m["blocked"]: util -= 50.0
        util = float(np.clip(util, 0.0, 100.0))
        if last is not None and choice != last:
            switches += 1; util -= SWITCH_COST
        last = choice
        total_q += util
        bytes_ok += m["speed"] * 1e6 / 8
        ttfb_sum += m["latency"]
        disconnected += int(m["fail"] or m["blocked"])
        selector.update(choice, m, util)
        sim.step()

    return {
        "mean_quality": total_q / horizon,
        "switches": float(switches),
        "bytes_ok": bytes_ok,
        "ttfb_mean": ttfb_sum / horizon,
        "disconnections": float(disconnected),
    }

def run_all(seeds=(42, 123, 456, 789, 1011), horizon: int = 200,
            n_proxies: int = 5, include_dqn: bool = False):
    factories: Dict[str, Type[BaseSelector]] = {
        "ANOS":          lambda s: ANOSSelector(n_proxies, switch_cost=SWITCH_COST),
        "ANOS-adaptive": lambda s: ANOSSelector(n_proxies, switch_cost=SWITCH_COST,
                                                adaptive_weights=True),
        "Random":        lambda s: RandomSelector(n_proxies, seed=s),
        "Round-Robin":   lambda s: RoundRobinSelector(n_proxies),
        "Best-Single":   lambda s: BestSingleSelector(n_proxies),
        "LinUCB":        lambda s: LinUCBSelector(n_proxies, dim=8, alpha=0.5, seed=s),
        "Thompson":      lambda s: ThompsonSelector(n_proxies, seed=s),
        "Oracle":        lambda s: OracleSelector(n_proxies, switch_cost=SWITCH_COST),
    }
    if include_dqn:
        factories["DQN"] = lambda s: DQNSelector(n_proxies, dim=24, seed=s)

    keys = ["mean_quality", "switches", "bytes_ok", "ttfb_mean", "disconnections"]
    results = {name: {k: [] for k in keys} for name in factories}

    for seed in seeds:
        for name, factory in factories.items():
            sel = factory(seed)
            out = run_episode(sel, seed=seed, horizon=horizon, n_proxies=n_proxies)
            for k in keys:
                results[name][k].append(out[k])

    return {n: {k: np.asarray(v) for k, v in d.items()} for n, d in results.items()}

def report(results, baseline: str = "ANOS"):
    print(f"\n=== Comparison against {baseline} (paired across seeds) ===")
    quality = {k: v["mean_quality"] for k, v in results.items()}
    for row in summarize(quality, baseline=baseline):
        print(f"{row['name']:>15s}  p={row['p']:.4f}  "
              f"effect={row['effect']:+.3f}  "
              f"holm_sig={row['significant_after_holm']}")
    print("\n=== Mean quality with 95% bootstrap CI ===")
    for name, arr in quality.items():
        lo, hi = bootstrap_ci(arr)
        print(f"{name:>15s}  mean={arr.mean():6.2f}  CI=[{lo:6.2f}, {hi:6.2f}]")
