"""Proper statistics: paired Wilcoxon, effect size, bootstrap CI, Holm."""
from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple

def paired_wilcoxon(a, b) -> Dict[str, float]:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b):
        raise ValueError("Paired test requires equal-length arrays.")
    diffs = a - b
    if np.allclose(diffs, 0):
        return {"stat": 0.0, "p": 1.0, "effect": 0.0}
    stat, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    ranks = stats.rankdata(np.abs(diffs)); n = len(diffs)
    r_plus  = ranks[diffs > 0].sum()
    r_minus = ranks[diffs < 0].sum()
    total   = n * (n + 1) / 2
    effect  = (r_plus - r_minus) / total if total > 0 else 0.0
    return {"stat": float(stat), "p": float(p), "effect": float(effect)}

def bootstrap_ci(x, n_boot: int = 5000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed); x = np.asarray(x)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean()
                      for _ in range(n_boot)])
    return float(np.percentile(means, 100*alpha/2)), \
           float(np.percentile(means, 100*(1-alpha/2)))

def holm_bonferroni(pvals: List[float], alpha: float = 0.05) -> List[bool]:
    m = len(pvals); order = np.argsort(pvals); reject = [False] * m
    for rank, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject

def summarize(results: Dict[str, np.ndarray], baseline: str) -> List[Dict]:
    base = results[baseline]; rows, pvals = [], []
    for name, arr in results.items():
        if name == baseline: continue
        w = paired_wilcoxon(arr, base); pvals.append(w["p"])
        rows.append({"name": name, **w})
    rej = holm_bonferroni(pvals)
    for row, r in zip(rows, rej): row["significant_after_holm"] = bool(r)
    return rows
