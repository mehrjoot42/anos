"""Data-driven quality scoring with adaptive weights."""
from __future__ import annotations
import numpy as np
from collections import deque
from typing import Deque, Dict

class PercentileNormalizer:
    def __init__(self, window: int = 500, lower_q: float = 5.0, upper_q: float = 95.0):
        self.buf: Deque[float] = deque(maxlen=window)
        self.lower_q = lower_q
        self.upper_q = upper_q

    def update(self, x: float):
        self.buf.append(x)

    def _bounds(self):
        if len(self.buf) < 10:
            return None, None
        arr = np.asarray(self.buf)
        return np.percentile(arr, self.lower_q), np.percentile(arr, self.upper_q)

    def score(self, x: float, higher_is_better: bool = True) -> float:
        self.update(x)
        lo, hi = self._bounds()
        if lo is None or hi <= lo:
            return 50.0
        s = float(np.clip(100.0 * (x - lo) / (hi - lo), 0.0, 100.0))
        return s if higher_is_better else 100.0 - s

class QualityScorer:
    def __init__(self, weights: Dict[str, float] | None = None, adaptive: bool = False):
        self.weights = weights or {
            "speed": 0.40, "latency": 0.20, "jitter": 0.15,
            "loss": 0.15, "fail": 0.10,
        }
        self.norm = {k: PercentileNormalizer() for k in self.weights}
        self.adaptive = adaptive
        self.lr = 0.01
        self._last_scores: Dict[str, float] = {}

    def score(self, m: Dict[str, float]) -> float:
        s = {
            "speed":   self.norm["speed"].score(m["speed"], True),
            "latency": self.norm["latency"].score(m["latency"], False),
            "jitter":  self.norm["jitter"].score(m["jitter"], False),
            "loss":    self.norm["loss"].score(m["loss"], False),
            "fail":    self.norm["fail"].score(m["fail"], False),
        }
        penalty = 60.0 * m.get("blocked", 0) + 40.0 * m.get("fail", 0)
        self._last_scores = s
        q = sum(self.weights[k] * v for k, v in s.items())
        return float(np.clip(q - penalty, 0.0, 100.0))

    def observe_utility(self, realized_utility: float):
        if not self.adaptive or not self._last_scores:
            return
        predicted = sum(self.weights[k] * v for k, v in self._last_scores.items())
        err = realized_utility - predicted
        for k in self.weights:
            self.weights[k] += self.lr * err * self._last_scores[k] / 100.0
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights:
                self.weights[k] /= total
