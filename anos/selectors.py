"""Selection strategies: ANOS + baselines + RL."""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional
from .scoring import QualityScorer

class BaseSelector:
    name = "base"
    def select(self, t, features): raise NotImplementedError
    def update(self, i, features, r): pass

class RandomSelector(BaseSelector):
    name = "Random"
    def __init__(self, n: int, seed: int = 0):
        self.n, self.rng = n, np.random.default_rng(seed)
    def select(self, t, features): return int(self.rng.integers(0, self.n))

class RoundRobinSelector(BaseSelector):
    name = "Round-Robin"
    def __init__(self, n: int): self.n, self.i = n, 0
    def select(self, t, features):
        idx = self.i % self.n; self.i += 1; return idx

class BestSingleSelector(BaseSelector):
    name = "Best-Single"
    def __init__(self, n: int):
        self.n = n
        self.sum = np.zeros(n); self.cnt = np.zeros(n) + 1e-9
    def select(self, t, features): return int(np.argmax(self.sum / self.cnt))
    def update(self, i, features, r):
        self.sum[i] += r; self.cnt[i] += 1

class ANOSSelector(BaseSelector):
    name = "ANOS"
    def __init__(self, n: int, switch_cost: float = 5.0, margin: float = 3.0,
                 min_dwell: int = 3, adaptive_weights: bool = False):
        self.n = n
        self.scorer = QualityScorer(adaptive=adaptive_weights)
        self.switch_cost = switch_cost
        self.margin = margin
        self.min_dwell = min_dwell
        self.current: Optional[int] = None
        self.dwell = 0
        self._last = {}

    def select(self, t, features):
        scores = {i: self.scorer.score(m) for i, m in features.items()}
        self._last = features
        best = max(scores, key=scores.get)
        if self.current is None:
            self.current = best; self.dwell = 0; return best
        self.dwell += 1
        if self.dwell < self.min_dwell:
            return self.current
        gain = scores[best] - scores[self.current]
        threshold = self.margin + self.switch_cost / max(1, self.dwell)
        if gain > threshold:
            self.current = best; self.dwell = 0
        return self.current

    def update(self, i, features, r):
        self.scorer.observe_utility(r)

class LinUCBSelector(BaseSelector):
    name = "LinUCB"
    def __init__(self, n: int, dim: int, alpha: float = 0.5, seed: int = 0):
        self.n, self.dim, self.alpha = n, dim, alpha
        self.A = [np.eye(dim) for _ in range(n)]
        self.b = [np.zeros(dim) for _ in range(n)]
        self._x = {}
    @staticmethod
    def _feat(f):
        return np.array([1.0, f["speed"]/100.0, f["latency"]/1000.0,
                         f["jitter"]/500.0, f["loss"]/100.0,
                         float(f.get("fail", 0)), float(f.get("blocked", 0)),
                         float(f.get("bad_state", 0))], dtype=float)
    def select(self, t, features):
        best, best_ucb = None, -1e18
        for i, m in features.items():
            x = self._feat(m); self._x[i] = x
            A_inv = np.linalg.inv(self.A[i]); theta = A_inv @ self.b[i]
            ucb = float(theta @ x) + self.alpha * np.sqrt(max(0.0, x @ A_inv @ x))
            if ucb > best_ucb: best_ucb, best = ucb, i
        return best
    def update(self, i, features, r):
        x = self._x.get(i, self._feat(features))
        self.A[i] += np.outer(x, x); self.b[i] += r * x

class ThompsonSelector(BaseSelector):
    name = "Thompson"
    def __init__(self, n: int, seed: int = 0, prior_var: float = 100.0):
        self.n, self.rng = n, np.random.default_rng(seed)
        self.mu = np.zeros(n); self.var = np.full(n, prior_var)
        self.k = np.zeros(n)
    def select(self, t, features):
        return int(np.argmax(self.rng.normal(self.mu, np.sqrt(self.var))))
    def update(self, i, features, r):
        k = self.k[i]; old = self.mu[i]
        new = (old * k + r) / (k + 1)
        if k > 0:
            self.var[i] = (self.var[i] * k + (r - new) * (r - old)) / (k + 1)
        self.mu[i] = new; self.k[i] = k + 1

class DQNSelector(BaseSelector):
    name = "DQN"
    def __init__(self, n: int, dim: int, seed: int = 0, lr: float = 1e-3,
                 buffer_size: int = 2000, batch_size: int = 32):
        self.n, self.dim = n, dim
        try:
            import torch
            import torch.nn as nn
            self.torch = torch
        except ImportError:
            self._fallback = LinUCBSelector(n, dim, seed=seed); return
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buf, self.batch_size, self.buffer_size = [], batch_size, buffer_size
        self.eps, self.gamma = 1.0, 0.95
        self._last_x = None
    def _feat(self, f):
        return np.array([1.0, f["speed"]/100.0, f["latency"]/1000.0,
                         f["jitter"]/500.0, f["loss"]/100.0,
                         float(f.get("fail", 0)), float(f.get("blocked", 0)),
                         float(f.get("bad_state", 0))], dtype=float)
    def select(self, t, features):
        if hasattr(self, "_fallback"): return self._fallback.select(t, features)
        torch = self.torch
        X = np.stack([self._feat(features[i]) for i in range(self.n)])
        ctx = np.concatenate([X.mean(0), X.max(0), X.min(0)])
        ctx = (ctx if ctx.shape[0] >= self.dim else np.pad(ctx, (0, self.dim - ctx.shape[0])))[:self.dim]
        self._last_x = ctx
        if np.random.random() < self.eps:
            a = int(np.random.randint(self.n))
        else:
            with torch.no_grad():
                a = int(self.net(torch.tensor(ctx, dtype=torch.float32)).argmax().item())
        return a
    def update(self, i, features, r):
        if hasattr(self, "_fallback"): return self._fallback.update(i, features, r)
        self.buf.append((self._last_x, i, r))
        if len(self.buf) > self.buffer_size: self.buf.pop(0)
        if len(self.buf) < self.batch_size: return
        torch = self.torch
        idx = np.random.choice(len(self.buf), self.batch_size, replace=False)
        xs  = torch.tensor(np.stack([self.buf[j][0] for j in idx]), dtype=torch.float32)
        as_ = torch.tensor([self.buf[j][1] for j in idx], dtype=torch.long)
        rs  = torch.tensor([self.buf[j][2] for j in idx], dtype=torch.float32)
        q = self.net(xs).gather(1, as_.unsqueeze(1)).squeeze(1)
        loss = ((q - rs) ** 2).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.eps = max(0.05, self.eps * 0.999)

class OracleSelector(BaseSelector):
    name = "Oracle"
    def __init__(self, n: int, switch_cost: float = 5.0):
        self.n, self.switch_cost = n, switch_cost
        self.current = None; self.scorer = QualityScorer()
    def select(self, t, features):
        scores = {i: self.scorer.score(m) for i, m in features.items()}
        best = max(scores, key=scores.get)
        if self.current is None:
            self.current = best
        elif scores[best] - scores[self.current] > self.switch_cost:
            self.current = best
        return self.current
    def update(self, i, features, r): pass
