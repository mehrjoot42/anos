"""Realistic censored-network proxy simulator.

Fixes over the paper:
- Non-Gaussian noise (Student-t for latency/jitter, log-normal for speed)
- Bursty failures via 2-state Markov chain per proxy
- Non-stationary drift + periodic active-censorship events
- Per-protocol blocking windows
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

PROTOCOLS = ["psiphon", "v2ray", "shadowsocks", "hysteria2", "psiphon_secondary"]

@dataclass
class ProxyState:
    name: str
    protocol: str
    base_speed: float
    base_latency: float
    base_jitter: float
    base_loss: float
    p_good_to_bad: float = 0.02
    p_bad_to_good: float = 0.30
    in_bad_state: bool = False
    drift_speed: float = 0.0
    drift_latency: float = 0.0

class CensoredNetworkSimulator:
    def __init__(self, seed: int = 0, n_proxies: int = 5, horizon: int = 200):
        self.rng = np.random.default_rng(seed)
        self.horizon = horizon
        self.t = 0
        self.proxies: List[ProxyState] = []
        base = [
            ("psiphon",            "psiphon",     20.0,  90.0, 25.0, 1.0),
            ("v2ray",              "v2ray",       45.0,  60.0, 10.0, 0.5),
            ("shadowsocks",        "shadowsocks", 30.0,  75.0, 15.0, 0.8),
            ("hysteria2",          "hysteria2",   60.0,  45.0,  8.0, 0.3),
            ("psiphon_secondary",  "psiphon",     15.0, 110.0, 35.0, 1.5),
        ][:n_proxies]
        for name, proto, sp, lat, jit, loss in base:
            self.proxies.append(ProxyState(name, proto, sp, lat, jit, loss))
        self._build_censorship_schedule()

    def _build_censorship_schedule(self):
        rng = self.rng
        self.censorship_events = []
        t = 0
        while t < self.horizon:
            t += int(rng.integers(15, 40))
            if t >= self.horizon:
                break
            dur = int(rng.integers(8, 20))
            k = int(rng.integers(1, 3))
            blocked = list(rng.choice(PROTOCOLS, size=k, replace=False))
            self.censorship_events.append((t, t + dur, set(blocked)))
            t += dur

    def _is_blocked(self, proto: str) -> bool:
        for s, e, protos in self.censorship_events:
            if s <= self.t < e and proto in protos:
                return True
        return False

    def _update_markov(self, p: ProxyState):
        if p.in_bad_state:
            if self.rng.random() < p.p_bad_to_good:
                p.in_bad_state = False
        else:
            if self.rng.random() < p.p_good_to_bad:
                p.in_bad_state = True

    def _update_drift(self, p: ProxyState):
        p.drift_speed   = 0.95 * p.drift_speed   + self.rng.normal(0, 0.5)
        p.drift_latency = 0.95 * p.drift_latency + self.rng.normal(0, 2.0)

    def sample(self, proxy_idx: int) -> Dict[str, float]:
        p = self.proxies[proxy_idx]
        self._update_markov(p)
        self._update_drift(p)
        blocked = self._is_blocked(p.protocol)
        if blocked:
            speed   = max(0.0, self.rng.normal(0.05, 0.05))
            latency = self.rng.normal(2000, 300)
            jitter  = self.rng.normal(500, 100)
            loss    = min(100.0, self.rng.normal(80, 10))
            fail    = 1
        else:
            speed   = max(0.1, self.rng.lognormal(np.log(p.base_speed + p.drift_speed), 0.35))
            lat_df  = max(3.0, 6.0 - 0.5 * p.in_bad_state)
            latency = max(1.0, p.base_latency + p.drift_latency
                          + self.rng.standard_t(lat_df) * p.base_jitter)
            jitter  = max(0.0, abs(self.rng.standard_t(lat_df) * p.base_jitter))
            loss    = max(0.0, min(100.0, self.rng.normal(
                p.base_loss * (3.0 if p.in_bad_state else 1.0), p.base_loss * 0.5)))
            fail = int(self.rng.random() < (0.4 if p.in_bad_state else 0.02) + loss / 200.0)
        return {
            "speed": float(speed), "latency": float(latency), "jitter": float(jitter),
            "loss": float(loss), "fail": fail,
            "blocked": int(blocked), "bad_state": int(p.in_bad_state),
        }

    def step(self):
        self.t += 1

    def reset(self):
        self.t = 0
