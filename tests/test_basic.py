"""Basic sanity tests."""
import numpy as np
from anos.simulator import CensoredNetworkSimulator
from anos.scoring   import QualityScorer
from anos.selectors import ANOSSelector, LinUCBSelector, ThompsonSelector
from anos.experiments import run_episode

def test_simulator_shapes():
    sim = CensoredNetworkSimulator(seed=0)
    m = sim.sample(0)
    assert set(m.keys()) >= {"speed", "latency", "jitter", "loss", "fail"}

def test_scorer_range():
    sc = QualityScorer()
    for _ in range(50):
        q = sc.score({"speed": 30, "latency": 50, "jitter": 5,
                      "loss": 1, "fail": 0, "blocked": 0})
        assert 0.0 <= q <= 100.0

def test_selectors_return_valid_index():
    sim = CensoredNetworkSimulator(seed=1)
    for sel in (ANOSSelector(5), LinUCBSelector(5, 8), ThompsonSelector(5)):
        features = {i: sim.sample(i) for i in range(5)}
        idx = sel.select(0, features)
        assert 0 <= idx < 5

def test_episode_runs():
    out = run_episode(ANOSSelector(5), seed=42, horizon=30)
    assert out["mean_quality"] >= 0.0
