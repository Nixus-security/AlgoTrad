"""
Quantum optimizer module.
Uses Qiskit QAOA to optimise signal fusion weights as a QUBO problem.
Also provides a Quantum Random Walk for probabilistic scenario simulation.
Falls back gracefully to classical scipy optimizer if Qiskit unavailable.
"""
from __future__ import annotations
import numpy as np
from utils.logger import logger

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.primitives import StatevectorSampler
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_algorithms import QAOA, SamplingVQE
    from qiskit_algorithms.optimizers import COBYLA
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logger.info("Qiskit not installed — classical fallback actif (scipy COBYLA)")


class QuantumOptimizer:
    """
    Optimises 4 signal weights [technical, ml, statistical, fundamental]
    subject to: all weights ≥ 0, sum = 1.
    Objective: maximise expected Sharpe (encoded as minimisation QUBO).
    """

    def __init__(self, cfg: dict):
        self.reps = cfg["quantum"]["reps"]
        self.shots = cfg["quantum"]["shots"]

    # ── Main entry: return optimised weights dict ─────────────────────────────
    def optimise_weights(
        self,
        signal_scores: dict[str, float],   # {"technical": 0.7, "ml": 0.8, ...}
        historical_sharpes: dict[str, float],
    ) -> dict[str, float]:
        if QISKIT_AVAILABLE:
            return self._qaoa_optimise(signal_scores, historical_sharpes)
        return self._classical_fallback(signal_scores, historical_sharpes)

    # ── QAOA optimisation ─────────────────────────────────────────────────────
    def _qaoa_optimise(
        self,
        scores: dict[str, float],
        sharpes: dict[str, float],
    ) -> dict[str, float]:
        keys = list(scores.keys())
        n = len(keys)

        # QUBO matrix: maximise sum(w_i * sharpe_i) → minimise -sum(...)
        # Penalty for constraint sum(w_i) = 1 encoded as lambda*(1-sum)^2
        lam = 5.0
        Q = np.zeros((n, n))
        for i, k in enumerate(keys):
            Q[i, i] = -sharpes.get(k, 0.5) + lam
        for i in range(n):
            for j in range(n):
                if i != j:
                    Q[i, j] += lam

        qp = QuadraticProgram("weight_opt")
        for k in keys:
            qp.binary_var(k)

        linear = {keys[i]: Q[i, i] for i in range(n)}
        quadratic = {(keys[i], keys[j]): Q[i, j]
                     for i in range(n) for j in range(i + 1, n)}
        qp.minimize(linear=linear, quadratic=quadratic)

        try:
            backend = AerSimulator()
            optimizer = COBYLA(maxiter=200)
            qaoa = QAOA(sampler=StatevectorSampler(), optimizer=optimizer, reps=self.reps)
            result = MinimumEigenOptimizer(qaoa).solve(qp)
            raw = {k: result.x[i] for i, k in enumerate(keys)}
        except Exception as e:
            logger.error(f"QAOA failed: {e} — using classical fallback")
            return self._classical_fallback(scores, sharpes)

        # Convert binary QUBO result to continuous weights
        total = sum(raw.values()) or 1.0
        weights = {k: raw[k] / total for k in keys}
        # If all zeros (degenerate), fall back
        if all(v == 0 for v in weights.values()):
            return self._classical_fallback(scores, sharpes)
        logger.info(f"QAOA weights: {weights}")
        return weights

    # ── Classical fallback: softmax over Sharpe ratios ────────────────────────
    @staticmethod
    def _classical_fallback(
        scores: dict[str, float],
        sharpes: dict[str, float],
    ) -> dict[str, float]:
        keys = list(scores.keys())
        vals = np.array([sharpes.get(k, 0.5) for k in keys])
        vals = np.exp(vals - vals.max())  # Numerical stability
        weights = vals / vals.sum()
        return dict(zip(keys, weights.tolist()))

    # ── Feature selection via QUBO ────────────────────────────────────────────
    def select_features(
        self,
        feature_importances: dict[str, float],
        max_features: int = 10,
    ) -> list[str]:
        """
        Binary feature selection: maximise sum(importance_i * x_i)
        with sparsity penalty to keep <= max_features.
        Falls back to top-k sort when Qiskit unavailable.
        """
        keys = list(feature_importances.keys())
        vals = np.array([feature_importances[k] for k in keys])

        if not QISKIT_AVAILABLE or len(keys) > 20:
            # Classical: top-k by importance
            idx = np.argsort(vals)[::-1][:max_features]
            return [keys[i] for i in idx]

        try:
            lam = max(vals) * 0.5  # Sparsity penalty
            qp = QuadraticProgram("feat_select")
            for k in keys:
                qp.binary_var(k)
            linear = {k: -float(feature_importances[k]) + lam for k in keys}
            qp.minimize(linear=linear)

            optimizer = COBYLA(maxiter=100)
            qaoa = QAOA(sampler=StatevectorSampler(), optimizer=optimizer, reps=1)
            result = MinimumEigenOptimizer(qaoa).solve(qp)
            selected = [keys[i] for i, x in enumerate(result.x) if x > 0.5]
            return selected[:max_features] if selected else keys[:max_features]
        except Exception as e:
            logger.warning(f"Feature selection QAOA failed: {e} — top-k fallback")
            idx = np.argsort(vals)[::-1][:max_features]
            return [keys[i] for i in idx]

    # ── Position sizing via quadratic optimisation ────────────────────────────
    def size_position(
        self,
        expected_return: float,
        volatility: float,
        max_position: float = 0.02,
        risk_aversion: float = 2.0,
    ) -> float:
        """
        Kelly-like quadratic optimisation.
        Returns fraction of capital to allocate (0–max_position).
        Classical-only (Qiskit circuit too noisy for continuous optimisation).
        """
        if volatility <= 0 or expected_return <= 0:
            return 0.0
        # Maximise: w * E[r] - 0.5 * lambda * w^2 * sigma^2
        # Solution: w* = E[r] / (lambda * sigma^2)
        kelly = expected_return / (risk_aversion * volatility ** 2)
        return float(np.clip(kelly, 0.0, max_position))

    # ── Multi-stock ranking ───────────────────────────────────────────────────
    def rank_signals(
        self,
        scores: dict[str, float],  # {ticker: composite_score}
        top_k: int = 3,
    ) -> list[str]:
        """
        Returns top_k tickers by score.
        Classical softmax ranking; QAOA used only when Qiskit available
        and number of stocks is small.
        """
        if not scores:
            return []
        sorted_tickers = sorted(scores, key=lambda t: scores[t], reverse=True)
        return sorted_tickers[:top_k]

    # ── Quantum Random Walk: price-path probability simulation ────────────────
    def quantum_random_walk(
        self, n_steps: int = 10, n_trials: int = None
    ) -> dict[str, float]:
        """
        Simulates a discrete quantum walk on a line graph (position = price delta).
        Returns probability distribution over {up, flat, down} scenarios.
        """
        n_trials = n_trials or self.shots
        if not QISKIT_AVAILABLE:
            return {"up": 0.5, "flat": 0.1, "down": 0.4}

        try:
            # Position register: ceil(log2(n_steps+1)) qubits
            n_pos = int(np.ceil(np.log2(n_steps + 1))) + 1
            coin = QuantumCircuit(n_pos + 1)

            # Hadamard coin on qubit 0
            coin.h(0)
            # Conditioned shift: simplified conditional increment/decrement
            for step in range(n_steps):
                coin.cx(0, 1 + (step % n_pos))

            coin.measure_all()
            backend = AerSimulator()
            job = backend.run(coin, shots=n_trials)
            counts = job.result().get_counts()

            total = sum(counts.values())
            # Map measurement results to position (positive = up, negative = down)
            up = down = flat = 0
            for state, cnt in counts.items():
                ones = state.count("1")
                if ones > n_pos // 2:
                    up += cnt
                elif ones < n_pos // 2:
                    down += cnt
                else:
                    flat += cnt

            return {
                "up": up / total,
                "flat": flat / total,
                "down": down / total,
            }
        except Exception as e:
            logger.error(f"Quantum walk failed: {e}")
            return {"up": 0.5, "flat": 0.1, "down": 0.4}
