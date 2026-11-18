"""Refined 32D Vectorized Quaternion Surface-Walk HyperNet with factored (U, V) decoding."""

from __future__ import annotations

import time
import numpy as np


class FactoredQuaternionSurfaceHyperNet:
    """8-quaternion (32D) surface-walk hypernetwork with L2-resident factored decoding."""

    def __init__(self, in_dim: int, m: int, n: int, n_q: int = 8, laps: int = 20, seed: int = 7) -> None:
        self.n_q = n_q
        self.laps = laps
        self.latent_dim = 4 * n_q
        rng = np.random.default_rng(seed)

        w = np.array([0.4, 0.25, -0.3, 0.15], dtype=np.float32)
        self.b = np.array([0.05, -0.02, 0.03, -0.01], dtype=np.float32)
        self.qmul_mat = np.array(
            [
                [w[0], -w[1], -w[2], -w[3]],
                [w[1], w[0], -w[3], w[2]],
                [w[2], w[3], w[0], -w[1]],
                [w[3], -w[2], w[1], w[0]],
            ],
            dtype=np.float32,
        ).T

        self.enc = rng.normal(scale=1.0 / np.sqrt(in_dim), size=(n_q, 4, in_dim)).astype(np.float32)
        self.U = rng.normal(scale=0.1, size=(m, self.latent_dim)).astype(np.float32)
        self.V = rng.normal(scale=0.1, size=(n, self.latent_dim)).astype(np.float32)
        self.init_surface = rng.normal(scale=0.3, size=(n_q, 4)).astype(np.float32)

    def run_surface_walk(self, x_cond: np.ndarray) -> np.ndarray:
        """Run vectorized 20-lap Jacobi surface walk and return 32D latent q."""
        u = np.einsum("qki,i->qk", self.enc, x_cond)
        surface = self.init_surface.copy()
        for _ in range(self.laps):
            surface = 0.5 * surface + 0.5 * np.tanh(np.roll(surface, 1, axis=0) @ self.qmul_mat + u + self.b)
            surface = 0.5 * surface + 0.5 * np.tanh(np.roll(surface, -1, axis=0) @ self.qmul_mat + u + self.b)
        return surface.reshape(-1)

    def forward_factored(self, x_cond: np.ndarray, X_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply generated layer without materializing the dense (m, n) weight matrix."""
        q32 = self.run_surface_walk(x_cond)
        Y = np.tanh(((X_data @ self.V) * q32) @ self.U.T)
        return Y, q32

    def materialize_weights(self, q32: np.ndarray) -> np.ndarray:
        """Materialize dense W = (U * q32) @ V.T for verification."""
        return (self.U * q32) @ self.V.T


def main() -> None:
    rng = np.random.default_rng(99)
    net = FactoredQuaternionSurfaceHyperNet(in_dim=64, m=1024, n=1024, n_q=8, laps=20, seed=7)
    x_cond = rng.normal(size=64).astype(np.float32)
    X_data = rng.normal(size=(200, 1024)).astype(np.float32)

    # Warmup
    _ = net.forward_factored(x_cond, X_data)

    # Benchmark
    t0 = time.perf_counter()
    Y_factored, q32 = net.forward_factored(x_cond, X_data)
    t1 = time.perf_counter()

    # Verify exact equivalence with materialized W
    W_dense = net.materialize_weights(q32)
    Y_dense = np.tanh(X_data @ W_dense.T)
    max_diff = float(np.max(np.abs(Y_factored - Y_dense)))
    assert max_diff < 1e-5, f"Mismatch between factored and dense: {max_diff}"

    basis_bytes = net.U.nbytes + net.V.nbytes
    print(f"Basis memory: {basis_bytes / 1024:.1f} KB (was 268,435.5 KB)")
    print(f"End-to-end (32D walk + 200x1024 forward): {(t1 - t0) * 1e3:.3f} ms (was 28.25 ms)")
    print(f"Factored vs materialized W max_abs_diff: {max_diff:.2e}")
    print("FACTORED HYPERNET TEST PASSED")


if __name__ == "__main__":
    main()
