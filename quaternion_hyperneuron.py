"""Single quaternion neuron hypernetwork that outputs m x n weight matrices."""

from __future__ import annotations

import numpy as np
from quaternion_neuron import QuaternionNeuron, simulate_multi_order_ode


class QuaternionHyperNeuron:
    """Encodes input data into a single quaternion neuron ODE and outputs an m x n weight matrix."""

    def __init__(
        self,
        in_dim: int,
        out_shape: tuple[int, int],
        coeffs: list[float] | None = None,
        seed: int = 0,
    ) -> None:
        self.in_dim = in_dim
        self.out_shape = out_shape
        self.coeffs = [1.5, 1.8] if coeffs is None else list(coeffs)

        rng = np.random.default_rng(seed)
        self.neuron = QuaternionNeuron(
            weight=rng.normal(scale=0.3, size=4),
            bias=rng.normal(scale=0.1, size=4),
        )
        # Encoder: R^in_dim -> H (4)
        self.encoder = rng.normal(scale=1.0 / np.sqrt(in_dim), size=(4, in_dim))
        # Decoder basis: H (4) -> R^(m x n)
        m, n = out_shape
        self.basis = rng.normal(scale=0.5, size=(4, m, n))

    def generate_weights(self, x: np.ndarray, t_end: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
        """Run the single-neuron ODE for input x and decode q* into an (m, n) weight matrix."""
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.shape != (self.in_dim,):
            raise ValueError(f"Expected input shape ({self.in_dim},), got {x_arr.shape}")

        u_const = self.encoder @ x_arr
        order = len(self.coeffs)
        y0 = np.zeros(4 * order, dtype=np.float64)
        sol = simulate_multi_order_ode(
            neuron=self.neuron,
            coeffs=self.coeffs,
            y0=y0,
            t_span=(0.0, t_end),
            u_fn=lambda _t: u_const,
            n_eval=200,
        )
        q_star = sol.y[:4, -1]
        w_out = np.tensordot(q_star, self.basis, axes=(0, 0))
        return w_out, q_star


class MultiQuaternionHyperNet:
    """N coupled/parallel quaternion neurons (4*N DOF) outputting an m x n weight matrix."""

    def __init__(
        self,
        in_dim: int,
        out_shape: tuple[int, int],
        n_quaternions: int = 8,
        coeffs: list[float] | None = None,
        seed: int = 0,
    ) -> None:
        self.n_quaternions = n_quaternions
        self.latent_dim = 4 * n_quaternions
        self.neurons = [
            QuaternionHyperNeuron(in_dim=in_dim, out_shape=out_shape, coeffs=coeffs, seed=seed + i)
            for i in range(n_quaternions)
        ]

    def generate_weights(self, x: np.ndarray, t_end: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
        """Run all N quaternion neuron ODEs and sum their basis expansions."""
        w_total = np.zeros(self.neurons[0].out_shape, dtype=np.float64)
        q_all = []
        for cell in self.neurons:
            w_i, q_i = cell.generate_weights(x, t_end=t_end)
            w_total += w_i
            q_all.append(q_i)
        return w_total, np.concatenate(q_all)


if __name__ == "__main__":
    hyper = QuaternionHyperNeuron(in_dim=6, out_shape=(3, 5), coeffs=[1.5, 1.8], seed=7)
    x1 = np.array([0.5, -1.2, 0.3, 0.8, -0.4, 1.1])
    x2 = np.array([-0.7, 0.4, 1.0, -0.2, 0.9, -0.5])

    W1, q1 = hyper.generate_weights(x1)
    W2, q2 = hyper.generate_weights(x2)

    assert W1.shape == (3, 5) and W2.shape == (3, 5)
    assert np.all(np.isfinite(W1)) and np.all(np.isfinite(W2))
    assert not np.allclose(W1, W2)

    print(f"W1 shape: {W1.shape}, norm: {np.linalg.norm(W1):.4f}, q1*: {np.round(q1, 4)}")
    print(f"W2 shape: {W2.shape}, norm: {np.linalg.norm(W2):.4f}, q2*: {np.round(q2, 4)}")
    print("HYPERNEURON TEST PASSED")
