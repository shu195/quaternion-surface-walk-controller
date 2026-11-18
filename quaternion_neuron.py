"""Quaternion neuron and multi-order ODE dynamics in H (R^4)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def qmul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product p * q for quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def qconj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate [w, -x, -y, -z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def qnorm(q: np.ndarray) -> float:
    """Euclidean norm ||q|| in H."""
    return float(np.linalg.norm(q))


def split_tanh(q: np.ndarray) -> np.ndarray:
    """Component-wise split-tanh activation."""
    return np.tanh(q)


class QuaternionNeuron:
    """Single quaternion-valued neuron with Hamilton weight product."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        self.weight = np.asarray(weight, dtype=np.float64)
        self.bias = np.asarray(bias, dtype=np.float64)
        if self.weight.shape != (4,) or self.bias.shape != (4,):
            raise ValueError("weight and bias must have shape (4,)")

    def forward(self, q: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        """Compute sigma(W * q + u + b)."""
        pre = qmul(self.weight, q) + self.bias
        if u is not None:
            pre = pre + u
        return split_tanh(pre)


def simulate_multi_order_ode(
    neuron: QuaternionNeuron,
    coeffs: list[float] | np.ndarray,
    y0: np.ndarray,
    t_span: tuple[float, float],
    u_fn=None,
    n_eval: int = 500,
    rtol: float = 1e-8,
    atol: float = 1e-10,
):
    """Integrate q^(k) + sum_{j=0}^{k-1} c_j q^(j) = neuron.forward(q, u(t)).

    coeffs: [c_0, c_1, ..., c_{k-1}] where k = len(coeffs) >= 1.
    y0: initial state of shape (4 * k,) stacked as [q, q', ..., q^(k-1)].
    """
    coeffs_arr = np.asarray(coeffs, dtype=np.float64)
    order = len(coeffs_arr)
    if order < 1:
        raise ValueError("ODE order must be >= 1")
    y0_arr = np.asarray(y0, dtype=np.float64)
    if y0_arr.shape != (4 * order,):
        raise ValueError(f"y0 must have shape ({4 * order},), got {y0_arr.shape}")

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        states = y.reshape(order, 4)
        q = states[0]
        u = np.zeros(4, dtype=np.float64) if u_fn is None else np.asarray(u_fn(t), dtype=np.float64)
        drive = neuron.forward(q, u)
        highest = drive - np.tensordot(coeffs_arr, states, axes=(0, 0))
        dydt = np.empty_like(states)
        if order > 1:
            dydt[:-1] = states[1:]
        dydt[-1] = highest
        return dydt.reshape(-1)

    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    sol = solve_ivp(rhs, t_span, y0_arr, method="RK45", t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    return sol
