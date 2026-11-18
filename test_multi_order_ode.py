"""Tests for quaternion neuron algebra and multi-order ODE (k = 1, 2, 3) dynamics."""

import numpy as np
from quaternion_neuron import QuaternionNeuron, qmul, qnorm, simulate_multi_order_ode


def test_hamilton_algebra() -> None:
    i = np.array([0.0, 1.0, 0.0, 0.0])
    j = np.array([0.0, 0.0, 1.0, 0.0])
    k = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(qmul(i, j), k)
    assert np.allclose(qmul(j, i), -k)
    p = np.array([1.2, -0.5, 0.8, -0.3])
    q = np.array([-0.4, 0.9, -0.2, 1.1])
    assert np.isclose(qnorm(qmul(p, q)), qnorm(p) * qnorm(q), atol=1e-12)


def test_multi_order_ode_single_neuron() -> None:
    weight = np.array([0.4, 0.25, -0.3, 0.15])
    bias = np.array([0.1, -0.05, 0.08, -0.04])
    u_const = np.array([0.5, -0.3, 0.2, 0.1])
    neuron = QuaternionNeuron(weight=weight, bias=bias)
    u_fn = lambda _t: u_const

    configs = [
        (1, [1.5], "1st-order"),
        (2, [1.5, 1.8], "2nd-order"),
        (3, [1.5, 2.2, 2.5], "3rd-order"),
    ]

    equilibria = []
    for order, coeffs, label in configs:
        y0 = np.zeros(4 * order, dtype=np.float64)
        y0[:4] = np.array([0.2, -0.1, 0.15, -0.05])
        sol = simulate_multi_order_ode(
            neuron=neuron,
            coeffs=coeffs,
            y0=y0,
            t_span=(0.0, 60.0),
            u_fn=u_fn,
            n_eval=500,
        )
        q_traj = sol.y[:4, :]
        q_end = q_traj[:, -1]
        residual = np.linalg.norm(coeffs[0] * q_end - neuron.forward(q_end, u_const))
        max_norm = float(np.max(np.linalg.norm(q_traj, axis=0)))
        deriv_end = float(np.linalg.norm(sol.y[4:, -1])) if order > 1 else 0.0
        print(
            f"{label}: q*={np.round(q_end, 6)} | "
            f"fp_residual={residual:.2e} | deriv_end={deriv_end:.2e} | max_||q||={max_norm:.4f}"
        )
        assert residual < 1e-6, f"{label} did not reach fixed point: residual={residual}"
        assert np.isfinite(max_norm) and max_norm < 5.0
        equilibria.append(q_end)

    for idx in range(1, len(equilibria)):
        diff = float(np.linalg.norm(equilibria[idx] - equilibria[0]))
        assert diff < 1e-6, f"Equilibrium mismatch between order 1 and {idx + 1}: {diff}"


if __name__ == "__main__":
    test_hamilton_algebra()
    test_multi_order_ode_single_neuron()
    print("ALL TESTS PASSED")
