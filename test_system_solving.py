"""Test whether a single quaternion neuron can solve a system of equations."""

import numpy as np
from scipy.optimize import root
from quaternion_neuron import QuaternionNeuron, qmul, simulate_multi_order_ode


def test_coupled_4d_nonlinear_system() -> None:
    """A single quaternion neuron ODE solves a coupled 4-equation nonlinear system."""
    weight = np.array([0.4, 0.25, -0.3, 0.15])
    bias = np.array([0.1, -0.05, 0.08, -0.04])
    u_const = np.array([0.5, -0.3, 0.2, 0.1])
    c0 = 1.5
    neuron = QuaternionNeuron(weight=weight, bias=bias)

    # Solve F(q) = c0 * q - tanh(W * q + u + b) = 0 directly with scipy.optimize.root
    f_sys = lambda q: c0 * q - neuron.forward(q, u_const)
    root_sol = root(f_sys, np.zeros(4), tol=1e-12)
    assert root_sol.success

    # Solve via 2nd-order quaternion neuron ODE
    y0 = np.zeros(8, dtype=np.float64)
    ode_sol = simulate_multi_order_ode(
        neuron=neuron, coeffs=[c0, 1.8], y0=y0, t_span=(0.0, 60.0), u_fn=lambda _t: u_const
    )
    q_ode = ode_sol.y[:4, -1]
    err = float(np.linalg.norm(q_ode - root_sol.x))
    print(f"4D nonlinear system | ODE vs scipy.root error: {err:.2e}")
    assert err < 1e-8


def test_arbitrary_4x4_linear_system_limit() -> None:
    """H_0 check: left-multiplication L(W) only spans 4 of 16 dims of R^(4x4)."""
    basis_mats = []
    for idx in range(4):
        e = np.zeros(4)
        e[idx] = 1.0
        # Columns of L(e) are e * e_j
        mat = np.column_stack([qmul(e, np.eye(4)[j]) for j in range(4)])
        basis_mats.append(mat.reshape(-1))
    basis = np.column_stack(basis_mats)  # 16 x 4

    rng = np.random.default_rng(42)
    A_arbitrary = rng.normal(size=(4, 4)).reshape(-1)
    proj = basis @ np.linalg.lstsq(basis, A_arbitrary, rcond=None)[0]
    rel_err = float(np.linalg.norm(A_arbitrary - proj) / np.linalg.norm(A_arbitrary))
    print(f"Arbitrary 4x4 matrix projection error onto L(H): {rel_err:.4f} (rank={np.linalg.matrix_rank(basis)})")
    assert rel_err > 0.1


if __name__ == "__main__":
    test_coupled_4d_nonlinear_system()
    test_arbitrary_4x4_linear_system_limit()
    print("SYSTEM SOLVING TESTS PASSED")
