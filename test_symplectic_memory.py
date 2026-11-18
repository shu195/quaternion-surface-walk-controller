"""Test @theorist's Symplectic Quaternion Ring Walk for Continual Memory (2,000 steps)."""

import numpy as np


def qmul_batch(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Batch Hamilton product for (N, 4) quaternions."""
    w1, x1, y1, z1 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def exp_so3(omega: np.ndarray, dt: float) -> np.ndarray:
    """Exact exponential map from angular velocity (N, 3) to unit quaternion (N, 4)."""
    theta = np.linalg.norm(omega, axis=-1, keepdims=True) * dt
    half = 0.5 * theta
    scale = np.where(theta > 1e-8, np.sin(half) / (theta / dt + 1e-12), 0.5 * dt)
    return np.concatenate([np.cos(half), omega * scale], axis=-1)


def ring_torque(q: np.ndarray, kappa: float) -> np.ndarray:
    """Torque tau_i = kappa * imag(conj(q_i) * (q_{i-1} + q_{i+1})) on S^3."""
    q_conj = q * np.array([1.0, -1.0, -1.0, -1.0])
    neighbors = np.roll(q, 1, axis=0) + np.roll(q, -1, axis=0)
    return 0.5 * kappa * qmul_batch(q_conj, neighbors)[:, 1:]


def total_energy(q: np.ndarray, omega: np.ndarray, kappa: float) -> float:
    kinetic = 0.5 * float(np.sum(omega ** 2))
    potential = -kappa * float(np.sum(q * np.roll(q, -1, axis=0)))
    return kinetic + potential


def main() -> None:
    rng = np.random.default_rng(7)
    n_q, kappa, dt, steps = 8, 0.5, 0.05, 2000

    q = rng.normal(size=(n_q, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    omega = rng.normal(scale=0.5, size=(n_q, 3))

    e0 = total_energy(q, omega, kappa)
    energies = []
    max_norm_err = 0.0

    # Symplectic Lie-group Verlet (kick-drift-kick)
    for _ in range(steps):
        omega += 0.5 * dt * ring_torque(q, kappa)
        q = qmul_batch(q, exp_so3(omega, dt))
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        omega += 0.5 * dt * ring_torque(q, kappa)

        energies.append(total_energy(q, omega, kappa))
        max_norm_err = max(max_norm_err, float(np.max(np.abs(np.linalg.norm(q, axis=-1) - 1.0))))

    energies = np.array(energies)
    e_drift = float(np.max(np.abs(energies - e0)) / (abs(e0) + 1e-12))
    print(f"Symplectic 2,000-step walk | E0={e0:.6f} | max_rel_energy_drift={e_drift:.2e} | max_||q||_err={max_norm_err:.2e}")
    assert e_drift < 1e-3, f"Energy drifted: {e_drift}"
    assert max_norm_err < 1e-12
    print("SYMPLECTIC MEMORY TEST PASSED")


if __name__ == "__main__":
    main()
