"""Test in-place closed-loop walk over an n-dimensional neuron surface with per-step drive u[i]."""

import numpy as np
from quaternion_neuron import QuaternionNeuron


def main() -> None:
    rng = np.random.default_rng(7)
    n_points = 32
    neuron = QuaternionNeuron(
        weight=np.array([0.4, 0.25, -0.3, 0.15]),
        bias=np.array([0.05, -0.02, 0.03, -0.01]),
    )

    surface = rng.normal(scale=0.3, size=(n_points, 4))
    # Index/data-dependent drive u[i] so each point on the ring has its own target
    u_drive = rng.normal(scale=0.4, size=(n_points, 4))

    for lap in range(1, 21):
        prev = surface.copy()
        for idx in range(n_points):
            prev_idx = (idx - 1) % n_points
            surface[idx] = 0.5 * surface[idx] + 0.5 * neuron.forward(surface[prev_idx], u=u_drive[idx])
        drift = float(np.linalg.norm(surface - prev))
        if lap in (1, 5, 10, 20):
            print(f"lap {lap:2d} | drift={drift:.2e}")

    W = surface
    rank = int(np.linalg.matrix_rank(W))
    V = rng.normal(size=(50, 4))
    Y = np.tanh(V @ W.T)

    assert rank == 4, f"Expected full rank 4, got {rank}"
    print(f"W shape={W.shape}, rank={rank}/4, norm={np.linalg.norm(W):.4f}")
    print(f"Y shape={Y.shape}, min={Y.min():.4f}, max={Y.max():.4f}, std={Y.std():.4f}")
    print("DRIVEN SURFACE WALK PASSED")


if __name__ == "__main__":
    main()
