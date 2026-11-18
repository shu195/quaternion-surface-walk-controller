"""Train the hyperneuron's OWN internal parameters (W_cell, b_cell) with basis B frozen."""

import numpy as np
from quaternion_hyperneuron import MultiQuaternionHyperNet


def main() -> None:
    rng = np.random.default_rng(42)
    in_dim, m, n, n_samples = 16, 64, 64, 100

    net = MultiQuaternionHyperNet(in_dim=in_dim, out_shape=(m, n), n_quaternions=8, seed=7)
    x_cond = rng.normal(size=in_dim)
    V = rng.normal(scale=0.3, size=(n_samples, n))

    # Frozen basis B
    B = np.concatenate([cell.basis for cell in net.neurons], axis=0).copy()

    # Target generated from a different internal state
    q_target = rng.uniform(-0.5, 0.5, size=32)
    W_true = np.tensordot(q_target, B, axes=(0, 0))
    Y_target = np.tanh(V @ W_true.T)

    lr = 0.5
    eps = 1e-4

    for step in range(1, 51):
        W, q32 = net.generate_weights(x_cond, t_end=25.0)
        Y_pred = np.tanh(V @ W.T)
        diff = Y_pred - Y_target
        loss = float(np.mean(diff ** 2))

        if step in (1, 10, 25, 50):
            print(f"step={step:2d} | mse_loss={loss:.6f}")

        # Gradient wrt W_out and q32
        dZ = (2.0 / diff.size) * diff * (1.0 - Y_pred ** 2)
        dW = dZ.T @ V
        dq = np.tensordot(B, dW, axes=([1, 2], [0, 1]))  # (32,)

        # Update each quaternion neuron's internal weight & bias via finite-diff sensitivity
        for i, cell in enumerate(net.neurons):
            q_base = q32[4 * i : 4 * (i + 1)]
            dq_i = dq[4 * i : 4 * (i + 1)]

            for attr in ("weight", "bias"):
                param = getattr(cell.neuron, attr)
                grad = np.zeros(4)
                for j in range(4):
                    param[j] += eps
                    _, q_pert = cell.generate_weights(x_cond, t_end=25.0)
                    param[j] -= eps
                    grad[j] = float(np.dot(dq_i, (q_pert - q_base) / eps))
                param -= lr * grad

    print(f"final_loss={loss:.6f}")


if __name__ == "__main__":
    main()
