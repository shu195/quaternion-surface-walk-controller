"""Train the 32D quaternion hyperneuron's basis on a synthetic regression task."""

import numpy as np
from quaternion_hyperneuron import MultiQuaternionHyperNet


def main() -> None:
    rng = np.random.default_rng(42)
    in_dim, m, n, n_samples = 16, 64, 64, 100

    net = MultiQuaternionHyperNet(in_dim=in_dim, out_shape=(m, n), n_quaternions=8, seed=7)
    x_cond = rng.normal(size=in_dim)
    V = rng.normal(scale=0.3, size=(n_samples, n))

    # Target matrix generated from a fixed 32D latent + true basis
    _, q32 = net.generate_weights(x_cond)
    B_true = rng.normal(scale=0.2, size=(32, m, n))
    W_true = np.tensordot(q32, B_true, axes=(0, 0))
    Y_target = np.tanh(V @ W_true.T)

    # Trainable basis B (32, m, n)
    B = np.concatenate([cell.basis for cell in net.neurons], axis=0).copy()
    lr = 80.0

    for step in range(1, 201):
        W = np.tensordot(q32, B, axes=(0, 0))
        Z = V @ W.T
        Y_pred = np.tanh(Z)
        diff = Y_pred - Y_target
        loss = float(np.mean(diff ** 2))

        if step in (1, 50, 100, 200):
            print(f"step={step:3d} | mse_loss={loss:.6f}")

        # Backprop through tanh and tensordot
        dZ = (2.0 / diff.size) * diff * (1.0 - Y_pred ** 2)
        dW = dZ.T @ V
        dB = np.einsum("k,mn->kmn", q32, dW)
        B -= lr * dB

    assert loss < 1e-3, f"Loss did not converge: {loss}"
    print("TRAINING CONVERGED")


if __name__ == "__main__":
    main()
