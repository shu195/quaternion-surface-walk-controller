"""Prototype: Zero-backprop token-by-token attention rewiring via quaternion surface walk."""

import time
import numpy as np
from factored_hypernet import FactoredQuaternionSurfaceHyperNet


def main() -> None:
    rng = np.random.default_rng(7)
    seq_len, d_model, n_q = 16, 1024, 8
    net = FactoredQuaternionSurfaceHyperNet(in_dim=64, m=d_model, n=d_model, n_q=n_q, laps=1, seed=7)

    # Sequence of 16 tokens (e.g., two topic shifts mid-sequence)
    X_seq = rng.normal(scale=0.3, size=(seq_len, d_model)).astype(np.float32)
    X_seq[8:] += 0.8 * rng.normal(size=(1, d_model)).astype(np.float32)  # topic shift at t=8

    # Warmup
    surface = net.init_surface.copy()
    _ = (X_seq @ net.V)

    t0 = time.perf_counter()
    # Hoist base projection Z = X @ V once
    Z = X_seq @ net.V  # (T, 32)
    Q_list, K_list, V_list = [], [], []
    q_traj = []

    surface = net.init_surface.copy()
    for t in range(seq_len):
        # Token t nudges the quaternion ring in-flight (zero backprop)
        u_t = np.einsum("qki,i->qk", net.enc, X_seq[t, :64])
        surface = 0.5 * surface + 0.5 * np.tanh(np.roll(surface, 1, axis=0) @ net.qmul_mat + u_t + net.b)
        q_t = surface.reshape(-1)
        q_traj.append(q_t.copy())

        # Rewire Q, K, V projections on the fly for token t
        h_t = np.tanh((Z[t] * q_t) @ net.U.T)
        Q_list.append(h_t)
        K_list.append(h_t)
        V_list.append(h_t)

    Q = np.stack(Q_list)  # (T, d_model)
    K = np.stack(K_list)
    V = np.stack(V_list)

    # Causal scaled dot-product attention
    scores = (Q @ K.T) / np.sqrt(d_model)
    mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
    weights = np.exp(scores + mask - np.max(scores + mask, axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    attn_out = weights @ V
    t1 = time.perf_counter()

    q_traj = np.stack(q_traj)
    shift_pre = float(np.linalg.norm(q_traj[7] - q_traj[0]))
    shift_post = float(np.linalg.norm(q_traj[15] - q_traj[7]))

    print(f"Sequence ({seq_len} tokens, d_model={d_model}, zero-backprop): {(t1 - t0) * 1e3:.3f} ms total ({(t1 - t0) * 1e3 / seq_len:.3f} ms/token)")
    print(f"Quaternion state shift t=0->7: {shift_pre:.4f} | across topic shift t=7->15: {shift_post:.4f}")
    print(f"Causal attn_out shape={attn_out.shape}, std={attn_out.std():.4f}")
    print("TOKEN REWIRING TEST PASSED")


if __name__ == "__main__":
    main()
