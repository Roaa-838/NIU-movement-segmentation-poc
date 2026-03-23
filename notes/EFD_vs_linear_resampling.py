# notes/EFD_vs_linear_resampling.py
#
# Contour compression experiment: Elliptic Fourier Descriptors vs linear resampling.
#
# Result: EFD reconstruction failed for this contour type — RMSE was 6.27px
# regardless of K (16, 32, or 64). Linear resampling N=128 gives 0.56px RMSE.
# See proposal for implications: linear resampling with arc-length
# parameterization is the correct approach for NaN-padded contour storage.
#
# Keeping this file as an honest record of what I tested and what didn't work.

import numpy as np
from pyefd import elliptic_fourier_descriptors, reconstruct_contour


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


# Synthetic worm-like ellipse with slight noise — representative of OCTRON output.
t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
noise = np.random.default_rng(42).normal(0, 0.5, (400, 2))
contour = np.column_stack([80 * np.cos(t) + 20 * np.cos(2 * t), 30 * np.sin(t)]) + noise

print("EFD (Elliptic Fourier Descriptors):")
for K in [16, 32, 64]:
    coeffs  = elliptic_fourier_descriptors(contour, order=K)
    recon   = reconstruct_contour(coeffs, num_points=len(contour))
    storage = K * 4 * 4   # (K, 4) float32
    print(f"  K={K:3d}: RMSE={rmse(contour, recon):.4f}px  storage={storage}B")

print("\nLinear resampling (vertex-index, Sparsh's approach):")
for N in [64, 128]:
    idx       = np.round(np.linspace(0, len(contour) - 1, N)).astype(int)
    resampled = contour[idx]
    x_r = np.interp(np.linspace(0, 1, 400), np.linspace(0, 1, N), resampled[:, 0])
    y_r = np.interp(np.linspace(0, 1, 400), np.linspace(0, 1, N), resampled[:, 1])
    recon_lin = np.column_stack([x_r, y_r])
    storage   = N * 2 * 4   # (N, 2) float32
    print(f"  N={N:3d}: RMSE={rmse(contour, recon_lin):.4f}px  storage={storage}B")

print(f"\nRaw NaN-padded (500 pts): {500 * 2 * 4}B")
print("\nConclusion: linear N=128 at 0.56px RMSE beats EFD on this contour type.")