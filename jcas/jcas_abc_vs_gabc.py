import numpy as np
import random
import matplotlib.pyplot as plt

from jcas_abc import ABC, beamforming_objective
from jcas_gabc import GABC

# ========================
# Setup JCAS
# ========================
M = 4
K = 50

theta = np.linspace(-np.pi/2, np.pi/2, K)

A = np.exp(1j * np.outer(np.sin(theta), np.arange(M)))
v = np.exp(-(theta / 0.2)**2)

D = np.eye(K)
for k in range(K):
    D[k, k] = 5.0 if abs(theta[k]) < 0.15 else 1.0

bounds = [(-5, 5)] * (2 * M)

# ========================
# Chạy ABC
# ========================
abc = ABC(
    lambda x: beamforming_objective(x, A, v, D),
    bounds,
    pop_size=30,
    max_iter=200,
    limit=40
)
best_pos_abc, best_val_abc = abc.run()
history_abc = abc.history[:100]

# ========================
# Chạy GABC
# ========================
gabc = GABC(
    lambda x: beamforming_objective(x, A, v, D),
    bounds,
    pop_size=30,
    max_iter=200,
    limit=40,
    C=1.5
)
best_pos_gabc, best_val_gabc = gabc.run()
history_gabc = gabc.history[:100]

# VẼ ĐỒ THỊ HỘI TỤ
plt.figure(figsize=(10, 6))
plt.plot(history_abc, marker='o', markersize=3, linewidth=1, label='ABC')
plt.plot(history_gabc, marker='s', markersize=3, linewidth=1, label='g-best ABC')
plt.xlabel("Iteration")
plt.ylabel("Best LS Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
