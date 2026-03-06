# viz_samples.py
import numpy as np
import matplotlib.pyplot as plt

x = np.load("samples.npy")  # (n,64,64)

n = x.shape[0]
cols = 4
rows = (n + cols - 1) // cols

plt.figure(figsize=(12, 6))
for i in range(n):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(x[i], origin="lower")
    plt.axis("off")
plt.tight_layout()
plt.savefig("samples.png")