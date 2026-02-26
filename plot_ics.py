import torch
import matplotlib.pyplot as plt

d = torch.load("sample_000.pt", map_location="cpu")
X, Y = d["X"], d["Y"]

print("X shape:", tuple(X.shape))
print("Y shape:", tuple(Y.shape))

# normalize shapes to (T, C, H, W)
X_tc = X[0] if X.ndim == 5 else X
Y_tc = Y[0] if Y.ndim == 5 else Y

t = 0
c = 0
x_img = X_tc[t, c].numpy()
y_img = Y_tc[t, c].numpy()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title(f"X (t={t}, c={c})")
plt.imshow(x_img, origin="lower")
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(1, 2, 2)
plt.title(f"Y (t={t}, c={c})")
plt.imshow(y_img, origin="lower")
plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()