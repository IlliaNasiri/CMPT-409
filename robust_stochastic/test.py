import numpy as np

X = np.load("X.npy")
y = np.load("y.npy")
w_star = np.load("w_star.npy")


print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")
print(f"w_star: {w_star}")