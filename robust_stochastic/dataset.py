import numpy as np
from sklearn.model_selection import train_test_split

print(f"CAUTION: the generated w_star is wrong. Use data_vizzy.py's get_empirical_max_margin() function to find w_star for the generated data.")
print(f"CAUTION: the test data is large, ask Arya to share it with you.")

def make_soudry_dataset_old(n=200, d=5000, margin=0.1, sigma=3.0):
    """
    Original single-split generator (kept here for reference/compat).
    Usually produces a linearly separable sample when d >> n.
    """
    v = np.ones(d) / np.sqrt(d)
    n2 = n // 2

    noise_pos = sigma * np.random.randn(n2, d)
    noise_neg = sigma * np.random.randn(n2, d)

    X_pos = margin * v + noise_pos
    X_neg = -margin * v + noise_neg

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n2), -np.ones(n2)])

    perm = np.random.permutation(n)
    return X[perm], y[perm], v


# def make_soudry_dataset(
#     n=200,
#     d=5000,
#     margin=0.1,
#     sigma=3.0,
#     n_test=10_000,
#     seed=None,
# ):
#     """
#     New version: returns train + test from the same Soudry-style distribution.

#     If seed is None:
#         - Train split is generated EXACTLY like make_soudry_dataset_old()
#           (np.random.randn + np.random.permutation), so you keep the old
#           “usually separable” behavior.
#         - Test split is generated with an independent RNG.

#     If seed is not None:
#         - Train uses RNG(seed), test uses RNG(seed+1), so both are reproducible.
#     """

#     assert n % 2 == 0, "n must be even (half positive, half negative)."
#     assert n_test % 2 == 0, "n_test must be even (half positive, half negative)."

#     # max-margin direction (same as v in the old function)
#     w_star = np.ones(d) / np.sqrt(d)

#     def _generate_split(n_points, margin, sigma, w_star, rng=None, use_legacy=False):
#         n2 = n_points // 2

#         if use_legacy or rng is None:
#             # EXACTLY the old behavior: global np.random
#             noise_pos = sigma * np.random.randn(n2, d)
#             noise_neg = sigma * np.random.randn(n2, d)

#             X_pos = margin * w_star + noise_pos
#             X_neg = -margin * w_star + noise_neg

#             X = np.vstack([X_pos, X_neg])
#             y = np.concatenate([np.ones(n2), -np.ones(n2)])

#             perm = np.random.permutation(n_points)
#             return X[perm], y[perm]
#         else:
#             # Reproducible variant using a passed-in Generator
#             noise_pos = sigma * rng.standard_normal(size=(n2, d))
#             noise_neg = sigma * rng.standard_normal(size=(n2, d))

#             X_pos = margin * w_star + noise_pos
#             X_neg = -margin * w_star + noise_neg

#             X = np.vstack([X_pos, X_neg])
#             y = np.concatenate([np.ones(n2), -np.ones(n2)])

#             perm = rng.permutation(n_points)
#             return X[perm], y[perm]

#     if seed is None:
#         # Train: old behavior (global RNG) → “usually separable” like before
#         X_train, y_train = _generate_split(
#             n, margin, sigma, w_star, rng=None, use_legacy=True
#         )
#         # Test: independent generator, doesn't affect training distribution
#         rng_test = np.random.default_rng()
#         X_test, y_test = _generate_split(
#             n_test, margin, sigma, w_star, rng=rng_test, use_legacy=False
#         )
#     else:
#         # Fully reproducible train + test
#         rng_train = np.random.default_rng(seed)
#         rng_test = np.random.default_rng(seed + 1)

#         X_train, y_train = _generate_split(
#             n, margin, sigma, w_star, rng=rng_train, use_legacy=False
#         )
#         X_test, y_test = _generate_split(
#             n_test, margin, sigma, w_star, rng=rng_test, use_legacy=False
#         )

#     return X_train, y_train, X_test, y_test, w_star

# X_train, y_train, X_test, y_test, w_star = make_soudry_dataset(
#     n=200, d=5000, margin=0.1, sigma=3.0, n_test=10_000, seed=None
# )

# np.save("X_train.npy", X_train)
# np.save("y_train.npy", y_train)
# np.save("X_test.npy", X_test)
# np.save("y_test.npy", y_test)
# np.save("w_star.npy", w_star)


def make_soudry_dataset(n=200, d=5000, margin=1, sigma=0.3):
    v = np.ones(d) / np.sqrt(d)
    n2 = n // 2

    noise_pos = sigma * np.random.randn(n2, d)
    noise_neg = sigma * np.random.randn(n2, d)

    X_pos = margin * v + noise_pos
    X_neg = -margin * v + noise_neg

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n2), -np.ones(n2)])

    perm = np.random.permutation(n)
    return X[perm], y[perm], v


if __name__ == "__main__":

    # --- Use your existing function exactly as written ---
    X, y, w_star = make_soudry_dataset(n=200, d=5000, margin=1.0, sigma=0.3)

    # --- Split using sklearn (stratified to preserve class balance) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,          # adjust as needed
        random_state=42,        # for reproducibility
        shuffle=True,
        stratify=y,
    )

    # --- Save everything as .npy files ---
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

    print("Saved: X_train.npy, y_train.npy, X_test.npy, y_test.npy")


