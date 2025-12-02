import numpy as np
from tqdm import tqdm
from .metrics import (
    get_angle, get_direction_distance,
    exponential_loss, get_error_rate
)

from sklearn.model_selection import train_test_split

def run_training(
    X, y, w_star,
    optimizers,
    learning_rates,
    total_iters=100_000,
    debug=True  # <--- debug mode enabled by default
):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    # ALWAYS log t=1 (important when total_iters is small)
    record_steps = np.unique(
        np.concatenate(([1], np.logspace(0, np.log10(total_iters), 200))).astype(int)
    )

    if debug:
        print("\n[DEBUG] record_steps[:10] =", record_steps[:10])

    for lr in learning_rates:
        results[lr] = {}

        if debug:
            print(f"\n=== Learning Rate Sweep: lr={lr} ===")

        for name, step_fn in optimizers.items():

            if debug:
                print(f"\n--- Starting optimizer '{name}' @ lr={lr} ---")

            D = X.shape[1]
            w = np.random.randn(D) * 1e-6   # fresh init

            hist = {"steps": [], "dist": [], "angle": [], "loss": [], "test_err": [], "test_loss": [] }
            rec_idx = 0

            # ------------------------------
            # MAIN TRAINING LOOP
            # ------------------------------
            for t in tqdm(range(1, total_iters + 1), leave=False, desc=f"{name} lr={lr}"):

                try:
                    # UPDATE STEP
                    w = step_fn(w, X_train, y_train, lr)

                    # DETECT silent numeric issues
                    if np.isnan(w).any() or np.isinf(w).any():
                        raise FloatingPointError(
                            f"NaN/Inf detected in '{name}' at step {t} (lr={lr})"
                        )

                except Exception as e:
                    print("\n" + "=" * 80)
                    print(f"🔥 ERROR in optimizer '{name}' | LR={lr} | STEP={t}")
                    print("Exception:", repr(e))
                    print("=" * 80 + "\n")
                    break

                # LOGGING POINTS
                if rec_idx < len(record_steps) and t == record_steps[rec_idx]:
                    hist["steps"].append(t)
                    hist["dist"].append(get_direction_distance(w, w_star))
                    hist["angle"].append(get_angle(w, w_star))
                    hist["loss"].append(exponential_loss(w, X_train, y_train))
                    hist["test_loss"].append(exponential_loss(w, X_test, y_test))
                    err_val = get_error_rate(w, X_test, y_test)
                    hist["test_err"] = err_val

                    rec_idx += 1

            # -----------------------------------------
            # FINISHED TRAINING THE OPTIMIZER
            # -----------------------------------------
            if len(hist["steps"]) == 0:
                print(f"⚠ WARNING: Optimizer '{name}' @ lr={lr} "
                      f"produced NO recorded metrics (maybe crashed early).")

            if debug:
                print(f"[DEBUG] Done: {name}, recorded {len(hist['steps'])} points.")

            results[lr][name] = hist

    return results
