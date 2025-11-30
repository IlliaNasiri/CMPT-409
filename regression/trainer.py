import numpy as np
from tqdm import tqdm
from .metrics import (
    get_angle, get_direction_distance,
    exponential_loss, get_error_rate
)

def run_training(
    X, y, w_star,
    optimizers,
    learning_rates,
    total_iters=100_000,
    debug=True  # <--- debug mode enabled by default
):

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

            hist = {"steps": [], "dist": [], "angle": [], "loss": [], "err": []}
            rec_idx = 0

            # ------------------------------
            # MAIN TRAINING LOOP
            # ------------------------------
            for t in tqdm(range(1, total_iters + 1), leave=False, desc=f"{name} lr={lr}"):

                try:
                    # UPDATE STEP
                    w = step_fn(w, X, y, lr)

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
                    hist["loss"].append(exponential_loss(w, X, y))
                    err_val = get_error_rate(w, X, y)
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
