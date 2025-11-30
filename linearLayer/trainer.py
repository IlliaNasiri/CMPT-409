import torch
import numpy as np
from tqdm import tqdm
from .optimizers import *

def run_training(
    X_np, y_np,
    optimizers,
    learning_rates,
    k=20,
    total_iters=50000,
    debug=True,
    device="cpu"
):

    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)

    N, D = X.shape
    results = {}

    record_steps = np.unique(
        np.concatenate(([1], np.logspace(0, np.log10(total_iters), 200))).astype(int)
    )

    for lr in learning_rates:
        results[lr] = {}

        for name, step_fn in optimizers.items():

            model = TwoLayerNet(D, k).to(device)

            hist = {"steps": [], "loss": [], "err": []}
            rec_idx = 0

            if debug:
                print(f"\n=== RUNNING {name} @ lr={lr} ===")

            for t in tqdm(range(1, total_iters+1), leave=False):

                try:
                    model = step_fn(model, X, y, lr)
                except Exception as e:
                    print(f"Optimizer crashed: {name} lr={lr} step={t}")
                    print(repr(e))
                    break

                if rec_idx < len(record_steps) and t == record_steps[rec_idx]:
                    loss = exponential_loss_2layer_torch(model, X, y).item()
                    err  = error_rate_2layer_torch(model, X, y).item()

                    hist["steps"].append(t)
                    hist["loss"].append(loss)
                    hist["err"].append(err)

                    rec_idx += 1

            results[lr][name] = hist

    return results
