import utils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# algos to implement: SGD, SGD+SLS, SGD+SPS, SVRG, SGD w constant step-size under interpolation assumption
# might be worth it to implement regularied logistic regression as well?

NUM_ITERS = 100000
BATCH_SIZE = 128
EPS = 1e-5

np.random.seed(42)

# def get_logistic_smoothness_unreg(X):
#     n = X.shape[0]
#     smax = np.linalg.svd(X, compute_uv=False)[0]
#     return (smax**2) / (4.0 * n)

def stochastic_logistic_grad(w, batch_indices, X, y):
    X_batch = X[batch_indices,:]
    y_batch = y[batch_indices]
    return utils.logistic_grad(w, X_batch, y_batch)

def stochastic_logistic_loss(w, batch_indices, X, y):
    X_batch = X[batch_indices,:]
    y_batch = y[batch_indices]
    return utils.logistic_loss(w, X_batch, y_batch)

# def sgd(w_start, T=NUM_ITERS, eps=EPS): # note: T = n here i.e. in one pass we use one data point to calculte the stochastic gradient

#     # random permutation of data point indices for sampling w/out replacement
#     inds = np.random.permutation(np.arange(n))

#     eta = 1e-5

#     w = w_start.copy()

#     norms = []
#     losses = []
#     dists = []

#     for k in tqdm(range(T)):

#         norms.append(np.linalg.norm(w, 2))
#         losses.append(utils.logistic_loss(w,X,y))
#         dists.append(utils.direction_distance(w,w_star))
        
#         batch_indices = np.random.randint(low=0, high=n, size=BATCH_SIZE)

#         curr_grad_i = stochastic_logistic_grad(w, batch_indices)

#         w = w - eta * curr_grad_i

#     return w, norms, losses, dists

# def svrg(w_start, T=n, eps=EPS):

#     # random permutation of data point indices for sampling w/out replacement
#     inds = np.random.permutation(np.arange(n))

#     w = w_start.copy()
#     v = w.copy()
#     eta = 1.0 / (6*L)
#     p = 1.0 / n
#     grad_v = utils.logistic_grad(v,X,y)

#     for k in range(T):
#         rand_ind = inds[k]

#         g_k = stochastic_logistic_grad(w,rand_ind) - stochastic_logistic_grad(v, rand_ind) + grad_v

#         w = w - eta * g_k

#         if np.random.rand() < p:
#             v = w.copy()
#             grad_v = utils.logistic_grad(v,X,y)
    
#     return w


def sgd_step(w, sgrad, lr):
    w_new = w - lr * sgrad
    return w_new

def stochastic_armijo_ls(w, batch_indices, X, y, eta_max=1, c=0.5, beta=0.99):
    f_i = stochastic_logistic_loss(w, batch_indices, X, y)
    sgrad_w = stochastic_logistic_grad(w, batch_indices, X, y)
    sgrad_normsq = np.linalg.norm(sgrad_w, 2)**2
        
    eta = eta_max

    while stochastic_logistic_loss(w - eta * sgrad_w, batch_indices, X, y) > f_i - c*eta*sgrad_normsq:
        eta = eta * beta     

    return eta

    
def stochastic_polyak_stepsize(w, batch_indices, f_star, X, y, eta_max = 1, c=0.5):
    f_ik = stochastic_logistic_loss(w, batch_indices, X, y)
    nabla_f_ik = stochastic_logistic_grad(w, batch_indices, X, y)
    grad_normsq = c * (np.linalg.norm(nabla_f_ik, 2)**2)

    return np.min(
        [
            (f_ik-f_star)/grad_normsq,
            eta_max
        ]
    )

def svrg_step(w, v, sgrad_w, sgrad_v, grad_v, lr, p):
    g_k = sgrad_w - sgrad_v + grad_v
    w_new = w - lr * g_k
    v_new = v
        
    if np.random.rand() < p:
        v_new = w.copy() # note: as per both lecture notes and slides, snapshot is updated with PREVIOUS iterate
        
    return w_new, v_new

def main():
    print("Generating dataset...")
    X, y, w_star = utils.make_soudry_dataset(
        n=20000,
        d=500,
        margin=0.1,
        sigma=3.0,
    )

    f_star = utils.logistic_loss(w_star, X, y) # needed for SPS
    # f_star = 0.0    # if data is linearly separable

    n,d = X.shape

    sigma_max = np.linalg.norm(X, ord=2)
    lr = 1.0 / (sigma_max ** 2) # might wanna make this smaller later

    rho = 0.1
    print("Learning rate:", lr)

    names = ["SGD", "SGD+SLS", "SGD+SPS", "SVRG"]

    steps_dict = {k: [] for k in names}
    norms = {k: [] for k in names}
    losses = {k: [] for k in names}
    angles = {k: [] for k in names}
    dists = {k: [] for k in names}

    ws = {
        "SGD": np.zeros(d),
        "SGD+SLS": np.zeros(d),
        "SGD+SPS": np.zeros(d),
        "SVRG": np.zeros(d),
    }

    record_steps = np.unique(np.logspace(0, np.log10(NUM_ITERS), 400).astype(int))
    step_idx = 0

    print("Training...")


    # setup for SVRG
    v_svrg = ws["SVRG"].copy()
    grad_v_svrg = utils.logistic_grad(v_svrg,X,y)
    p_svrg = (1.0/n)

    for t in tqdm(range(1, NUM_ITERS + 1)):

        # compute each optimizer's gradient from its own weights

        batch_indices = np.random.randint(low=0, high=n, size=BATCH_SIZE)

        # update step-sizes
        eta_sgd = lr/np.sqrt(t)
        eta_sls = stochastic_armijo_ls(ws["SGD+SLS"], batch_indices, X, y)
        eta_sps = stochastic_polyak_stepsize(ws["SGD+SPS"], batch_indices, f_star, X, y)

        # get SGD gradient
        grad_sgd = stochastic_logistic_grad(ws["SGD"], batch_indices, X, y)
        grad_sls = stochastic_logistic_grad(ws["SGD+SLS"], batch_indices, X, y)
        grad_sps = stochastic_logistic_grad(ws["SGD+SPS"], batch_indices, X, y)

        # svrg gradients
        sgrad_w_svrg = stochastic_logistic_grad(ws["SVRG"], batch_indices, X, y)
        sgrad_v_svrg = stochastic_logistic_grad(v_svrg, batch_indices, X, y)

        # SGD updates w and step-size
        ws["SGD"] = sgd_step(ws["SGD"], grad_sgd, eta_sgd)

        # SGD+SLS updates w and step-size
        ws["SGD+SLS"] = sgd_step(ws["SGD+SLS"], grad_sls, eta_sls)

        # SGD+SPS
        ws["SGD+SPS"] = sgd_step(ws["SGD+SPS"], grad_sps, eta_sps)

        # SVRG updates w, v, and grad(v)
        ws["SVRG"], v_new_svrg = svrg_step(ws["SVRG"], v_svrg, sgrad_w_svrg, sgrad_v_svrg, grad_v_svrg, lr, p_svrg)

        if not np.array_equal(v_svrg, v_new_svrg):
            grad_v_svrg = utils.logistic_grad(v_new_svrg, X, y)
        v_svrg = v_new_svrg

        if t == record_steps[step_idx]:
            for name in names:
                wcur = ws[name]
                steps_dict[name].append(t)
                norms[name].append(np.linalg.norm(wcur))
                losses[name].append(utils.logistic_loss(wcur, X, y))
                angles[name].append(utils.angle_between(wcur, w_star))
                dists[name].append(utils.direction_distance(wcur, w_star))

            step_idx += 1
            if step_idx >= len(record_steps):
                break

    # ===========================================================
    # Plots
    # ===========================================================

    # Norm plot
    plt.figure()
    for name in names:
        plt.plot(steps_dict[name], norms[name], label=name)
    plt.xscale("log")
    plt.title("||w(t)|| growth")
    plt.xlabel("iteration")
    plt.ylabel("norm")
    plt.legend()
    plt.grid()
    plt.savefig("norm_compare.png")

    # Loss plot
    plt.figure()
    for name in names:
        plt.plot(steps_dict[name], losses[name], label=name)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Logistic loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    plt.savefig("loss_compare.png")

    # Angle plot
    plt.figure()
    for name in names:
        plt.plot(steps_dict[name], angles[name], label=name)
    plt.xscale("log")
    plt.title("Angle between w(t) and w*")
    plt.xlabel("iteration")
    plt.ylabel("angle (radians)")
    plt.legend()
    plt.grid()
    plt.savefig("angle_compare.png")

    # Distance plot
    plt.figure()
    for name in names:
        plt.plot(steps_dict[name], dists[name], label=name)
    plt.xscale("log")
    plt.title("Direction distance")
    plt.xlabel("iteration")
    plt.ylabel("||w_hat - w_star_hat||")
    plt.legend()
    plt.grid()
    plt.savefig("distance_compare.png")

    print("Done. Saved norm_compare.png, loss_compare.png, angle_compare.png, distance_compare.png")


if __name__ == "__main__":
    main()


