import torch


def make_torch_adam_step(D, lr):
    w_torch = torch.zeros(D, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w_torch], lr=lr)

    def step(w_np, X, y, lr_unused=None):
        nonlocal w_torch, opt
        with torch.no_grad():
            w_torch[:] = torch.from_numpy(w_np).float()
        w_torch.requires_grad_(True)

        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).float()

        margins = y_t * (X_t @ w_torch)
        loss = torch.mean(torch.exp(-torch.clamp(margins, min=-50)))

        opt.zero_grad()
        loss.backward()
        opt.step()

        return w_torch.detach().numpy()

    return step



def make_torch_adagrad_step(D, lr, eps=1e-8):
    w_torch = torch.zeros(D, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adagrad([w_torch], lr=lr, eps=eps)

    def step(w_np, X, y, lr_unused=None):
        nonlocal w_torch, opt
        with torch.no_grad():
            w_torch[:] = torch.from_numpy(w_np).float()
        w_torch.requires_grad_(True)

        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).float()

        margins = y_t * (X_t @ w_torch)
        loss = torch.mean(torch.exp(-torch.clamp(margins, min=-50)))

        opt.zero_grad()
        loss.backward()
        opt.step()

        return w_torch.detach().numpy()

    return step





def make_torch_sam_adam_step(D, lr, rho=0.05):
    """
    SAM-Adam: SAM outer loop + Adam inner step at w_adv
    """
    # Internal Adam state
    w_torch = torch.zeros(D, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w_torch], lr=lr)

    def step(w_np, X, y, lr_unused=None):
        nonlocal w_torch, opt

        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).float()


        w = torch.from_numpy(w_np).float().requires_grad_(True)

        margins = y_t * (X_t @ w)
        loss = torch.mean(torch.exp(-torch.clamp(margins, min=-50)))
        grad = torch.autograd.grad(loss, w)[0]


        g_norm = grad.norm() + 1e-12
        w_adv = (w + rho * grad / g_norm).detach()


        with torch.no_grad():
            w_torch[:] = w_adv  
        w_torch.requires_grad_(True)

        margins_adv = y_t * (X_t @ w_torch)
        loss_adv = torch.mean(torch.exp(-torch.clamp(margins_adv, min=-50)))

        opt.zero_grad()
        loss_adv.backward()
        opt.step()

        return w_torch.detach().numpy()

    return step




def make_torch_sam_adagrad_step(D, lr, rho=0.05, eps=1e-8):

    w_torch = torch.zeros(D, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adagrad([w_torch], lr=lr, eps=eps)

    def step(w_np, X, y, lr_unused=None):
        nonlocal w_torch, opt

        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).float()

        w = torch.from_numpy(w_np).float().requires_grad_(True)

        margins = y_t * (X_t @ w)
        loss = torch.mean(torch.exp(-torch.clamp(margins, min=-50)))
        grad = torch.autograd.grad(loss, w)[0]

        g_norm = grad.norm() + 1e-12
        w_adv = (w + rho * grad / g_norm).detach()


        with torch.no_grad():
            w_torch[:] = w_adv
        w_torch.requires_grad_(True)

        margins_adv = y_t * (X_t @ w_torch)
        loss_adv = torch.mean(torch.exp(-torch.clamp(margins_adv, min=-50)))

        opt.zero_grad()
        loss_adv.backward()
        opt.step()

        return w_torch.detach().numpy()

    return step
