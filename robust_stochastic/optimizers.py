import numpy as np

'''
All definitions relating to loss functions, gradients, algorithm steps and their stochastic analogues
'''

###################
# Logistic Loss 
###################

def logistic_loss(w, X, y):
    margins = y * (X @ w)
    return np.mean(np.logaddexp(0, -margins))

def logistic_grad(w, X, y):
    margins = y * (X @ w)
    probs = 1.0 / (1.0 + np.exp(margins))
    return -(y * probs) @ X / len(X)

def stochastic_logistic_grad(w, batch_indices, X, y):
    X_batch = X[batch_indices,:]
    y_batch = y[batch_indices]
    return logistic_grad(w, X_batch, y_batch)

def stochastic_logistic_loss(w, batch_indices, X, y):
    X_batch = X[batch_indices,:]
    y_batch = y[batch_indices]
    return logistic_loss(w, X_batch, y_batch)


###################
# Exponential Loss 
###################

def exponential_loss(w, X, y):
    margins = y * (X @ w)
    safe_margins = np.clip(margins, -100, 100)  # clip to prevent overflow errors
    return np.mean(np.exp(-safe_margins))

def exponential_grad(w, X, y):
    margins = y * (X @ w)
    safe_margins = np.clip(margins, -100, 100)
    coeffs = np.exp(-safe_margins)
    return -(X.T @ (y*coeffs)) / len(y) # note: we divide by len(y) = N for avg loss

def stochastic_exponential_grad(w, batch_indices, X, y):
    X_batch = X[batch_indices,:]
    y_batch = y[batch_indices]
    return exponential_grad(w, X_batch, y_batch)

def stochastic_exponential_loss(w, batch_indices, X, y):
    X_batch = X[batch_indices,:]
    y_batch = y[batch_indices]
    return exponential_loss(w, X_batch, y_batch)

def stochastic_normalized_grad(w, batch_indices, X, y):
    # X_batch = X[batch_indices,:]
    # y_batch = y[batch_indices]
    # margins = y_batch * (X_batch @ w)
    # shift = np.max(-margins) 
    # exps = np.exp(-margins - shift)  
    # softmax_weights = exps / np.sum(exps)

    # return -(X_batch.T @ (y_batch * softmax_weights))
    batch_grad = stochastic_exponential_grad(w, batch_indices, X, y)
    batch_loss = stochastic_exponential_loss(w, batch_indices, X, y)
    return batch_grad/(batch_loss+1e-12)  # add machine epsilon to avoid division by zero


def sgd_step(w, sgrad, lr):
    w_new = w - lr * sgrad
    return w_new

def sgd_sam_step(w, sgrad, batch_indices, X, y, lr, rho):

    X_batch = X[batch_indices,:]
    y_batch = y[batch_indices]

    sgrad_norm = np.linalg.norm(sgrad) + 1e-12  # add machine epsilon to ensure no division by 0 happens
    eps = rho * (sgrad / sgrad_norm)    # in Eucledian space, this equation is simple (see SAM paper)

    sam_grad = stochastic_logistic_grad(w + eps, batch_indices, X, y)

    return sgd_step(w, sam_grad, lr)    # reuse sgd_step function

def stochastic_ngd_step(w, sgrad, batch_indices, X, y, lr):

    ngd_grad = stochastic_normalized_grad(w, batch_indices, X, y)

    return sgd_step(w, ngd_grad, lr)    # reuse sgd_step function

def sam_ngd_step(w, sgrad, batch_indices, X, y, lr, rho):

    X_batch = X[batch_indices,:]
    y_batch = y[batch_indices]

    sgrad_norm = np.linalg.norm(sgrad) + 1e-12  # add machine epsilon to ensure no division by 0 happens
    eps = rho * (sgrad / sgrad_norm)    # in Eucledian space, this equation is simple (see SAM paper)

    sam_ngd_grad = stochastic_normalized_grad(w + eps, batch_indices, X, y)

    return sgd_step(w, sam_ngd_grad, lr)    # reuse sgd_step function