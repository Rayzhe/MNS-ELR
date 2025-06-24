
import numpy as np

def square_sum_error(W: np.ndarray, b: float,
                     X: np.ndarray, Y: np.ndarray) -> float:
    """
    Squared ℓ₂ error  ‖ W·vec(X) + b − Y ‖²

    Parameters
    ----------
    W : (m, m) weight matrix
    b : scalar bias
    X : (m, m, n) 3-D tensor (n trials of m×m matrices)
    Y : (n,) target values

    Returns
    -------
    float
        Squared error value.
    """
    n = len(Y)
    m = X.shape[1]
    w_vec = W.reshape(-1, 1)          # (m², 1)
    x_mat = X.reshape(m ** 2, n)      # (m², n)
    residual = w_vec.T @ x_mat + b - Y.T
    return np.linalg.norm(residual, ord=2) ** 2


def grad_W(W: np.ndarray, b: float,
           X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Gradient of the squared error w.r.t. W.
    """
    n = len(Y)
    m = W.shape[1]
    w_vec = W.reshape(-1, 1)
    x_mat = X.reshape(m ** 2, n)
    g = 2 * ((w_vec.T @ x_mat + b - Y.T) @ x_mat.T)  # (1, m²)
    return g.reshape(m, m)


def grad_b(W: np.ndarray, b: float,
           X: np.ndarray, Y: np.ndarray) -> float:
    """
    Gradient of the squared error w.r.t. b.
    """
    m = W.shape[1]
    n = len(Y)
    w_vec = W.reshape(-1, 1)
    x_mat = X.reshape(m ** 2, n)
    return np.sum(2 * (w_vec.T @ x_mat + b - Y.T))


def gamma_fun(mu: float, W: np.ndarray, W0: np.ndarray,
              b: float, b0: float, X: np.ndarray, Y: np.ndarray) -> float:
    """
    Quadratic majorization (upper-bound) for back-tracking line search.
    """
    f0 = square_sum_error(W0, b0, X, Y)
    gW = grad_W(W0, b0, X, Y)
    gb = grad_b(W0, b0, X, Y)
    quad_W = np.linalg.norm(W - W0, ord='fro') ** 2
    quad_b = (b - b0) ** 2
    return (
        f0
        + np.trace((W - W0).T @ gW)
        + (b - b0) * gb
        + quad_W / (2 * mu)
        + quad_b / (2 * mu)
    )

def L21L1(X: np.ndarray, Y: np.ndarray,
          lambda1: float, lambda21: float):
    """
    Solve   min_{W,b}  ‖W·vec(X) + b − Y‖² + λ₂‖W‖₁ + λ₃∑‖row_i(W)‖₂
    using accelerated proximal gradient (FISTA) with back-tracking.
    Returns
    -------
    W : (m, m)  optimal weight matrix
    b : scalar   optimal bias
    """

    m = X.shape[1]         # matrix dimension
    W  = np.zeros((m, m))
    W0 = np.zeros_like(W)
    Q  = np.zeros_like(W)

    alpha = alpha0 = 1.0   # FISTA momentum
    b = b0 = 0.0
    t = 1.0                # initial step size

    f = square_sum_error(W, b, X, Y)
    g = 0.0                # regularizer is zero at W = 0
    o0 = f + g             # initial objective value

    while True:
        # -------- Nesterov extrapolation --------
        P = W + (alpha0 - 1) / alpha * (W - W0)
        b = b + (alpha0 - 1) / alpha * (b - b0)

        # cache previous iterate
        W0[:] = W
        b0 = b
        alpha0 = alpha

        # gradients at extrapolated point
        G_P = grad_W(P, b0, X, Y)
        G_b = grad_b(P, b0, X, Y)

        # -------- Back-tracking line search --------
        while True:
            # gradient step
            b = b - t * G_b
            for i in range(m):
                Q[:, i] = P[:, i] - t * G_P[:, i]

            # element-wise L1 clipping + row-wise L21 shrinking
            for i in range(m):
                Q = np.clip(Q, -lambda1, lambda1)
                W[i, :] = (lambda21 * Q[i, :]) \
                          / max(np.linalg.norm(Q[i, :]), lambda21)

            # objective value
            f = square_sum_error(W, b, X, Y)
            g = sum(np.linalg.norm(W[i, :]) for i in range(m))
            g = lambda21 * g + lambda1 * np.linalg.norm(W, ord=1)
            o1 = f + g

            # sufficient decrease test
            if f <= gamma_fun(t, W, P, b, b0, X, Y):
                break
            t *= 0.5   # shrink step size

        # -------- convergence check --------
        delta = np.linalg.norm(o0 - o1) / o0
        if (delta <= 1e-5 and o0 != 0) or (t <= 1e-12):
            break

        # update momentum
        o0 = o1
        alpha = (1 + np.sqrt(1 + 4 * alpha0**2)) / 2

    return W, b
