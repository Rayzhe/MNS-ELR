import copy
import numpy as np

# ------------------------------------------------------------------
# 基础函数：损失与梯度   ——  与原版完全相同，只加了类型注释 & 英文 docstring
# ------------------------------------------------------------------
def square_sum_error(W: np.ndarray, b: float,
                     X: np.ndarray, Y: np.ndarray) -> float:
    """
    ‖ W·vec(X) + b  − Y ‖²
    X shape : (m, m, n)   ;  Y shape : (n,)
    """
    n = len(Y)
    m = X.shape[1]
    w_vec = W.reshape(-1, 1)          # (m²,1)
    x_mat = X.reshape(m ** 2, n)      # (m²,n)
    res   = w_vec.T @ x_mat + b - Y.T
    return np.linalg.norm(res, ord=2) ** 2


def grad_W(W: np.ndarray, b: float,
           X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    n = len(Y)
    m = W.shape[1]
    w_vec = W.reshape(-1, 1)
    x_mat = X.reshape(m ** 2, n)
    g = 2 * ((w_vec.T @ x_mat + b - Y.T) @ x_mat.T)  # (1,m²)
    return g.reshape(m, m)


def grad_b(W: np.ndarray, b: float,
           X: np.ndarray, Y: np.ndarray) -> float:
    m = W.shape[1]
    n = len(Y)
    w_vec = W.reshape(-1, 1)
    x_mat = X.reshape(m ** 2, n)
    return np.sum(2 * (w_vec.T @ x_mat + b - Y.T))


def gamma_fun(mu: float, W: np.ndarray, W0: np.ndarray,
              b: float, b0: float, X: np.ndarray, Y: np.ndarray) -> float:
    """
    Majorization upper-bound used in back-tracking line-search
    """
    f0 = square_sum_error(W0, b0, X, Y)
    gW = grad_W(W0, b0, X, Y)
    gb = grad_b(W0, b0, X, Y)
    quad_W = np.linalg.norm(W - W0, ord='fro') ** 2
    quad_b = (b - b0) ** 2
    return f0 + np.trace((W - W0).T @ gW) + (b - b0) * gb \
           + quad_W / (2 * mu) + quad_b / (2 * mu)

# ------------------------------------------------------------------
# 主函数：完全保留原版循环 / 更新逻辑
# ------------------------------------------------------------------
def L21L1(X: np.ndarray, Y: np.ndarray,
          lambda1: float, lambda21: float):
    """Exact replica of the original algorithm (FISTA + L21+L1)"""
    # 为保持符号与原脚本一致：

    m = X.shape[1]        # channel / 矩阵维度
    W  = np.zeros((m, m))
    W0 = np.zeros_like(W)
    Q  = np.zeros_like(W)

    alpha  = 1.0
    alpha0 = 1.0
    b  = 0.0
    b0 = b
    t  = 1.0              # 初始步长

    f = square_sum_error(W, b, X, Y)
    g = 0.0               # W=0 时正则项为 0
    o0 = f + g            # 初始目标

    while True:
        # ---------- Nesterov 外推 ----------
        P = W + (alpha0 - 1) / alpha * (W - W0)
        b = b + (alpha0 - 1) / alpha * (b - b0)

        # 缓存旧变量
        W0[:] = W
        b0 = b
        alpha0 = alpha

        # 梯度
        G_P = grad_W(P, b0, X, Y)
        G_b = grad_b(P, b0, X, Y)

        # ---------- 回溯线搜索 ----------
        while True:
            # 梯度下降步
            b = b - t * G_b
            for i in range(m):
                Q[:, i] = P[:, i] - t * G_P[:, i]

            # L1 clip + L21 row-shrink —— 与原版双循环逐元素一致
            for i in range(m):
                for j in range(m):
                    if Q[j, i] < -lambda1:
                        Q[j, i] = -lambda1
                    if Q[j, i] > lambda1:
                        Q[j, i] =  lambda1
                W[i, :] = (lambda21 * Q[i, :]) \
                          / max(np.linalg.norm(Q[i, :]), lambda21)

            # 目标函数
            f = square_sum_error(W, b, X, Y)
            g = sum(np.linalg.norm(W[i, :]) for i in range(m))
            g = lambda21 * g + lambda1 * np.linalg.norm(W, ord=1)
            o1 = f + g

            # Majorization 判断
            if f <= gamma_fun(t, W, P, b, b0, X, Y):
                break
            t *= 0.5

        # ---------- 收敛判据 ----------
        delta = np.linalg.norm(o0 - o1) / o0
        if (delta <= 1e-5 and o0 != 0) or (t <= 1e-12):
            break

        # 更新 FISTA 动量
        o0 = o1
        alpha = (1 + np.sqrt(1 + 4 * alpha0**2)) / 2

    return W, b
