import numpy as np

def simulate_lti(A, B, K, x0, N):
    x = x0
    xs, us = [], []

    for _ in range(N):
        u = -K @ x
        xs.append(x)
        us.append(u)
        x = A @ x + B @ u

    return np.array(xs), np.array(us)