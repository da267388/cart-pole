import numpy as np


def solve_dare_iterative(A, B, Q, R, max_iter=500, eps=1e-8):
    # P_{k+1} = A^{T}P_{k}A - A^{T}P_{k}B(R + B^{T}P_{k}B)^{-1}B^{T}P_{k}A + Q
    P = Q.copy()
    for _ in range(max_iter):
        # K = (R + B^{T}PB)^{-1}B^{T}PA
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        # P_{k+1} = A^{T}P_{k}A - A^{T}P_{k}BK + Q
        P_new = A.T @ P @ A - A.T @ P @ B @ K + Q

        if np.linalg.norm(P_new - P) < eps:
            break
        P = P_new
    return P