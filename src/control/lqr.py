import numpy as np
from src.control.dare import solve_dare_iterative

def lqr(A, B, Q, R):
    P = solve_dare_iterative(A, B, Q, R)
    # K = (R + B^{T}PB)^{-1}B^{T}PA
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K, P