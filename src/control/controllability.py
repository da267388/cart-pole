import numpy as np

def controllability_matrix(A, B):
    n = A.shape[0]
    # C = [B AB A^2B ... A^{n-1}B]
    C = B
    for i in range(1, n):
        C = np.hstack((C, np.linalg.matrix_power(A, i) @ B))
    return C

def is_controllable(A, B):
    C = controllability_matrix(A, B)
    return np.linalg.matrix_rank(C) == A.shape[0]