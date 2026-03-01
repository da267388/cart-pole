import numpy as np
from scipy.linalg import expm

def zoh_discretize(A, B, dt):
    """
    Discretize continuous-time LTI system using zero-order hold:
    [ Ad  Bd ] = exp( [A  B] dt )
    [ 0   0 ]       [0  0]
    """
    n = A.shape[0]
    m = B.shape[1]

    M = np.zeros((n+m, n+m))
    M[0:n, 0:n] = A
    M[0:n, n:n+m] = B

    Md = expm(M * dt)

    Ad = Md[0:n, 0:n]
    Bd = Md[0:n, n:n+m]

    return Ad, Bd

# matrix exponential
# x(t) = e^{AT} * x(0)
# zoh_discretize_A = e^{AT}