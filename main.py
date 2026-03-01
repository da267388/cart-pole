import numpy as np
from src.models.cartpole import cartpole_linearized
from src.models.discretization import zoh_discretize
from src.control.controllability import is_controllable
from src.control.lqr import lqr
from src.control.simulation import simulate_lti

import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov, eigvals

def draw(Ad, Bd, K):
    # stabilized A
    Acl = Ad - Bd @ K
    
    # 可控 Gramian
    Wc = solve_discrete_lyapunov(Acl, Bd @ Bd.T)
    Wc = solve_discrete_lyapunov(Acl, Bd @ Bd.T)

    # 取 eigenvalues
    eig_vals = np.real(eigvals(Wc))
    
    #_, eigvecs = np.linalg.eig(Wc)

    print("Controllability Gramian eigenvalues:", eig_vals)
    #print("Controllability Gramian eigenvectors:", eigvecs)

    # 視覺化
    plt.figure(figsize=(5,4))
    plt.bar(range(len(eig_vals)), eig_vals)
    plt.title("Controllability Gramian Eigenvalues")
    plt.xlabel("State index")
    plt.ylabel("Eigenvalue magnitude")
    plt.show()

def compute_settling_time(x_traj, dt, eps=0.01):
    norms = np.linalg.norm(x_traj.T, axis=0)
    N = norms.size
    for i in range(N):
        if np.all(norms[i:] < eps):
            return i*dt, i
    return None, None

def main():
    dt=0.02
    A, B = cartpole_linearized()
    print("Controllable:", is_controllable(A, B))

    Ad, Bd = zoh_discretize(A, B, dt=dt)

    Q = np.diag([5, 1, 50, 1])
    R = np.array([[0.1]])
    
    print("Q=\n", Q)
    print("R=\n", R)

    K, P = lqr(Ad, Bd, Q, R)
    print("LQR Gain K:\n", K)

    x0 = np.array([0.1, 0, 0.1, 0])
    xs, us = simulate_lti(Ad, Bd, K, x0, N=300)

    print("Final state:", xs[-1])
    print("Max control input:", np.max(np.abs(us)))
    
    time, iter = compute_settling_time(xs, dt=dt)
    print("Time until converge:", time)
    print("Iter until converge:", iter)
    
    #draw(Ad, Bd, K)
    

if __name__ == "__main__":
    main()