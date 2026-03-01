import numpy as np
from src.models.cartpole import cartpole_linearized
from src.models.discretization import zoh_discretize
from src.control.controllability import is_controllable
from src.control.lqr import lqr
from src.control.simulation import simulate_lti

def main():
    A, B = cartpole_linearized()
    print("Controllable:", is_controllable(A, B))

    Ad, Bd = zoh_discretize(A, B, dt=0.02)

    Q = np.diag([10, 1, 50, 1])
    R = np.array([[0.1]])

    K, P = lqr(Ad, Bd, Q, R)
    print("LQR Gain K:\n", K)

    x0 = np.array([0.1, 0, 0.1, 0])
    xs, us = simulate_lti(Ad, Bd, K, x0, N=300)

    print("Final state:", xs[-1])

if __name__ == "__main__":
    main()