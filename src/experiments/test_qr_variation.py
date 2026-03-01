import numpy as np
from src.models.cartpole import cartpole_linearized
from src.models.discretization import zoh_discretize
from src.control.lqr import lqr
from src.control.simulation import simulate_lti

def run_experiment():
    A, B = cartpole_linearized()
    Ad, Bd = zoh_discretize(A, B, dt=0.02)

    test_cases = [
        (np.diag([1,1,10,1]), np.array([[0.1]])),
        (np.diag([10,1,50,1]), np.array([[0.1]])),
        (np.diag([1,1,10,1]), np.array([[1.0]])),
    ]

    x0 = np.array([0.1, 0, 0.1, 0])
    N = 300

    for Q, R in test_cases:
        K, _ = lqr(Ad, Bd, Q, R)
        xs, us = simulate_lti(Ad, Bd, K, x0, N)

        print("Q=\n", Q)
        print("R=\n", R)
        print("Final state:", xs[-1])
        print("Max control input:", np.max(np.abs(us)))
        print("------")

if __name__ == "__main__":
    run_experiment()