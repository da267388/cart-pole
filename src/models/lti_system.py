import numpy as np

class LTISystem:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def step(self, x, u):
        return self.A @ x + self.B @ u