import random
from backprop import *


class Linear:
    def __init__(self, n_inputs, n_outputs, activation=True):
        self.weights = Tensor([[random.uniform(-1, 1) for _ in range(n_inputs)] for _ in range(n_outputs)])

        self.bias = Tensor([random.uniform(-1, 1) for _ in range(n_outputs)])

        self.activation = activation

    def __call__(self, x):
        z = self.weights @ x
        z = z + self.bias

        if self.activation:
            z = z.relu()

        return z

    def parameters(self):
        return [self.weights, self.bias]

class MLP:
    def __init__(self):
        # 5 -> 8 -> 8 -> 8 -> 8 -> 1

        self.layers = [
            Linear(5, 8, activation=True),
            Linear(8, 8, activation=True),
            Linear(8, 8, activation=True),
            Linear(8, 8, activation=True),
            Linear(8, 1, activation=False)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        params = []

        for layer in self.layers:
            params += layer.parameters()

        return params