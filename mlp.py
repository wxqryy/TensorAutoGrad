import random
from backprop import *


class Neuron:
    def __init__(self, n_inputs, activation=True):
        self.weights = [
            Value(random.uniform(-1, 1), [])
            for _ in range(n_inputs)
        ]

        self.bias = Value(random.uniform(-1, 1), [])
        self.activation = activation

    def __call__(self, x):
        # z = x1*w1 + x2*w2 + ... + b
        z = self.bias

        for xi, wi in zip(x, self.weights):
            z = z + xi * wi

        if self.activation:
            z = z.relu()

        return z

    def parameters(self):
        return self.weights + [self.bias]

class Layer:
    def __init__(self, n_inputs, n_outputs, activation=True):
        self.neurons = [
            Neuron(n_inputs, activation)
            for _ in range(n_outputs)
        ]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        params = []

        for neuron in self.neurons:
            params += neuron.parameters()

        return params

class MLP:
    def __init__(self):
        # 5 -> 8 -> 8 -> 8 -> 8 -> 1

        self.layers = [
            Layer(5, 8, activation=True),
            Layer(8, 8, activation=True),
            Layer(8, 8, activation=True),
            Layer(8, 8, activation=True),
            Layer(8, 1, activation=False)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x[0]

    def parameters(self):
        params = []

        for layer in self.layers:
            params += layer.parameters()

        return params