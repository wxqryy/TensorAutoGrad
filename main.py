import mlp
from backprop import Value

net = mlp.MLP()

inputs = [
    Value(1),
    Value(2),
    Value(1.5),
    Value(0.4),
    Value(3)
]

y = Value(0.5)
prediction = net(inputs)

diff = prediction - y
L = diff.square()
L = L / Value(2)
L.grad = 1.0