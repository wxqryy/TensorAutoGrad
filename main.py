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

topo = []
visited = set()
def build_topo(v):
    if v in visited:
        return

    visited.add(v)

    for parent in v.parents:
        build_topo(parent)

    topo.append(v)

build_topo(L)

for v in reversed(topo):
    v._backward()

for i in net.parameters():
    print(f"{i.value} | {i.grad}")