import mlp
from backprop import Tensor


net = mlp.MLP()

inputs = Tensor([
    1,
    2,
    1.5,
    0.4,
    3
])

y = Tensor([0.5])
prediction = net(inputs)
diff = prediction - y
loss = diff.square().mean()

loss.backward()

for i, parameter in enumerate(net.parameters()):
    if i%2==0:
        print(f"\nweights {(i+2)//2}")
    else:
        print(f"\nbiases {(i + 2) // 2}")
    print("data:")
    for i in parameter.data:
        if isinstance(i, list):
            print(" | ".join(map(lambda x: f"{x:.15f}" if x < 0 else f" {x:.15f}", i)))
        else:
            print(" | ".join(map(str, parameter.data)))
    print("grad:")
    for i in parameter.grad:
        if isinstance(i, list):
            print(" | ".join(map(lambda x: f"{x:.15f}" if x < 0 else f" {x:.15f}", i)))
        else:
            print(" | ".join(map(str, parameter.grad)))