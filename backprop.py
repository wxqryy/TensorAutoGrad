from __future__ import annotations
import matrix


class Tensor:
    def __init__(self, data, parents=None, operation=None):
        self.data = data
        self.grad = matrix.zeros_like(data)
        self.parents = parents or []
        self.operation = operation

    def __add__(self, other):
        return Tensor(
            matrix.add(self.data, other.data),
            parents=[self, other],
            operation="+"
        )

    def __sub__(self, other):
        return Tensor(
            matrix.sub(self.data, other.data),
            parents=[self, other],
            operation="-"
        )

    def __mul__(self, other):
        # elementwise mul
        return Tensor(
            matrix.mul_elementwise(self.data, other.data),
            parents=[self, other],
            operation="*"
        )

    def __matmul__(self, other):
        # matrix @ vector
        return Tensor(
            matrix.mul_on_vector(self.data, other.data),
            parents=[self, other],
            operation="@"
        )

    def relu(self):
        return Tensor(
            matrix.relu(self.data),
            parents=[self],
            operation="relu"
        )

    def square(self):
        return Tensor(
            [x * x for x in self.data],
            parents=[self],
            operation="square"
        )

    def mean(self):
        return Tensor(
            sum(self.data) / len(self.data),
            parents=[self],
            operation="mean"
        )

    def _backward(self):
        if self.operation == "+":
            a, b = self.parents
            a.grad = matrix.add(a.grad, self.grad)
            b.grad = matrix.add(b.grad, self.grad)
        elif self.operation == "-":
            a, b = self.parents
            a.grad = matrix.add(a.grad, self.grad)
            b.grad = matrix.add(b.grad, [-g for g in self.grad])
        elif self.operation == "*":
            a, b = self.parents
            grad_a = matrix.mul_elementwise(self.grad, b.data)
            grad_b = matrix.mul_elementwise(self.grad, a.data)
            a.grad = matrix.add(a.grad, grad_a)
            b.grad = matrix.add(b.grad, grad_b)
        elif self.operation == "relu":
            x = self.parents[0]
            local_grad = [g if value > 0 else 0.0 for g, value in zip(self.grad, x.data)]
            x.grad = matrix.add(x.grad, local_grad)
        elif self.operation == "square":
            x = self.parents[0]
            local_grad = [g * 2 * value for g, value in zip(self.grad, x.data)]
            x.grad = matrix.add(x.grad, local_grad)
        elif self.operation == "mean":
            x = self.parents[0]
            n = len(x.data)
            local_grad = [self.grad / n for _ in x.data]
            x.grad = matrix.add(x.grad, local_grad)
        elif self.operation == "@":
            W, x = self.parents
            grad_x = matrix.mul_on_vector(matrix.transpose(W.data), self.grad)
            grad_W = matrix.outer(self.grad, x.data)
            x.grad = matrix.add(x.grad, grad_x)
            W.grad = matrix.add_matrix(W.grad, grad_W)

    def backward(self):
        if isinstance(self.data, list):
            raise ValueError("only scalar tensor")

        topo = []
        visited = set()

        def build_topo(v):
            if v in visited:
                return
            visited.add(v)
            for parent in v.parents:
                build_topo(parent)
            topo.append(v)
        build_topo(self)
        for v in topo:
            if v.parents:
                v.zero_grad()
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        self.grad = matrix.zeros_like(self.data)