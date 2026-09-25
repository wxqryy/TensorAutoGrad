from __future__ import annotations

class Value:
    def __init__(self, value, parents=None, operation=None):
        self.value = value
        self.grad = 0.0
        self.parents = parents or []
        self.operation = operation

    def __add__(self, other):
        return Value(self.value + other.value, parents=[self, other], operation='+')

    def __sub__(self, other):
        return Value(self.value - other.value, parents=[self, other], operation='-')

    def __mul__(self, other):
        return Value(self.value * other.value, parents=[self, other], operation='*')

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError("Divide by zero")
        return Value(self.value / other.value, parents=[self, other], operation='/')

    def square(self):
        return Value(self.value * self.value, parents=[self], operation='square')

    def relu(self):
        return Value(max(0, self.value), parents=[self], operation='relu')

    def _backward(self):
        if self.operation == "+":
            for p in self.parents:
                p.grad += self. grad * 1
        elif self.operation == "*":
            self.parents[0].grad += self.grad * self.parents[1].value
            self.parents[1].grad += self.grad * self.parents[0].value
        elif self.operation == "relu":
            if self.parents[0].value < 0:
                self.parents[0].grad += self.grad * 0
            else:
                self.parents[0].grad += self.grad * 1
        elif self.operation == '-':
            self.parents[0].grad += self.grad * 1
            self.parents[1].grad += self.grad * -1
        elif self.operation == '/':
            self.parents[0].grad += self.grad / self.parents[1].value
            self.parents[1].grad += self.grad / (-self.parents[0].value / (self.parents[1].value**2))
        elif self.operation == 'square':
            self.parents[0].grad += self.grad * 2 * self.parents[0].value
