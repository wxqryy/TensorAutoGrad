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