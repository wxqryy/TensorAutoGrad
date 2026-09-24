from __future__ import annotations

class Value:
    def __init__(self, value, parents: list[Value], operation=None):
        self.value = value
        self.grad = 0.0
        self.parents = parents
        self.operation = operation

    def __add__(self, other):
        return Value(self.value + other.value, parents=[self, other], operation='+')

    def __mul__(self, other):
        return Value(self.value * other.value, parents=[self, other], operation='*')

    def __relu__(self):
        return Value(max(0, self.value), parents=[self], operation='relu')