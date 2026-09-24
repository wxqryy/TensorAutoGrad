from enum import Enum
from math import *
from sys import int_info
from xml.etree.ElementTree import Element


def safeDivide(x, y):
    if y != 0:
        return x / y
    else:
        raise ZeroDivisionError("Divide by 0")


class Operation(Enum):
    ADD = (
        lambda x, y: x + y,
        lambda x, y: 1.0,
        lambda x, y: 1.0
    )
    SUBTRACT = (
        lambda x, y: x - y,
        lambda x, y: 1.0,
        lambda x, y: -1.0
    )
    MULTIPLE = (
        lambda x, y: x * y,
        lambda x, y: y,
        lambda x, y: x
    )
    DIVIDE = (
        safeDivide,
        lambda x, y: 1.0 / y,
        lambda x, y: -x / (y ** 2)
    )
    EXPONENTIATION = (
        lambda x, i: x ** i,
        lambda x, i: i * (x ** (i - 1)) if x != 0 or i > 1 else 0.0,
        lambda x, i: (x ** i) * log(x) if x > 0 else 0.0
    )

    def __init__(self, forward, back_l, back_r):
        self.forward = forward
        self.back_l = back_l
        self.back_r = back_r

class DirectedAcyclicGraph:
    def __init__(self):
        self.history = []

    @property
    def stack(self):
        return self.history

    @stack.setter
    def stack(self, value):
        self.history.append(value)

    class Node:
        def __init__(self, value):
            self.value = value
            self.grad = 0.0

    class Element(Node):
        def __init__(self, value):
            super().__init__(value)

    class Operation(Node):
        def __init__(self, op_type: Operation, left, right, dag):
            if not isinstance(op_type, Operation):
                raise TypeError(f"{op_type} isn't Operation")
            if not isinstance(left, DirectedAcyclicGraph.Node) or not isinstance(right, DirectedAcyclicGraph.Node):
                raise TypeError(f"Term of operation isn't Element")

            self.op_type = op_type
            self.left = left
            self.right = right

            super().__init__(self.op_type.forward(left.value, right.value))

            dag.stack = self

# DAG = DirectedAcyclicGraph()
#
# inputs = [DAG.Element(3), DAG.Element(10), DAG.Element(2), DAG.Element(24), DAG.Element(17)]
# weights = [DAG.Element(0.4), DAG.Element(0.5), DAG.Element(0.1), DAG.Element(0), DAG.Element(1)]
# biases = [DAG.Element(0.6), DAG.Element(0.8), DAG.Element(0.3), DAG.Element(0.9), DAG.Element(0.2)]
#
# for w in weights:
#     for x in inputs:
#         DAG.Operation(Operation.MULTIPLE, x, w, DAG)