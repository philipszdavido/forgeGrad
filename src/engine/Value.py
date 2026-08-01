import math


class Value:
    def __init__(self, data, label = ""):
        self.label = label
        self.data = data
        self.backward = lambda: None
        self.grad = 0

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, label={self.label})"

    def __add__(self, other):
        out = Value(self.data + other.data)
        def backward():
            # out = I + other
            # dout/dI = 1 + 0 = 1
            self.grad += out.grad * 1
            other.grad += out.grad * 1

        out.backward = backward
        return out

    def __sub__(self, other):
        out = Value(self.data - other.data)
        def backward():
            # out = self - other
            # dout/dself = 1 - 0 = 1
            self.grad += out.grad * 1
            # dout/dother = 0 - 1 = -1
            other.grad += out.grad * (-1)
            pass
        out.backward = backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data)
        def backward():
            # out = self * other
            # dout/dself = other
            self.grad += out.grad * other.data
            # out = self * other
            # dout/dother = self
            other.grad += out.grad * self.data
        out.backward = backward
        return out

    def __div__(self, other):
        out = Value(self.data / other.data)
        def backward():
            # out = self / other
            # dout/dself = 1/other
            # dout/dother = self * (-1) * other ** (-2)
            self.grad += out.grad * ( 1 / other.data)
            other.grad += out.grad * (-1) * (other.data ** 2)
        out.backward = backward
        return out

    def __pow__(self, other):
        out = Value(self.data**other)
        def backward():
            # out = self ** other
            # dout/dself = other * self ** (other - 1)
            self.grad += out.grad * (other * self.data ** (other - 1))
        out.backward = backward
        return out

    def sin(self):
        out = Value(math.sin(self.data))
        def backward():
            # out = sin(self)
            # dout/dself = cos(self)
            self.grad += out.grad * math.cos(self.data)
        out.backward = backward
        return out
