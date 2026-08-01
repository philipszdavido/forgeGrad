import math


class Value:
    def __init__(self, data, children = (), label = ""):
        self.label = label
        self.data = data
        self._backward = lambda: None
        self.grad = 0
        self.children = children

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, label={self.label})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other))
        def _backward():
            # out = self + other
            # 𝜹out/𝜹self = 1 + 0 = 1
            self.grad += out.grad * 1
            other.grad += out.grad * 1

        out._backward = _backward
        return out

    def __sub__(self, other):
        out = Value(self.data - other.data, (self, other))
        def _backward():
            # out = self - other
            # 𝜹out/𝜹self = 1 - 0 = 1
            self.grad += out.grad * 1
            # 𝜹out/𝜹other = 0 - 1 = -1
            other.grad += out.grad * (-1)
            pass
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other))
        def _backward():
            # out = self * other
            # 𝜹out/𝜹self = other
            self.grad += out.grad * other.data
            # out = self * other
            # 𝜹out/𝜹other = self
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __div__(self, other):
        out = Value(self.data / other.data, (self, other))
        def _backward():
            # out = self / other
            # 𝜹out/𝜹self = 1/other
            # 𝜹out/𝜹other = self * (-1) * other ** (-2)
            self.grad += out.grad * ( 1 / other.data)
            other.grad += out.grad * (-1) * (other.data ** 2)
        out._backward = _backward
        return out

    def __pow__(self, other):
        out = Value(self.data ** other, (self,))
        def _backward():
            # out = self ** other
            # 𝜹out/𝜹self = other * self ** (other - 1)
            self.grad += out.grad * (other * self.data ** (other - 1))
        out._backward = _backward
        return out

    def sin(self):
        out = Value(math.sin(self.data), (self,))
        def _backward():
            # out = sin(self)
            # 𝜹out/𝜹self = cos(self)
            self.grad += out.grad * math.cos(self.data)
        out._backward = _backward
        return out

    def backward(self):
        self._backward()

        children = []

        for child in self.children:
            child._backward()
            children.append(child._children)

        for child in children:
            child.backward()


