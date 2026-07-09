import math

class Value:
    """
    A lightweight scalar value that tracks its gradient and computational graph history.
    Enables reverse-mode automatic differentiation (backpropagation) from scratch.
    Similar to Andrej Karpathy's micrograd.
    """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # Topological sort to find execution order
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)
        
        # Backpropagate gradients
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


if __name__ == "__main__":
    print("--- Autograd Engine Demo ---")
    # Example expression: y = x^2 + 2x
    x = Value(5.0, label='x')
    y = x*x + 2*x
    y.backward()
    
    print(f"For y = x^2 + 2x at x = 5:")
    print(f"y value: {y.data} (expected: 35.0)")
    print(f"dy/dx (gradient): {x.grad} (expected: 2x + 2 = 12.0)")
    
    # Another example with activations
    print("\n--- Activation backprop demo ---")
    a = Value(-2.0)
    b = Value(3.0)
    c = a * b
    d = c.tanh()
    d.backward()
    print(f"c data: {c.data}, d data (tanh(c)): {d.data}")
    print(f"dc/da: {b.data}")
    print(f"dd/da (via chain rule): {a.grad}")
