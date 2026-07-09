import random
from basics.autograd import Value

class Module:
    """
    Base class for neural network modules.
    Provides utility methods to manage gradients and parameters.
    """
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []

class Neuron(Module):
    """
    A single neuron that performs linear combination and applies an activation function.
    """
    def __init__(self, nin, nonlin=True):
        # Initialize weights with small random values
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'Tanh' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    """
    A layer of neurons.
    """
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    """
    A Multi-Layer Perceptron (feed-forward neural network).
    """
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i!=len(nouts)-1)) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


if __name__ == "__main__":
    print("--- Custom MLP & XOR Training Demo ---")
    random.seed(42)
    
    # 1. Initialize MLP: 2 inputs, 1 hidden layer with 4 neurons, 1 output neuron
    model = MLP(2, [4, 1])
    print(f"Initialized Model:\n{model}")
    print(f"Total Parameters: {len(model.parameters())}")
    
    # 2. XOR Dataset
    xs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    ys = [-1.0, 1.0, 1.0, -1.0] # XOR targets using -1 and 1 (due to Tanh output activation)
    
    # 3. Training Loop
    print("\nTraining MLP on XOR classification...")
    for epoch in range(100):
        # Forward pass
        ypred = [model(x) for x in xs]
        
        # Calculate mean squared error loss
        loss = sum((yp - y)**2 for y, yp in zip(ys, ypred)) / len(ys)
        
        # Backward pass (gradient calculation)
        model.zero_grad()
        loss.backward()
        
        # Update weights (gradient descent)
        learning_rate = 0.1
        for p in model.parameters():
            p.data -= learning_rate * p.grad
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {loss.data:.4f} | Predictions: {[f'{yp.data:.2f}' for yp in ypred]}")
            
    print("\nFinal evaluation:")
    for x, y in zip(xs, ys):
        pred = model(x).data
        print(f"Input: {x} -> Target: {y:4.1f} | Prediction: {pred:6.2f}")
