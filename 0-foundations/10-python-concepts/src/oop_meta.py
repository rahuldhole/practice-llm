from abc import ABC, ABCMeta, abstractmethod

# Registry dictionary to store registered layer classes
LAYER_REGISTRY = {}


class RegistryMeta(ABCMeta):
    """
    A custom Metaclass that registers any concrete subclass of Layer into LAYER_REGISTRY.
    Demonstrates dynamic class creation hook: __new__ or __init__ in metaclasses.
    """
    def __new__(mcs, name, bases, namespace):
        # Create the new class object
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Avoid registering the base class itself
        # (We check if 'Layer' is already defined or if ABC is in bases to identify base classes)
        if name != "Layer" and not namespace.get("__is_abstract_base__", False):
            # Use lower-case class name as key, or a custom class attribute 'type_name' if provided
            registry_name = namespace.get("type_name", name.lower())
            LAYER_REGISTRY[registry_name] = cls
            
        return cls


class Layer(ABC, metaclass=RegistryMeta):
    """
    Abstract Base Class for all network layers.
    Combines ABC interface enforcement and RegistryMeta dynamic class tracking.
    """
    __is_abstract_base__ = True  # Used by metaclass to skip registration of the base class

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        """Property getter for the layer's name."""
        return self._name

    @name.setter
    def name(self, val):
        """Property setter with validation."""
        if not val or not isinstance(val, str):
            raise ValueError("Layer name must be a non-empty string")
        self._name = val

    @abstractmethod
    def forward(self, x):
        """
        Calculates the forward pass. Concrete subclasses must override this.
        """
        pass

    @abstractmethod
    def get_params(self):
        """
        Returns parameters (weights/biases) for this layer.
        """
        pass


# Concrete Subclass Implementations (will be registered automatically)

class Linear(Layer):
    """
    A concrete Linear (Dense) layer.
    """
    type_name = "linear"

    def __init__(self, in_features, out_features, name="linear_layer"):
        # Demonstrate super() calling base class constructor
        super().__init__(name)
        self.in_features = in_features
        self.out_features = out_features
        # Initialize mock weight and bias matrices (as nested lists)
        self._weight = [[1.0] * in_features for _ in range(out_features)]
        self._bias = [0.0] * out_features

    # Using properties to manage weights/bias access and validation
    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, val):
        if not isinstance(val, list) or len(val) != self.out_features:
            raise ValueError("Invalid weight matrix dimensions")
        self._weight = val

    def forward(self, x):
        """
        Performs matrix-vector multiplication: y = Wx + b
        """
        if len(x) != self.in_features:
            raise ValueError(f"Input size {len(x)} mismatch with expected {self.in_features}")
        
        out = []
        for row in self._weight:
            dot_product = sum(w * val for w, val in zip(row, x))
            out.append(dot_product)
        
        return [y + b for y, b in zip(out, self._bias)]

    def get_params(self):
        return {"weight": self._weight, "bias": self._bias}


class Activation(Layer):
    """
    A concrete Activation (ReLU) layer.
    """
    type_name = "relu"

    def __init__(self, name="relu_layer"):
        super().__init__(name)

    def forward(self, x):
        return [max(0.0, val) for val in x]

    def get_params(self):
        return {}  # Activation layers have no parameters


# Factory function to instantiate layers from configuration dicts
def create_layer(config):
    """
    Instantiates a Layer object from a configuration dictionary,
    leveraging the dynamic LAYER_REGISTRY.
    """
    layer_type = config.get("type")
    if layer_type not in LAYER_REGISTRY:
        raise ValueError(f"Unknown layer type: '{layer_type}'. Registered types: {list(LAYER_REGISTRY.keys())}")
    
    # Extract arguments and construct
    args = config.get("args", {})
    return LAYER_REGISTRY[layer_type](**args)


if __name__ == "__main__":
    print("--- OOP & Metaprogramming Registry Demo ---")
    
    print("Registered layers in LAYER_REGISTRY:")
    for name, cls in LAYER_REGISTRY.items():
        print(f"  '{name}': {cls.__name__}")
        
    # Define a neural network model configuration
    model_config = [
        {"type": "linear", "args": {"in_features": 3, "out_features": 2, "name": "layer1"}},
        {"type": "relu", "args": {"name": "relu1"}}
    ]
    
    # Construct network using registry lookup
    network = []
    print("\nConstructing model from config:")
    for conf in model_config:
        layer = create_layer(conf)
        print(f"  Created layer: '{layer.name}' (type: {type(layer).__name__})")
        network.append(layer)
        
    # Forward pass simulation
    x_input = [1.0, 2.0, 3.0]
    print(f"\nForward Pass input: {x_input}")
    
    current = x_input
    for layer in network:
        current = layer.forward(current)
        print(f"  After '{layer.name}': {current}")
        
    print(f"Final Output: {current}")
