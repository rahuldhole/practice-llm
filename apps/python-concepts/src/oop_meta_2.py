from abc import ABC, ABCMeta, abstractmethod

# --- EXAMPLE 1: Standard Use (The "Shape" Example) ---
# Inheriting from ABC is the standard, readable way to define interfaces.

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Square(Shape):
    def __init__(self, side):
        self.side = side
        
    def area(self):
        return self.side * self.side

print("--- Example 1: Standard ABC ---")
sq = Square(5)
print(f"Square area: {sq.area()}")


# --- EXAMPLE 2: Advanced Use (Custom Metaclass with ABCMeta) ---
# Use this when you need custom logic during class creation (like registration 
# or validation) while still enforcing abstract methods.
print("\n--- Example 2: ABCMeta with Custom Logic ---")
class RegistryMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        print(f"DEBUG: Initialized class '{name}'")
        return cls

class BasePlugin(ABC, metaclass=RegistryMeta):
    @abstractmethod
    def run(self):
        pass

class EmailPlugin(BasePlugin):
    def run(self):
        return "Sending email..."

plugin = EmailPlugin()
print(plugin.run())