from abc import ABC, ABCMeta, abstractmethod

# --- EXAMPLE 1: Standard Use (The "Shape" Example) ---
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


# --- EXAMPLE 3: Metadata Enforcement ---
# This demonstrates how Metaclasses can force subclasses to define specific 
# metadata (like 'version') before they are even allowed to exist.
print("\n--- Example 3: Metadata Enforcement ---")

class VersionedMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace):
        # Enforce that every plugin MUST define a 'version' attribute
        if name != "BaseVersionedPlugin" and "version" not in namespace:
            raise TypeError(f"Class {name} is missing required 'version' attribute!")
        return super().__new__(mcls, name, bases, namespace)

class BaseVersionedPlugin(ABC, metaclass=VersionedMeta):
    @abstractmethod
    def perform(self):
        pass

# This will work
class ValidPlugin(BaseVersionedPlugin):
    version = "1.0.0"
    def perform(self):
        return "Running valid plugin"

# This would raise a TypeError if uncommented:
# class InvalidPlugin(BaseVersionedPlugin):
#     def perform(self):
#         return "Missing version"

v_plugin = ValidPlugin()
print(f"Plugin running: {v_plugin.perform()} (Version: {v_plugin.version})")