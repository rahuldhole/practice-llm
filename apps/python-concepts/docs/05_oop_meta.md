# 🧬 Tutorial 05: OOP & Metaprogramming

**TLDR:** Advanced Object-Oriented Programming and metaclass concepts.

Modern machine learning libraries make heavy use of Object-Oriented Programming (OOP) and Metaprogramming to manage network layers, optimizers, and models dynamically.

---

## 1. Abstract Base Classes (ABCs)
An **Abstract Base Class (ABC)** defines a common interface that all subclasses must implement. You cannot instantiate an ABC directly if it defines abstract methods.
- Import `ABC` and `abstractmethod` from the `abc` standard module.
- Mark abstract methods using the `@abstractmethod` decorator.

*Code reference*: [`Layer` in oop_meta.py](../src/oop_meta.py#L22-L56)

---

## 2. Property Descriptors
Properties allow you to define methods that can be accessed like attributes (e.g. `obj.name`). This is useful for:
- Read-only attributes (getter only).
- Input validation when values are written (using a setter).

```python
class MyClass:
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_val):
        if new_val < 0:
            raise ValueError("Must be non-negative")
        self._value = new_val
```

*Code reference*: [`name` in Layer](../src/oop_meta.py#L32-L44) and [`weight` in Linear](../src/oop_meta.py#L71-L81)

---

## 3. Metaclasses and Dynamic Registries
In Python, **classes are objects too**, and their type is a **metaclass**. By default, Python uses the `type` metaclass to construct classes. By defining a custom metaclass, you can hook into the class creation process itself.

### The Registry Pattern
In deep learning frameworks, we often want to register custom modules, loss functions, or datasets automatically under string names, so we can instantiate them directly from configuration files (e.g. JSON or YAML).
Instead of manually adding classes to a lookup dictionary, we can use a custom metaclass that intercepts subclass creation and registers the class automatically.

```text
Class declaration (e.g., class Linear)
   |
   V
Custom Metaclass __new__ interceptor
   |
   +--> Instantiate Class Object
   +--> Auto-register Class object in LAYER_REGISTRY
   V
Class is ready to use
```

*Code reference*: [`RegistryMeta` in oop_meta.py](../src/oop_meta.py#L6-L19) and [`create_layer` factory in oop_meta.py](../src/oop_meta.py#L110-L121)

---

## 💡 Practical Challenge
Open [oop_meta.py](../src/oop_meta.py) and execute it with `task run -- src/oop_meta.py`. Observe how the registry mapping is built automatically. Try adding a new class `Sigmoid(Layer)` that performs element-wise sigmoid activation and check if it gets registered automatically and if you can instantiate it from a config dictionary.
