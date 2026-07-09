# 🧬 Tutorial 01: Dunder (Magic) Methods

In Python, **Dunder (Double Underscore)** or **Magic Methods** allow user-defined classes to hook into Python's syntax. Instead of calling explicit methods like `obj.add(other)`, dunder methods enable you to write `obj + other`.

This is critical in deep learning:
- **PyTorch Tensors** overload `+`, `-`, `*`, `@` (matrix multiplication) so that writing neural equations looks clean and readable.
- **Autograd engines** (like PyTorch or Karpathy's Micrograd) overload operators to build the computational graph behind the scenes when mathematical operations occur.
- **Datasets** implement indexing (`dataset[i]`) and length tracking (`len(dataset)`) so standard data loaders can query them.

---

## 1. Core Magic Methods Explained

### A. Initialization & Representation
- `__init__(self, ...)`: The constructor method. Sets up instance variables when an object is created.
- `__repr__(self)`: Returns an unambiguous developer-friendly string representation of the object. Ideally: `eval(repr(obj)) == obj`.
- `__str__(self)`: Returns a clean, user-friendly representation of the object. Called by `print()` or `str()`.

*Code reference*: [`__repr__` and `__str__` in dunder_methods.py](../src/dunder_methods.py#L12-L28)

### B. Collections and Indexing
- `__len__(self)`: Invoked by the `len()` function. Returns the dimension or size of a container.
- `__getitem__(self, index)`: Invoked by bracket indexing `obj[index]`. Enables retrieving a specific element or a slice/sub-object (e.g. `obj[1:3]`).

*Code reference*: [`__len__` and `__getitem__` in dunder_methods.py](../src/dunder_methods.py#L30-L46)

### C. Operator Overloading
Operator overloading allows objects to participate in mathematical operations:
- `__add__(self, other)`: Implements `self + other`.
- `__sub__(self, other)`: Implements `self - other`.
- `__mul__(self, other)`: Implements `self * other`.
- `__rmul__(self, other)`: Implements `other * self` when the left-hand operand does not support multiplication with your object. Useful for operations like `2.0 * vector`.

*Code reference*: [`__add__`, `__sub__`, `__mul__`, and `__rmul__` in dunder_methods.py](../src/dunder_methods.py#L57-L99)

### D. Callability
- `__call__(self, *args, **kwargs)`: Makes an instance callable like a standard function (e.g., `obj(...)`). This pattern is ubiquitous in PyTorch (`nn.Module`), where layers are called like functions: `output = linear_layer(input)`.

*Code reference*: [`__call__` in dunder_methods.py](../src/dunder_methods.py#L101-L110)

---

## 💡 Practical Challenge
Open [dunder_methods.py](../src/dunder_methods.py) and execute it with `task run -- src/dunder_methods.py`. Try implementing a custom `__truediv__(self, other)` method to allow element-wise division (`v1 / 2`) and verify it with tests.
