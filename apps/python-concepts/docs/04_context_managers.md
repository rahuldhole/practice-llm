# 🧬 Tutorial 04: Context Managers

A **Context Manager** is an object that defines the runtime context to be established when executing a `with` statement. They automate resource allocation and cleanup (e.g. closing files, releasing locks, setting global environment states).

---

## 1. The Context Manager Protocol
A class can act as a context manager by implementing two magic methods:
1. `__enter__(self)`: Runs before the code block. It sets up the context. Whatever this method returns is bound to the target variable in the `as` clause.
2. `__exit__(self, exc_type, exc_val, exc_tb)`: Runs after the code block exits (even if an exception occurred). It manages cleanup and exception handling.

*Code reference*: [`TimerContext` in context_managers.py](../src/context_managers.py#L5-L22)

---

## 2. Managing Exceptions
The `__exit__` method receives details about any exception raised within the `with` block:
- `exc_type`: The exception class (e.g., `ValueError`).
- `exc_val`: The exception instance (e.g., `ValueError("Invalid argument")`).
- `exc_tb`: A traceback object.

### Suppressing Exceptions
If `__exit__` returns a truthy value (`True`), the exception raised inside the `with` block is suppressed, and execution continues normally after the `with` statement. If it returns a falsy value (`False` or `None`), the exception propagates.

*Code reference*: [`SafeLogger` in context_managers.py](../src/context_managers.py#L54-L78)

---

## 3. Generator-Based Context Managers (`contextlib`)
Instead of defining a class with `__enter__` and `__exit__`, you can use the `@contextmanager` decorator from the `contextlib` standard library. This decorator wraps a generator function that yields exactly once:
- Everything before the `yield` statement is executed as the `__enter__` phase.
- The yielded value is bound to the `as` variable.
- Everything after the `yield` statement is executed as the `__exit__` phase.

```python
from contextlib import contextmanager

@contextmanager
def simple_context():
    # __enter__ code
    try:
        yield
    finally:
        # __exit__ code
        pass
```

### Temporary Configurations (State Scopes)
In deep learning libraries, context managers are widely used to temporarily override runtime parameters. For example, PyTorch's `with torch.no_grad():` disables gradient computation temporarily.

*Code reference*: [`config_scope` in context_managers.py](../src/context_managers.py#L31-L51)

---

## 💡 Practical Challenge
Open [context_managers.py](../src/context_managers.py) and execute it with `task run -- src/context_managers.py`. Try creating a `mock_device(device_name)` context manager using `@contextmanager` that temporarily sets a global device configuration to `"gpu"` and restores it back to its original value when the context exits.
