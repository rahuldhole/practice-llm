# 🧬 Tutorial 02: Decorators & Closures

**TLDR:** Using decorators to modify function behavior dynamically.

A **decorator** is a design pattern in Python that allows you to modify, wrap, or extend the behavior of a function or class without permanently modifying its source code. Under the hood, decorators rely on **closures**.

---

## 1. What is a Closure?
A **closure** is a nested function that remembers and has access to variables from its enclosing lexical scope, even after the outer function has finished executing.

```python
def make_multiplier(factor):
    def multiplier(number):
        return number * factor  # 'factor' is remembered from the outer scope
    return multiplier

double = make_multiplier(2)
print(double(5))  # Outputs 10
```

---

## 2. Basic Decorators
A decorator is simply a function that takes another function as an argument, defines a wrapper function, and returns that wrapper.

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        # Do something before call
        result = func(*args, **kwargs)
        # Do something after call
        return result
    return wrapper
```

### Preserving Metadata with `functools.wraps`
When wrapping a function, its original docstring, name, and attributes are overwritten by the wrapper function. To prevent this loss of information, Python provides the `@wraps` decorator inside the `functools` module, which copies the metadata back to the wrapper.

*Code reference*: [`@timer` in decorators.py](../src/decorators.py#L4-L21)

---

## 3. Decorator Factories (Decorators with Arguments)
To pass arguments to a decorator itself, you must create a third layer of nesting: a function that returns a decorator, which in turn returns a wrapper.

*Code reference*: [`@logger` in decorators.py](../src/decorators.py#L44-L67)

```text
[Decorator Factory]  --> Takes configuration arguments (e.g. level="DEBUG")
   L [Decorator]     --> Takes the target function 'func'
      L [Wrapper]    --> Takes input arguments (*args, **kwargs) and calls 'func'
```

---

## 4. Decorator Stacking
Decorators can be chained (stacked) on top of a single function. They apply from the bottom up (or inside out).

```python
@decorator_a
@decorator_b
def my_func():
    pass
```
Is equivalent to: `my_func = decorator_a(decorator_b(my_func))`

*Code reference*: [`expensive_fibonacci` in decorators.py](../src/decorators.py#L70-L80)

---

## 💡 Practical Challenge
Open [decorators.py](../src/decorators.py) and execute it with `task run -- src/decorators.py`. Note how stacking `expensive_fibonacci` with `@memoize` turns an exponential $O(2^n)$ runtime into a linear $O(n)$ runtime. Try writing a custom `@count_calls` decorator that counts and prints how many times a function has been called.
