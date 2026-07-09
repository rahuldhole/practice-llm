import time
from functools import wraps

def timer(func):
    """
    A decorator that measures and prints the execution time of the decorated function.
    Demonstrates basic function decoration and metadata preservation using functools.wraps.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"[Timer] Function '{func.__name__}' executed in {elapsed:.6f} seconds")
        # Store elapsed time on the wrapper for testing purposes
        wrapper.last_elapsed = elapsed
        return result
    wrapper.last_elapsed = 0.0
    return wrapper


def memoize(func):
    """
    A decorator that caches the results of a function based on its arguments.
    Demonstrates closure-based state preservation. Useful for expensive recursive
    functions like computing Fibonacci numbers or attention weights.
    """
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a key from args and kwargs (sorted items to ensure dict consistency)
        # Note: items must be hashable
        kwargs_key = tuple(sorted(kwargs.items()))
        key = (args, kwargs_key)
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    # Expose cache for inspection/testing
    wrapper.cache = cache
    return wrapper


def logger(level="INFO"):
    """
    A decorator factory (decorator that accepts arguments) that logs the input arguments
    and output of the decorated function.
    Demonstrates three-level nested functions for custom decorator configuration.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            args_str = ", ".join(repr(a) for a in args)
            kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
            params = ", ".join(filter(None, [args_str, kwargs_str]))
            
            print(f"[{level}] Calling '{func.__name__}({params})'")
            result = func(*args, **kwargs)
            print(f"[{level}] '{func.__name__}' returned {repr(result)}")
            
            # Store log calls for testing purposes
            wrapper.calls.append((args, kwargs, result))
            return result
        wrapper.calls = []
        return wrapper
    return decorator


@timer
@memoize
@logger(level="DEBUG")
def expensive_fibonacci(n):
    """
    Computes Fibonacci number recursively.
    Demonstrates decorator stacking.
    """
    if n < 2:
        return n
    return expensive_fibonacci(n - 1) + expensive_fibonacci(n - 2)


if __name__ == "__main__":
    print("--- Decorators & Closures Demo ---")
    
    # 1. Test Timer
    @timer
    def heavy_calculation(limit):
        total = 0
        for i in range(limit):
            total += i
        return total

    print("Running heavy calculation...")
    heavy_calculation(1_000_000)
    
    # 2. Test Memoize + Logging + Timer stacking
    print("\nRunning stacked Fibonacci(6)...")
    res1 = expensive_fibonacci(6)
    print(f"Result 1: {res1}")
    
    print("\nRunning stacked Fibonacci(6) again (should hit cache instantly)...")
    res2 = expensive_fibonacci(6)
    print(f"Result 2: {res2}")
