import pytest
import time
from python_concepts.decorators import timer, memoize, logger

def test_timer_decorator():
    @timer
    def dummy_func():
        time.sleep(0.01)
        return "done"
        
    assert dummy_func() == "done"
    assert dummy_func.last_elapsed >= 0.01


def test_memoize_decorator():
    call_count = 0
    
    @memoize
    def compute_square(x):
        nonlocal call_count
        call_count += 1
        return x * x
        
    assert compute_square(4) == 16
    assert call_count == 1
    
    # Second call should hit the cache and not execute the function
    assert compute_square(4) == 16
    assert call_count == 1
    
    assert compute_square(5) == 25
    assert call_count == 2
    
    assert len(compute_square.cache) == 2


def test_logger_decorator():
    @logger(level="TEST")
    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}"
        
    res = greet("Alice", greeting="Hi")
    assert res == "Hi, Alice"
    assert len(greet.calls) == 1
    assert greet.calls[0] == (("Alice",), {"greeting": "Hi"}, "Hi, Alice")
