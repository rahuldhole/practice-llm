import time
import sys
from contextlib import contextmanager

class TimerContext:
    """
    A class-based context manager that measures block execution time.
    Demonstrates the __enter__ and __exit__ protocol.
    """
    def __init__(self, description="Block"):
        self.description = description
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self  # Return self so user can bind it: `with TimerContext() as t:`

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        self.elapsed = end_time - self.start_time
        print(f"[TimerContext] {self.description} took {self.elapsed:.6f} seconds")
        # Return False so that any exception raised in the block propagates up
        return False


class SafeLogger:
    """
    A class-based context manager that writes logs to a file or stream,
    safely handling open/close resources and managing exceptions.
    """
    def __init__(self, file_path, suppress_errors=False):
        self.file_path = file_path
        self.suppress_errors = suppress_errors
        self.file = None

    def __enter__(self):
        self.file = open(self.file_path, "a")
        return self

    def write(self, message):
        if self.file:
            self.file.write(f"[{time.asctime()}] {message}\n")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        
        if exc_type is not None:
            print(f"[SafeLogger] Caught exception inside block: {exc_val}")
            # If suppress_errors is True, return True to prevent exception from propagating
            return self.suppress_errors
        return False





# Global runtime configuration dictionary
_RUN_CONFIG = {
    "gradient_enabled": True,
    "precision": "float32",
    "device": "cpu"
}

def get_config(key):
    return _RUN_CONFIG.get(key)


@contextmanager
def config_scope(**temp_config):
    """
    A generator-based context manager using contextlib.contextmanager.
    Temporarily overrides values in the global config, restoring them on exit.
    Similar to PyTorch's torch.no_grad() or torch.cuda.amp.autocast().
    """
    global _RUN_CONFIG
    original = _RUN_CONFIG.copy()
    
    # Apply temporary settings
    for k, v in temp_config.items():
        if k not in _RUN_CONFIG:
            raise KeyError(f"Invalid configuration key: '{k}'")
        _RUN_CONFIG[k] = v
        
    try:
        yield  # Yield control to the with-block
    finally:
        # Restore original settings (guaranteed to execute even if exceptions occur)
        _RUN_CONFIG = original



if __name__ == "__main__":
    print("--- Context Managers Demo ---")
    
    # 1. TimerContext demo
    with TimerContext("Matrix computation simulation") as t:
        total = 0
        for i in range(500_000):
            total += i
    print(f"Recorded elapsed time inside context variable: {t.elapsed:.6f}s")
    
    # 2. Exception suppression demo
    print("\nWriting logs with exception handling...")
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Open logger with suppress_errors=True
        with SafeLogger(temp_path, suppress_errors=True) as logger:
            logger.write("First log line")
            logger.write("Going to raise an error now...")
            raise RuntimeError("Something went wrong in the network training!")
            logger.write("This line will not run")
            
        print("Successfully recovered from the error inside SafeLogger because suppress_errors=True")
        
        # Read file contents to prove it worked
        with open(temp_path, "r") as f:
            print(f"Log file contents:\n{f.read()}")
            
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # 3. ConfigContext demo
    print(f"\nInitial config: gradient_enabled={get_config('gradient_enabled')}, precision={get_config('precision')}")
    
    print(f"Before config context: gradient_enabled={get_config('gradient_enabled')}, precision={get_config('precision')}")

    # Temporarily turn off gradients
    with config_scope(gradient_enabled=False, precision="float16"):
        print(f"Inside config context: gradient_enabled={get_config('gradient_enabled')}, precision={get_config('precision')}")
        
    print(f"After config context: gradient_enabled={get_config('gradient_enabled')}, precision={get_config('precision')}")