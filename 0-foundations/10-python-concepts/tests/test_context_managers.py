import pytest
import os
import tempfile
from python_concepts.context_managers import TimerContext, config_scope, get_config, SafeLogger

def test_timer_context():
    with TimerContext("Test block") as t:
        pass
    assert t.elapsed >= 0.0


def test_config_scope():
    # Initial config should have standard values
    assert get_config("gradient_enabled") is True
    assert get_config("precision") == "float32"
    
    with config_scope(gradient_enabled=False, precision="float16"):
        assert get_config("gradient_enabled") is False
        assert get_config("precision") == "float16"
        
    # Should restore on exit
    assert get_config("gradient_enabled") is True
    assert get_config("precision") == "float32"
    
    # Test error propagates but config is still restored
    with pytest.raises(RuntimeError):
        with config_scope(gradient_enabled=False):
            assert get_config("gradient_enabled") is False
            raise RuntimeError("Trigger error")
            
    assert get_config("gradient_enabled") is True


def test_safe_logger():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Test error suppression
        with SafeLogger(temp_path, suppress_errors=True) as logger:
            logger.write("Test message")
            raise ValueError("Test value error")
            
        # Verify file has the written log
        with open(temp_path, "r") as f:
            content = f.read()
            assert "Test message" in content
            
        # Verify error propagates when suppress_errors=False
        with pytest.raises(ValueError):
            with SafeLogger(temp_path, suppress_errors=False) as logger:
                raise ValueError("Propagating error")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
