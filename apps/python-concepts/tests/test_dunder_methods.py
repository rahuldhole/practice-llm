import pytest
from python_concepts.dunder_methods import Vector

def test_vector_init():
    v = Vector([1, 2, 3])
    assert len(v) == 3
    assert v.elements == [1.0, 2.0, 3.0]
    
    with pytest.raises(TypeError):
        Vector("not-a-list")
        
    with pytest.raises(TypeError):
        Vector([1, "two", 3])


def test_vector_representation():
    v = Vector([1, 2.5])
    assert repr(v) == "Vector([1.0, 2.5])"
    assert str(v) == "v[1.0, 2.5]"


def test_vector_getitem():
    v = Vector([10, 20, 30, 40])
    assert v[0] == 10.0
    assert v[2] == 30.0
    assert v[1:3] == Vector([20, 30])
    
    with pytest.raises(TypeError):
        _ = v["first"]


def test_vector_equality():
    assert Vector([1, 2]) == Vector([1.0, 2.0])
    assert Vector([1, 2]) != Vector([1, 3])
    assert Vector([1, 2]) != "not-a-vector"


def test_vector_arithmetic():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    
    # Addition
    assert v1 + v2 == Vector([5, 7, 9])
    with pytest.raises(ValueError):
        _ = v1 + Vector([1, 2])
        
    # Subtraction
    assert v2 - v1 == Vector([3, 3, 3])
    
    # Multiplication (scalar)
    assert v1 * 3 == Vector([3, 6, 9])
    assert 2 * v1 == Vector([2, 4, 6])
    
    # Multiplication (dot product)
    assert v1 * v2 == 1*4 + 2*5 + 3*6  # 4 + 10 + 18 = 32
    
    with pytest.raises(TypeError):
        _ = v1 * "string"


def test_vector_call():
    v = Vector([1, 2, 3])
    assert v(10) == Vector([10, 20, 30])
    with pytest.raises(TypeError):
        _ = v("ten")
