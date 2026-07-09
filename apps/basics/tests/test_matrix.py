import pytest
from basics.matrix import Matrix

def test_matrix_init():
    A = Matrix([[1, 2], [3, 4]])
    assert A.shape == (2, 2)
    assert A.data == [[1.0, 2.0], [3.0, 4.0]]
    
    with pytest.raises(ValueError):
        Matrix([])
    
    with pytest.raises(ValueError):
        Matrix([[1, 2], [3]])
        
    with pytest.raises(TypeError):
        Matrix([[1, "two"], [3, 4]])

def test_matrix_add_sub():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    
    C = A + B
    assert C == Matrix([[6, 8], [10, 12]])
    
    D = B - A
    assert D == Matrix([[4, 4], [4, 4]])
    
    with pytest.raises(ValueError):
        _ = A + Matrix([[1, 2, 3]])

def test_matrix_matmul():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[2, 0], [1, 2]])
    
    # matmul product:
    # [1*2 + 2*1, 1*0 + 2*2] -> [4, 4]
    # [3*2 + 4*1, 3*0 + 4*2] -> [10, 8]
    C = A.matmul(B)
    assert C == Matrix([[4, 4], [10, 8]])
    
    with pytest.raises(ValueError):
        A.matmul(Matrix([[1, 2, 3]]))

def test_matrix_transpose():
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    assert A.transpose() == Matrix([[1, 4], [2, 5], [3, 6]])

def test_matrix_stats():
    A = Matrix([[1, 2], [3, 4]])
    assert A.mean() == 2.5
    assert A.var() == 1.25  # mean=2.5, elements diff squares: (1.5^2 + 0.5^2 + 0.5^2 + 1.5^2)/4 = (2.25+0.25+0.25+2.25)/4 = 1.25
    
    col_mean = A.mean(axis=0)
    assert col_mean == Matrix([[2.0, 3.0]])
    
    row_mean = A.mean(axis=1)
    assert row_mean == Matrix([[1.5], [3.5]])
