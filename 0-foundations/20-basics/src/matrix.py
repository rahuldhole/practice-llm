class Matrix:
    """
    A simple 2D Matrix class implemented from scratch in pure Python.
    No external libraries (like NumPy or PyTorch) are used in the core math operations.
    """
    def __init__(self, data):
        if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
            raise TypeError("Matrix data must be a list of lists")
        if len(data) == 0 or len(data[0]) == 0:
            raise ValueError("Matrix cannot be empty")
        
        self.rows = len(data)
        self.cols = len(data[0])
        
        # Ensure all rows have the same length
        for r_idx, row in enumerate(data):
            if len(row) != self.cols:
                raise ValueError(f"Inconsistent row length at index {r_idx}")
            for val in row:
                if not isinstance(val, (int, float)):
                    raise TypeError("Matrix elements must be numeric (int or float)")
        
        self.data = [[float(val) for val in row] for row in data]

    @property
    def shape(self):
        return (self.rows, self.cols)

    def __repr__(self):
        return f"Matrix({self.data})"

    def __eq__(self, other):
        if not isinstance(other, Matrix) or self.shape != other.shape:
            return False
        for r in range(self.rows):
            for c in range(self.cols):
                if abs(self.data[r][c] - other.data[r][c]) > 1e-9:
                    return False
        return True

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Addition operand must be a Matrix")
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch for addition: {self.shape} vs {other.shape}")
        
        new_data = [
            [self.data[r][c] + other.data[r][c] for c in range(self.cols)]
            for r in range(self.rows)
        ]
        return Matrix(new_data)

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Subtraction operand must be a Matrix")
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch for subtraction: {self.shape} vs {other.shape}")
        
        new_data = [
            [self.data[r][c] - other.data[r][c] for c in range(self.cols)]
            for r in range(self.rows)
        ]
        return Matrix(new_data)

    def matmul(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Matmul operand must be a Matrix")
        if self.cols != other.rows:
            raise ValueError(f"Dimension mismatch for matmul: {self.shape} and {other.shape}")
        
        # Calculate dot product of row and column
        new_data = []
        for r in range(self.rows):
            new_row = []
            for c in range(other.cols):
                val = sum(self.data[r][k] * other.data[k][c] for k in range(self.cols))
                new_row.append(val)
            new_data.append(new_row)
        return Matrix(new_data)

    def transpose(self):
        new_data = [
            [self.data[r][c] for r in range(self.rows)]
            for c in range(self.cols)
        ]
        return Matrix(new_data)

    @classmethod
    def zeros(cls, rows, cols):
        return cls([[0.0] * cols for _ in range(rows)])

    @classmethod
    def ones(cls, rows, cols):
        return cls([[1.0] * cols for _ in range(rows)])

    def _flatten(self):
        return [val for row in self.data for val in row]

    def mean(self, axis=None):
        """
        Compute the mean of matrix elements.
        If axis is None, computes the mean of the flattened matrix.
        If axis is 0, computes the mean along the columns (returns a 1xCols Matrix).
        If axis is 1, computes the mean along the rows (returns a Rowsx1 Matrix).
        """
        if axis is None:
            flat = self._flatten()
            return sum(flat) / len(flat)
        elif axis == 0:
            # Mean along columns
            new_row = []
            for c in range(self.cols):
                col_sum = sum(self.data[r][c] for r in range(self.rows))
                new_row.append(col_sum / self.rows)
            return Matrix([new_row])
        elif axis == 1:
            # Mean along rows
            new_data = []
            for r in range(self.rows):
                row_sum = sum(self.data[r])
                new_data.append([row_sum / self.cols])
            return Matrix(new_data)
        else:
            raise ValueError("axis must be None, 0, or 1")

    def var(self, axis=None):
        """
        Compute the variance of matrix elements.
        If axis is None, computes the variance of the flattened matrix.
        """
        if axis is None:
            flat = self._flatten()
            m = sum(flat) / len(flat)
            return sum((x - m) ** 2 for x in flat) / len(flat)
        elif axis == 0:
            means = self.mean(axis=0).data[0]
            new_row = []
            for c in range(self.cols):
                col_var = sum((self.data[r][c] - means[c]) ** 2 for r in range(self.rows)) / self.rows
                new_row.append(col_var)
            return Matrix([new_row])
        elif axis == 1:
            means = [row[0] for row in self.mean(axis=1).data]
            new_data = []
            for r in range(self.rows):
                row_var = sum((self.data[r][c] - means[r]) ** 2 for c in range(self.cols)) / self.cols
                new_data.append([row_var])
            return Matrix(new_data)
        else:
            raise ValueError("axis must be None, 0, or 1")


if __name__ == "__main__":
    print("--- Matrix Calculator Demo ---")
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    print(f"Matrix A:\n{A.data}")
    print(f"Matrix B:\n{B.data}")
    
    C = A + B
    print(f"A + B:\n{C.data}")
    
    D = A.matmul(B)
    print(f"A * B (matmul):\n{D.data}")
    
    E = A.transpose()
    print(f"A Transpose:\n{E.data}")
    
    print(f"A Mean (flat): {A.mean()}")
    print(f"A Variance (flat): {A.var()}")
    print(f"A Mean along cols (axis=0): {A.mean(axis=0).data}")
    print(f"A Mean along rows (axis=1): {A.mean(axis=1).data}")
