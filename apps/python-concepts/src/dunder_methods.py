class Vector:
    """
    A mathematical Vector class demonstrating operator overloading,
    indexing, slicing, callability, and representation in Python.
    
    This illustrates the power of Magic/Dunder methods to make custom
    objects feel like native Python types.
    """
    def __init__(self, elements):
        if not isinstance(elements, (list, tuple)):
            raise TypeError("Elements must be a list or tuple of numbers")
        if not all(isinstance(x, (int, float)) for x in elements):
            raise TypeError("All elements must be numeric (int or float)")
        self.elements = [float(x) for x in elements]

    def __repr__(self):
        """
        Returns a developer-friendly representation of the object.
        Ideally, eval(repr(obj)) == obj.
        """
        return f"Vector({self.elements})"

    def __str__(self):
        """
        Returns a user-friendly string representation of the object.
        Used by str() and print().
        """
        return f"v[{', '.join(str(x) for x in self.elements)}]"

    def __len__(self):
        """
        Returns the number of elements in the vector.
        Invoked by the len() function.
        """
        return len(self.elements)

    def __getitem__(self, index):
        """
        Enables index indexing and slicing.
        Invoked by obj[index].
        """
        if isinstance(index, slice):
            return Vector(self.elements[index])
        elif isinstance(index, int):
            return self.elements[index]
        else:
            raise TypeError("Index must be an integer or slice")

    def __eq__(self, other):
        """
        Compares two vectors for element-wise equality.
        Invoked by the == operator.
        """
        if not isinstance(other, Vector):
            return False
        if len(self) != len(other):
            return False
        # Use abs difference for floating-point tolerance
        return all(abs(a - b) < 1e-9 for a, b in zip(self.elements, other.elements))

    def __add__(self, other):
        """
        Performs element-wise addition: Vector + Vector.
        Invoked by the + operator.
        """
        if not isinstance(other, Vector):
            raise TypeError("Addition operand must be a Vector")
        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimension for addition")
        return Vector([a + b for a, b in zip(self.elements, other.elements)])

    def __sub__(self, other):
        """
        Performs element-wise subtraction: Vector - Vector.
        Invoked by the - operator.
        """
        if not isinstance(other, Vector):
            raise TypeError("Subtraction operand must be a Vector")
        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimension for subtraction")
        return Vector([a - b for a, b in zip(self.elements, other.elements)])

    def __mul__(self, other):
        """
        Performs scaling by a scalar (Vector * Scalar) or dot product (Vector * Vector).
        Invoked by the * operator.
        """
        if isinstance(other, (int, float)):
            # Vector * Scalar
            return Vector([x * other for x in self.elements])
        elif isinstance(other, Vector):
            # Vector * Vector (Dot product)
            if len(self) != len(other):
                raise ValueError("Vectors must have the same dimension for dot product")
            return sum(a * b for a, b in zip(self.elements, other.elements))
        else:
            raise TypeError("Multiplication is only supported with a numeric scalar or another Vector")

    def __rmul__(self, other):
        """
        Performs scaling by a scalar from the left: Scalar * Vector.
        Invoked when the left-hand operand does not support __mul__ with Vector.
        """
        # delegate to __mul__
        return self * other

    def __call__(self, scale_factor):
        """
        Enables the object to be called as a function: vector(scale_factor).
        Useful for modular designs, e.g. simulating layer application.
        """
        if not isinstance(scale_factor, (int, float)):
            raise TypeError("Call argument must be a numeric scaling factor")
        return self * scale_factor


if __name__ == "__main__":
    print("--- Dunder Methods Demo ---")
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    
    print(f"v1 repr: {repr(v1)}")
    print(f"v1 str: {v1}")
    print(f"v1 length: {len(v1)}")
    print(f"v1 first element (v1[0]): {v1[0]}")
    print(f"v1 slice (v1[1:]): {v1[1:]}")
    
    print(f"\nOperations:")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v2 - v1 = {v2 - v1}")
    print(f"v1 * 3 (Scalar Mult) = {v1 * 3}")
    print(f"2 * v1 (Left Scalar Mult) = {2 * v1}")
    print(f"v1 * v2 (Dot Product) = {v1 * v2}")
    
    print(f"\nCallability:")
    print(f"Calling v1 as a function: v1(10) -> {v1(10)}")
