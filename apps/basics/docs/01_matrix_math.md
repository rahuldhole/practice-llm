# 🧬 Tutorial 01: Matrix Math Foundations

**TLDR:** Foundations of matrix math operations essential for deep learning.

In deep learning, all inputs, weights, activations, and gradients are represented as arrays of numbers. A 0D array is a **scalar**, a 1D array is a **vector**, and a 2D array is a **matrix**. Multi-dimensional arrays are called **tensors**.

Before building neural networks, we must master matrix mathematical operations, as they are the engine of all forward and backward passes.

---

## 1. Mathematical Entities

### Scalars, Vectors, and Matrices
* **Scalar ($s$)**: A single real number.
  $$s \in \mathbb{R}$$
  *Example*: $s = 4.2$
* **Vector ($\mathbf{v}$)**: A 1D array of numbers.
  $$\mathbf{v} = [v_1, v_2, \dots, v_n]^T \in \mathbb{R}^n$$
  *Example*: $\mathbf{v} = [1.0, 2.0, 3.0]^T$ (often written vertically)
* **Matrix ($\mathbf{A}$)**: A 2D grid of numbers.
  $$\mathbf{A} \in \mathbb{R}^{m \times n}$$
  where $m$ is the number of rows and $n$ is the number of columns.
  $$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}$$

---

## 2. Core Matrix Operations

### A. Element-Wise Addition and Subtraction
To add or subtract two matrices, they **must** have the exact same shape. We perform the operation element-by-element:
$$C_{ij} = A_{ij} + B_{ij}$$

```text
[1  2]  +  [5  6]  =  [1+5  2+6]  =  [6   8]
[3  4]     [7  8]     [3+7  4+8]     [10 12]
```
*Code reference*: [`__add__` and `__sub__` in matrix.py](../src/matrix.py#L38-L58)

### B. Matrix Multiplication (Matmul)
Matrix multiplication represents the mapping of inputs through a neural network layer.
* **Shape Constraint**: To multiply matrix $\mathbf{A}$ of shape $(m \times k)$ by matrix $\mathbf{B}$ of shape $(k \times n)$, the number of columns in $\mathbf{A}$ must equal the number of rows in $\mathbf{B}$. The resulting matrix $\mathbf{C}$ will have shape $(m \times n)$.
* **Calculation**: Each entry $C_{ij}$ is the dot product of row $i$ of $\mathbf{A}$ and column $j$ of $\mathbf{B}$:
  $$C_{ij} = \sum_{r=1}^{k} A_{ir} B_{rj}$$

```text
Row 1 of A: [1, 2]   Col 1 of B: [5, 7]^T
C_11 = 1*5 + 2*7 = 5 + 14 = 19
```
*Code reference*: [`matmul` in matrix.py](../src/matrix.py#L60-L75)

### C. Matrix Transpose
Transposing a matrix flips it over its main diagonal, swapping its row and column indices. The transpose of an $m \times n$ matrix is an $n \times m$ matrix.
$$A^T_{ji} = A_{ij}$$

```text
[1  2  3]^T  =  [1  4]
[4  5  6]       [2  5]
                [3  6]
```
*Code reference*: [`transpose` in matrix.py](../src/matrix.py#L77-L83)

---

## 3. Statistical Calculations

### Mean and Variance
Deep learning architectures use statistical properties of matrices to stabilize inputs during training (e.g., Layer Normalization).

* **Mean ($\mu$)**: The average value of all elements.
  $$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$
* **Variance ($\sigma^2$)**: Measures how far the numbers are spread out from their average.
  $$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$$

*Code reference*: [`mean` and `var` in matrix.py](../src/matrix.py#L88-L135)

---

## 💡 Practical Challenge
Open [matrix.py](../src/matrix.py). Read the methods and run the demo script. Try to clear the implementation of the `matmul` method and rewrite it yourself. Run `task test` to check if your manual implementation is correct!
