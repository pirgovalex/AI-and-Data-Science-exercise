import numpy as np
import pandas as pd
import argparse
import os
import sys

class Matrix:
    def __init__(self, array):
        if not isinstance(array, np.ndarray):
            raise ValueError("Input must be a numpy array")
        self.array = array
    def auto_print(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            print(f"{func.__name__} returned: {result} (type: {type(result)})")
            return result
        return wrapper
    def __array__(self):
        return self.array
    def __repr__(self):
        return f"Matrix({self.array})"

    def __str__(self):
        return str(f" self.array ")

    @auto_print
    def add(self, other) :
        if self.array.shape != other.array.shape:
            raise ValueError("Matrices must have the same shape for addition")
        return Matrix(self.array + other.array)

    @auto_print
    def subtract(self, other):
        if self.array.shape != other.array.shape:
            raise ValueError("Matrices must have the same shape for subtraction")
        return Matrix(self.array - other.array)
    @auto_print
    def multiply(self, other):
        if self.array.shape[1] != other.array.shape[0]:
            raise ValueError("Number of columns of first matrix must equal number of rows of second")
        return Matrix(np.dot(self.array, other.array))
    @auto_print
    def transpose(self):
        return Matrix(self.array.T)

    @auto_print
    def inverse(self):
        if self.array.shape[0] != self.array.shape[1]:
            raise ValueError("Matrix must be square to calculate inverse")
        if np.linalg.det(self.array) == 0:
            raise ValueError("Matrix is singular and cannot be inverted")
        return Matrix(np.linalg.inv(self.array))

    @auto_print
    def determinant(self):
        if self.array.shape[0] != self.array.shape[1]:
            raise ValueError("Matrix must be square to calculate determinant")
        return np.linalg.det(self.array)

    @auto_print
    def rank(self):
        return np.linalg.matrix_rank(self.array)
    @auto_print
    def eigen(self):
        if self.array.shape[0] != self.array.shape[1]:
            raise ValueError("Matrix must be square to calculate eigenvalues")
        return np.linalg.eig(self.array)
    @auto_print
    @staticmethod
    def solve(A, b):
        b= b.reshape(-1, 1) if b.ndim == 1 else b
        if A.shape[0] != b.shape[0]:
            raise ValueError("Number of rows in A must equal number of rows in b")
        try:
            return Matrix(np.linalg.solve(A, b))
        except np.linalg.LinAlgError:
            # least squares if the system is not solvable or is overdetermined
            return np.linalg.lstsq(A, b, rcond=None)[0]
def load_matrix_from_csv(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        return pd.read_csv(file_path, header=None).to_numpy(dtype=float)
    except Exception as e:
        raise ValueError(f"Error reading {file_path}: {e}")
def save_matrix_to_csv(matrix, output_path):
    if isinstance(matrix, Matrix):
        matrix = matrix.array
    matrix = np.atleast_2d(matrix)
    pd.DataFrame(matrix).to_csv(output_path, index=False, header=False)
def main()->None:
    parser = argparse.ArgumentParser(description="Matrix Calculator CLI")
    parser.add_argument('--operation', type=str, required=True,
                        choices=['add', 'sub', 'dot', 'transpose', 'inverse',
                                 'det', 'rank', 'solve', 'eig'],
                        help="Matrix operation to perform")
    parser.add_argument('--input1', type=str, required=True, help="Path to first input matrix (CSV)")
    parser.add_argument('--input2', type=str, help="Path to second input matrix (CSV) if needed")
    parser.add_argument('--output', type=str, required=True, help="Path to save the result")
    args = parser.parse_args()
    A = load_matrix_from_csv(args.input1)
    matrix_A = Matrix(A)
    result = None
    if args.operation in ['add', 'sub', 'dot', 'solve']:
        if not args.input2:
            print("Error: --input2 is required for this operation", file=sys.stderr)
            sys.exit(1)
        B = load_matrix_from_csv(args.input2)
        matrix_B = Matrix(B)
    if args.operation   == 'add':
        result = matrix_A.add(matrix_B)
    elif args.operation == 'sub':
        result = matrix_A.subtract(matrix_B)
    elif args.operation == 'dot':
        result = matrix_A.multiply(matrix_B)
    elif args.operation == 'transpose':
        result = matrix_A.transpose()
    elif args.operation == 'inverse':
        result = matrix_A.inverse()
    elif args.operation == 'det':
        result = matrix_A.determinant()
    elif args.operation == 'rank':
        result = matrix_A.rank()
    elif args.operation == 'solve':
        solution = Matrix.solve(A, B)
        result = solution
    elif args.operation == 'eig':
        values, vectors = matrix_A.eigen()
        save_matrix_to_csv(values.reshape(1, -1), args.output.replace('.csv', '_values.csv'))
        save_matrix_to_csv(vectors, args.output.replace('.csv', '_vectors.csv'))
        print(f"Eigenvalues and eigenvectors saved as {args.output.replace('.csv', '_values.csv')} and {args.output.replace('.csv', '_vectors.csv')}")
        sys.exit(0)
    save_matrix_to_csv(result, args.output)
    print(f"Result saved to {args.output}")

if __name__ == "__main__":
    main()