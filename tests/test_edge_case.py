import pytest
import numpy as np
from scipy.sparse import csr_matrix
from sparse_matrix_mult import sparse_matrix_multiply

@pytest.fixture(scope="module")
def test_matrices():
    # 1x1 matrices
    one_by_one_A = np.array([[5]])
    one_by_one_B = np.array([[2]])
    
    # Matrix with last three rows as zeros
    matrix_with_zero_rows = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    
    # All-zero matrices
    zero_matrix_3x3 = np.zeros((3, 3))
    zero_matrix_3x4 = np.zeros((3, 4))
    
    # Sparse versions
    sparse_matrix_with_zero_rows = csr_matrix(matrix_with_zero_rows)
    sparse_zero_matrix_3x3 = csr_matrix(zero_matrix_3x3)
    sparse_zero_matrix_3x4 = csr_matrix(zero_matrix_3x4)
    
    return {
        "one_by_one_A": one_by_one_A,
        "one_by_one_B": one_by_one_B,
        "matrix_with_zero_rows": matrix_with_zero_rows,
        "zero_matrix_3x3": zero_matrix_3x3,
        "zero_matrix_3x4": zero_matrix_3x4,
        "sparse_matrix_with_zero_rows": sparse_matrix_with_zero_rows,
        "sparse_zero_matrix_3x3": sparse_zero_matrix_3x3,
        "sparse_zero_matrix_3x4": sparse_zero_matrix_3x4
    }

def test_one_by_one_multiplication(test_matrices):
    result = sparse_matrix_multiply(test_matrices["one_by_one_A"], test_matrices["one_by_one_B"], output_format='dense', symmetric=True)
    expected = np.matmul(test_matrices["one_by_one_A"], test_matrices["one_by_one_B"])
    assert np.allclose(result, expected), "1x1 matrix multiplication result doesn't match expected value."

def test_matrix_with_zero_rows(test_matrices):
    B = np.random.rand(3, 4)
    result = sparse_matrix_multiply(test_matrices["matrix_with_zero_rows"], B, output_format='dense', symmetric=False)
    expected = np.matmul(test_matrices["matrix_with_zero_rows"], B)
    assert result.shape == expected.shape, "Matrix shapes don't match."
    assert np.allclose(result, expected), "Matrix with zero rows multiplication result doesn't match expected value."

def test_zero_matrix_multiplication(test_matrices):
    result = sparse_matrix_multiply(test_matrices["zero_matrix_3x3"], test_matrices["zero_matrix_3x4"], output_format='sparse', symmetric=False)
    expected = np.matmul(test_matrices["zero_matrix_3x3"], test_matrices["zero_matrix_3x4"])
    assert result.shape == expected.shape, "Matrix shapes don't match."
    assert np.allclose(result.toarray(), expected), "Zero matrix multiplication result doesn't match expected value."

def test_sparse_matrix_with_zero_rows(test_matrices):
    B = csr_matrix(np.random.rand(3, 4))
    result = sparse_matrix_multiply(test_matrices["sparse_matrix_with_zero_rows"], B, output_format='sparse', symmetric=False)
    expected = test_matrices["sparse_matrix_with_zero_rows"].dot(B)
    assert result.shape == expected.shape, "Matrix shapes don't match."
    assert np.allclose(result.toarray(), expected.toarray()), "Sparse matrix with zero rows multiplication result doesn't match expected value."

def test_sparse_zero_matrix_multiplication(test_matrices):
    result = sparse_matrix_multiply(test_matrices["sparse_zero_matrix_3x3"], test_matrices["sparse_zero_matrix_3x4"], output_format='sparse', symmetric=False)
    expected = test_matrices["sparse_zero_matrix_3x3"].dot(test_matrices["sparse_zero_matrix_3x4"])
    assert result.shape == expected.shape, "Matrix shapes don't match."
    assert np.allclose(result.toarray(), expected.toarray()), "Sparse zero matrix multiplication result doesn't match expected value."
