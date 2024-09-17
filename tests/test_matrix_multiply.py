import pytest
import numpy as np
from scipy.sparse import csr_matrix
from sparse_matrix_mult import sparse_matrix_multiply

@pytest.fixture(scope="module")
def test_matrices():
    # A_matrix: 8x8 sparse matrix
    A_matrix = np.array([
        [0.64, 0.99, 0.89, 0.72, 0,    0,    0,    0   ],
        [0,    0.67, 0.54, 0,    0.81, 0,    0,    0   ],
        [0,    0.32, 0,    0,    0,    0.45, 0,    0   ],
        [0.1,  0,    0,    0,    0,    0,    0.23, 0   ],
        [0,    0,    0.78, 0,    0.55, 0,    0,    0.91],
        [0.43, 0,    0,    0.12, 0,    0,    0,    0   ],
        [0,    0,    0.33, 0,    0,    0.68, 0,    0   ],
        [0,    0.21, 0,    0,    0,    0,    0.39, 0   ]
    ])
    
    # Convert A_matrix to CSR format for sparse operations
    csr_A_matrix = csr_matrix(A_matrix)
    
    # B_matrix: Another 8x8 sparse matrix
    B_matrix = np.array([
        [0.23, 0,    0,    0,    0.51, 0,    0,    0   ],
        [0,    0.72, 0,    0,    0,    0.38, 0,    0   ],
        [0,    0,    0.99, 0,    0,    0,    0.84, 0   ],
        [0,    0.76, 0.87, 0.97, 0,    0,    0,    0.29],
        [0.15, 0,    0,    0,    0.62, 0,    0,    0   ],
        [0,    0.44, 0,    0,    0,    0.75, 0,    0   ],
        [0,    0,    0.58, 0,    0,    0,    0.93, 0   ],
        [0.36, 0,    0,    0.82, 0,    0,    0,    0.47]
    ])
    
    # C_matrix: 9x12 dense matrix
    C_matrix = np.array([
        [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12],
        [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
        [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
        [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
        [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
        [73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84],
        [85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96],
        [97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108]
    ])
    
    # D_matrix: 12x6 dense matrix
    D_matrix = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
        [1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
        [2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
        [3.1, 3.2, 3.3, 3.4, 3.5, 3.6],
        [3.7, 3.8, 3.9, 4.0, 4.1, 4.2],
        [4.3, 4.4, 4.5, 4.6, 4.7, 4.8],
        [4.9, 5.0, 5.1, 5.2, 5.3, 5.4],
        [5.5, 5.6, 5.7, 5.8, 5.9, 6.0],
        [6.1, 6.2, 6.3, 6.4, 6.5, 6.6],
        [6.7, 6.8, 6.9, 7.0, 7.1, 7.2]
    ])
    
    # F_matrix: 12x9 dense matrix
    F_matrix = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14, 15, 16, 17, 18],
        [19, 20, 21, 22, 23, 24, 25, 26, 27],
        [28, 29, 30, 31, 32, 33, 34, 35, 36],
        [37, 38, 39, 40, 41, 42, 43, 44, 45],
        [46, 47, 48, 49, 50, 51, 52, 53, 54],
        [55, 56, 57, 58, 59, 60, 61, 62, 63],
        [64, 65, 66, 67, 68, 69, 70, 71, 72],
        [73, 74, 75, 76, 77, 78, 79, 80, 81],
        [82, 83, 84, 85, 86, 87, 88, 89, 90],
        [91, 92, 93, 94, 95, 96, 97, 98, 99],
        [100, 101, 102, 103, 104, 105, 106, 107, 108]
    ])
    
    return {
        'A_matrix': A_matrix,
        'csr_A_matrix': csr_A_matrix,
        'B_matrix': B_matrix,
        'C_matrix': C_matrix,
        'D_matrix': D_matrix,
        'F_matrix': F_matrix
    }

def test_sparse_nonsym(test_matrices):
    result = sparse_matrix_multiply(test_matrices['C_matrix'], test_matrices['D_matrix'], output_format='sparse', symmetric=False)
    expected = np.matmul(test_matrices['C_matrix'], test_matrices['D_matrix'])
    assert result.shape == expected.shape, "Shapes don't match"
    assert np.allclose(result.toarray(), expected), "Values don't match"

def test_dense_nonsym(test_matrices):
    result = sparse_matrix_multiply(test_matrices['C_matrix'], test_matrices['D_matrix'], output_format='dense', symmetric=False)
    expected = np.matmul(test_matrices['C_matrix'], test_matrices['D_matrix'])
    assert result.shape == expected.shape, "Shapes don't match"
    assert np.allclose(result, expected), "Values don't match"

def test_dense_sym(test_matrices):
    result = sparse_matrix_multiply(test_matrices['C_matrix'], test_matrices['F_matrix'], output_format='dense', symmetric=True)
    expected = np.matmul(test_matrices['C_matrix'], test_matrices['F_matrix'])
    assert result.shape == expected.shape, "Shapes don't match"
    assert np.allclose(np.triu(result), np.triu(expected)), "Upper triangular parts don't match"

def test_sparse_sym(test_matrices):
    result = sparse_matrix_multiply(test_matrices['C_matrix'], test_matrices['F_matrix'], output_format='sparse', symmetric=True)
    expected = np.matmul(test_matrices['C_matrix'], test_matrices['F_matrix'])
    result_dense = result.toarray()
    assert result.shape == expected.shape, "Shapes don't match"
    assert np.allclose(np.triu(result_dense), np.triu(expected)), "Upper triangular parts don't match"
