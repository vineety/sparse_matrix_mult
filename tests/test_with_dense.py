import pytest
import numpy as np
import time
from scipy.sparse import random as sparse_random, csr_matrix, eye
from scipy import stats
from sparse_matrix_mult import sparse_matrix_multiply

def time_execution(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def compare_execution_times(custom_time, scipy_time):
    if custom_time == 0 and scipy_time == 0:
        return "Both methods completed in negligible time."
    elif custom_time == 0:
        return "Custom routine completed in negligible time, significantly faster than SciPy."
    elif scipy_time == 0:
        return "SciPy completed in negligible time, significantly faster than custom routine."
    else:
        time_diff = abs(custom_time - scipy_time)
        faster_method = "faster" if custom_time < scipy_time else "slower"
        ratio = scipy_time / custom_time if custom_time > 0 else float('inf')
        return f"Custom routine is {faster_method} than SciPy by {time_diff:.6f} seconds. SciPy would take {ratio:.2f} times more to complete the same operation."

def calculate_sparsity(matrix):
    return 1 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))

@pytest.mark.parametrize("size,density", [(5, 0.01), (5, 0.1), (5, 0.3), (6, 0.01), (6, 0.1), (6, 0.3)])
def test_different_sparsity_levels(size, density):
    A_sparse = sparse_random(size, size, density=density, format='csr')
    B_sparse = sparse_random(size, size, density=density, format='csr')
    
    sparsity_A = calculate_sparsity(A_sparse)
    sparsity_B = calculate_sparsity(B_sparse)
    
    result_custom, custom_time = time_execution(sparse_matrix_multiply, A_sparse, B_sparse, output_format='sparse', symmetric=False)
    result_scipy, scipy_time = time_execution(A_sparse.dot, B_sparse)
    
    comparison = compare_execution_times(custom_time, scipy_time)
    
    assert np.allclose(result_custom.A, result_scipy.A), f"Results do not match for size {size}x{size}, density {density}"
    print(f"\nMatrix size: {size}x{size}, Target density: {density}")
    print(f"Actual sparsity: A: {sparsity_A:.4f}, B: {sparsity_B:.4f}")
    print(f"Custom Sparse Execution time: {custom_time:.6f} seconds")
    print(f"SciPy Sparse Execution time: {scipy_time:.6f} seconds")
    print(f"Difference between results: {np.sum(result_scipy.A - result_custom.A)}")
    print(comparison)

def test_non_square_sparse_matrices():
    A_nonsquare = sparse_random(500, 400, density=0.1, format='csr')
    B_nonsquare = sparse_random(400, 500, density=0.1, format='csr')
    
    sparsity_A = calculate_sparsity(A_nonsquare)
    sparsity_B = calculate_sparsity(B_nonsquare)
    
    result_custom, custom_time = time_execution(sparse_matrix_multiply, A_nonsquare, B_nonsquare, output_format='sparse', symmetric=False)
    result_scipy, scipy_time = time_execution(A_nonsquare.dot, B_nonsquare)
    
    comparison = compare_execution_times(custom_time, scipy_time)
    
    assert np.allclose(result_custom.A, result_scipy.A), "Results do not match for non-square matrices"
    print(f"\nSparsity: A (500x400): {sparsity_A:.4f}, B (400x500): {sparsity_B:.4f}")
    print(f"Custom Non-square Sparse Execution time: {custom_time:.6f} seconds")
    print(f"SciPy Non-square Sparse Execution time: {scipy_time:.6f} seconds")
    print(f"Difference between results: {np.sum(result_scipy.A - result_custom.A)}")
    print(comparison)

def test_sparse_identity_matrix_multiplication():
    size = 500
    A_sparse = sparse_random(size, size, density=0.1, format='csr')
    I_sparse = eye(size, format='csr')
    
    sparsity_A = calculate_sparsity(A_sparse)
    sparsity_I = calculate_sparsity(I_sparse)
    
    result_custom, custom_time = time_execution(sparse_matrix_multiply, A_sparse, I_sparse, output_format='sparse', symmetric=False)
    result_scipy, scipy_time = time_execution(A_sparse.dot, I_sparse)
    
    comparison = compare_execution_times(custom_time, scipy_time)
    
    assert np.allclose(result_custom.A, result_scipy.A), "Results do not match for identity matrix multiplication"
    print(f"\nSparsity: A: {sparsity_A:.4f}, I: {sparsity_I:.4f}")
    print(f"Custom Sparse Identity Multiplication Execution time: {custom_time:.6f} seconds")
    print(f"SciPy Sparse Identity Multiplication Execution time: {scipy_time:.6f} seconds")
    print(f"Difference between results: {np.sum(result_scipy.A - result_custom.A)}")
    print(comparison)

def test_large_sparse_matrix_multiplication():
    size = 1000
    density = 0.01
    A_large = sparse_random(size, size, density=density, format='csr')
    B_large = sparse_random(size, size, density=density, format='csr')
    
    sparsity_A = calculate_sparsity(A_large)
    sparsity_B = calculate_sparsity(B_large)
    
    result_custom, custom_time = time_execution(sparse_matrix_multiply, A_large, B_large, output_format='sparse', symmetric=False)
    result_scipy, scipy_time = time_execution(A_large.dot, B_large)
    
    comparison = compare_execution_times(custom_time, scipy_time)
    
    assert np.allclose(result_custom.A, result_scipy.A), "Results do not match for large sparse matrix multiplication"
    print(f"\nSparsity: A: {sparsity_A:.4f}, B: {sparsity_B:.4f}")
    print(f"Custom Large Sparse Execution time: {custom_time:.6f} seconds")
    print(f"SciPy Large Sparse Execution time: {scipy_time:.6f} seconds")
    print(f"Difference between results: {np.sum(result_scipy.A - result_custom.A)}")
    print(comparison)
