import numpy as np
from scipy.sparse import csr_matrix, random
import time

# Import your sparse_matrix_multiply function here
from sparse_matrix_mult import sparse_matrix_multiply
#%%
def setup_matrices(size=500, density=0.01):
    A = random(size, size, density=density, format='csr')
    B = random(size, size, density=density, format='csr')
    return A, B

def time_execution(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def compare_execution_times(custom_time, scipy_time):
    speedup = scipy_time / custom_time
    return f"Custom implementation: {custom_time:.6f} seconds\n" \
           f"SciPy implementation: {scipy_time:.6f} seconds\n" \
           f"Speedup: {speedup:.2f}x"

def test_triple_product():
    csr_matrix_A, csr_matrix_B = setup_matrices()
    
    print("Matrix A shape:", csr_matrix_A.shape)
    print("Matrix B shape:", csr_matrix_B.shape)
    
    result_custom, custom_time = time_execution(sparse_matrix_multiply, csr_matrix_A, csr_matrix_B, use_triple_product=True, compute_full_matrix=0)
    print("Custom result shape:", result_custom.shape)
    print("Custom result type:", type(result_custom))
    
    result_scipy_sparse, scipy_time_part1 = time_execution(csr_matrix_A.dot, csr_matrix_B)
    result_scipy_dense, scipy_time_part2 = time_execution(result_scipy_sparse.dot, csr_matrix_A.transpose())
    
    result_scipy_dense = result_scipy_dense.toarray()
    scipy_time = scipy_time_part1 + scipy_time_part2
    
    print("SciPy result shape:", result_scipy_dense.shape)
    
    # Ensure both results are 2D arrays
    assert result_custom.ndim == 2, f"Custom result should be 2D, but has {result_custom.ndim} dimensions"
    assert result_scipy_dense.ndim == 2, f"SciPy result should be 2D, but has {result_scipy_dense.ndim} dimensions"
    
    # Create upper triangular mask
    mask = np.triu(np.ones(result_custom.shape, dtype=bool))
    
    # Apply mask to both results
    custom_upper = result_custom[mask]
    scipy_upper = result_scipy_dense[mask]
    
    is_close = np.allclose(custom_upper, scipy_upper, rtol=1e-5, atol=1e-8)
    print("Upper triangular parts (including diagonal) match:", is_close)
    
    if not is_close:
        print("Max absolute difference:", np.max(np.abs(custom_upper - scipy_upper)))
        print("Mean absolute difference:", np.mean(np.abs(custom_upper - scipy_upper)))
    
    print(compare_execution_times(custom_time, scipy_time))

if __name__ == "__main__":
    test_triple_product()
