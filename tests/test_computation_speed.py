import pytest
import numpy as np
import time
from scipy.sparse import random as sparse_random, csr_matrix
from scipy import stats
from sparse_matrix_mult import sparse_matrix_multiply

@pytest.fixture
#%%
def setup_matrices():
    def _setup_matrices(rowsA=500, colsA=500, rowsB=500, colsB=500, density=0.3):
        sparse_matrix_A = sparse_random(rowsA, colsA, density=density, random_state=42, data_rvs=stats.uniform().rvs)
        sparse_matrix_B = sparse_random(rowsB, colsB, density=density, random_state=43, data_rvs=stats.uniform().rvs)
        return csr_matrix(sparse_matrix_A), csr_matrix(sparse_matrix_B)
    return _setup_matrices

def time_execution(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time
#%%
def compare_execution_times(custom_time, scipy_time):
    if custom_time == 0 and scipy_time == 0:
        return "Both methods completed in negligible time."
    elif custom_time == 0:
        return "Custom routine was faster than SciPy."
    elif scipy_time == 0:
        return "SciPy was faster than custom routine."
    else:
        ratio = scipy_time / custom_time if custom_time > 0 else float('inf')
        message = f"Time ratio (SciPy / Custom): {ratio:.2f}"
        if ratio >= 5:
            message += "\nWARNING: SciPy took significantly more time than custom routine."
        return message
#%%
def test_sparse_sparse_non_symmetric(setup_matrices):
    csr_matrix_A, csr_matrix_B = setup_matrices()
    
    result_custom, custom_time = time_execution(sparse_matrix_multiply, csr_matrix_A, csr_matrix_B, output_format='sparse', symmetric=False)
    result_scipy, scipy_time = time_execution(csr_matrix_A.dot, csr_matrix_B)
    
    assert np.allclose(result_custom.toarray(), result_scipy.toarray()), "Results do not match."
    print(compare_execution_times(custom_time, scipy_time))

def test_sparse_sparse_symmetric(setup_matrices):
    csr_matrix_A, csr_matrix_B = setup_matrices()
    
    result_custom, custom_time = time_execution(sparse_matrix_multiply, csr_matrix_A, csr_matrix_B, output_format='sparse', symmetric=True)
    result_scipy, scipy_time = time_execution(csr_matrix_A.dot, csr_matrix_B)
    
    result_scipy_dense = result_scipy.toarray()
    assert np.allclose(np.triu(result_scipy_dense), np.triu(result_custom.toarray())), "Upper triangular parts do not match."
    print(compare_execution_times(custom_time, scipy_time))

def test_sparse_dense_symmetric(setup_matrices):
    csr_matrix_A, csr_matrix_B = setup_matrices()
    
    result_custom, custom_time = time_execution(sparse_matrix_multiply, csr_matrix_A, csr_matrix_B, output_format='dense', symmetric=True)
    result_scipy_sparse, scipy_time = time_execution(csr_matrix_A.dot, csr_matrix_B)
    result_scipy_dense = result_scipy_sparse.toarray()
    
    assert np.allclose(np.triu(result_scipy_dense), np.triu(result_custom)), "Upper triangular parts do not match."
    print(compare_execution_times(custom_time, scipy_time))

def test_sparse_dense_non_symmetric(setup_matrices):
    csr_matrix_A, csr_matrix_B = setup_matrices()
    
    result_custom, custom_time = time_execution(sparse_matrix_multiply, csr_matrix_A, csr_matrix_B, output_format='dense', symmetric=False)
    result_scipy_sparse, scipy_time = time_execution(csr_matrix_A.dot, csr_matrix_B)
    result_scipy_dense = result_scipy_sparse.toarray()
    
    assert np.allclose(result_scipy_dense, result_custom), "Results do not match."
    print(compare_execution_times(custom_time, scipy_time))

def test_triple_product(setup_matrices):
    csr_matrix_A, csr_matrix_B = setup_matrices()
    
    result_custom, custom_time = time_execution(sparse_matrix_multiply, csr_matrix_A, csr_matrix_B, use_triple_product=True, compute_full_matrix=0)
    result_scipy_sparse, scipy_time_part1 = time_execution(csr_matrix_A.dot, csr_matrix_B)
    result_scipy_dense, scipy_time_part2 = time_execution(result_scipy_sparse.dot, csr_matrix_A.transpose())
    
    result_scipy_dense = result_scipy_dense.toarray()
    scipy_time = scipy_time_part1 + scipy_time_part2
    
    assert np.allclose(np.triu(result_scipy_dense), np.triu(result_custom)), "Upper triangular parts do not match."
    print(compare_execution_times(custom_time, scipy_time))
#%%    
if __name__ == "__main__":
    test_matrices=setup_matrices()
    test_sparse_sparse_non_symmetric(test_matrices)
    test_sparse_sparse_symmetric(test_matrices)
    test_sparse_dense_symmetric(test_matrices)
    test_sparse_dense_non_symmetric(test_matrices)
    test_triple_product(test_matrices)(test_matrices)
