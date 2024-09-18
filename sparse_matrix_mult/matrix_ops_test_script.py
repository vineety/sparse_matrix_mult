# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 05:53:50 2024

@author: vyadav
"""

"""
@file matrix_ops.py
@brief This module provides a Python interface to C-based matrix operations.

This module defines classes and functions for efficient matrix operations,
particularly for sparse matrices. It interfaces with a C library for
performance-critical operations.
"""

import numpy as np
from scipy.sparse import csr_matrix
from matrix_ops import sparse_matrix_multiply

"""
@brief Define test matrices for matrix multiplication operations
These matrices are used in various multiplication scenarios to demonstrate
the usage of custom sparse matrix routines and comparisons with NumPy.
"""

# Define test matrices in dense format and convert them to CSR sparse format where needed
A_matrix = np.array([
    [0.64, 0.99, 0.89, 0.72],
    [0,    0.67, 0.54, 0   ],
    [0,    0.32, 0,    0   ],
    [0.1,  0,    0,    0   ]
])
matrix_a = csr_matrix(A_matrix)

B_matrix = np.array([
    [0.23, 0,    0,    0.51],
    [0,    0.72, 0,    0   ],
    [0,    0,    0.99, 0   ],
    [0,    0.76, 0.87, 0.97]
])

C_matrix = np.array([
    [0.64, 0.99, 0.89, 0.72],
    [0,    0.67, 0.54, 0   ],
    [0,    0.32, 0,    0   ]
])

D_matrix = np.array([
    [0.64, 0.99, 0.89],
    [0,    0.67, 0.54],
    [0,    0.32, 0   ],
    [0.1,  0,    0   ]
])

matrix_a = csr_matrix(A_matrix)
matrix_b = csr_matrix(B_matrix)
matrix_c = csr_matrix(C_matrix)
matrix_d = csr_matrix(D_matrix)

print("\nDemonstration Matrices:")
print("A_matrix:")
print(A_matrix)
print("\nB_matrix:")
print(B_matrix)
print("\nC_matrix:")
print(C_matrix)
print("\nD_matrix:")
print(D_matrix)

#%% Example 1: Using the installed package
"""
@brief Example 1: Perform sparse matrix multiplications using the custom C-based library
This example demonstrates multiplying two sparse matrices using the custom sparse matrix 
multiplication routine and compares the result with NumPy's built-in matrix multiplication.
"""
print(" Test Examples from the installed library or package")
print("\nNote Matrices need to be stored in compressed sparse row format")
print("---------------------------------------")

# Multiply two sparse matrices, resulting in a compressed sparse row matrix
result_sparse = sparse_matrix_multiply(matrix_a, matrix_d, output_format='sparse', symmetric=False)
print("---------------------------------------")
print("\nResult of custom sparse x sparse matrix multiplication resulting in sparse matrix A * D:")
print("---------------------------------------")
print(result_sparse.toarray())

# Multiply two dense matrices using NumPy
numpy_result = np.dot(A_matrix, D_matrix)
print("---------------------------------------")
print("Result of NumPy A * D:")
print("---------------------------------------")
print(numpy_result)

# Check if the results from the custom routine and NumPy match
if np.allclose(numpy_result, result_sparse.toarray()):
    print("\nThe results from NumPy and the custom routine match!")
else:
    print("\nThere are differences between the NumPy result and the custom routine result.")

#%% Multiply two sparse matrices, resulting in a dense matrix
"""
@brief Multiply two sparse matrices that result in a dense matrix
This section multiplies two sparse matrices and compares the result with NumPy.
"""
result_dense = sparse_matrix_multiply(matrix_a, matrix_d, output_format='dense', symmetric=False)
print("---------------------------------------")
print("\nResult of custom sparse x sparse matrix multiplication resulting in dense matrix A * D:")
print(result_dense)
print("---------------------------------------")
print("Result of NumPy A * D:")
print("---------------------------------------")
print(numpy_result)

# Check if the results from the custom routine and NumPy match
if np.allclose(numpy_result, result_dense):
    print("\nThe results from NumPy and the custom routine match!")
else:
    print("\nThere are differences between the NumPy result and the custom routine result.")

#%% Multiply sparse matrices that result in a symmetric dense matrix
"""
@brief Multiply two sparse matrices, calculating only the upper triangular portion
This method is used when the output matrix is symmetric, even if the input matrices are not.
"""
result_sym_dense = sparse_matrix_multiply(matrix_c, matrix_d, output_format='dense', symmetric=True)
print("\nResult of dense symmetric C * D (upper triangular):")
print(result_sym_dense)

print("---------------------------------------")

# Multiply using NumPy for comparison
numpy_result = np.dot(C_matrix, D_matrix)
print("Result of NumPy C * D:")
print("---------------------------------------")
print(numpy_result)

# Check if the upper triangular part of the results match
if np.allclose(np.triu(numpy_result), result_sym_dense):
    print("\nThe results from NumPy and the custom routine match!")
else:
    print("\nThere are differences between the NumPy result and the custom routine result.")

#%% Multiply two sparse matrices, resulting in a sparse symmetric matrix
"""
@brief Multiply two sparse matrices resulting in a symmetric sparse matrix
This routine calculates only the upper triangular portion of the matrix.
"""
result_sym_sparse = sparse_matrix_multiply(matrix_c, matrix_d, output_format='sparse', symmetric=True)
print("\nResult of sparse symmetric C * D (upper triangular):")
print(np.triu(result_sym_sparse.toarray()))

# Check if the results match
if np.allclose(np.triu(numpy_result), result_sym_sparse.toarray()):
    print("\nThe results from NumPy and the custom routine match!")
else:
    print("\nThere are differences between the NumPy result and the custom routine result.")

#%% Multiply three sparse matrices using the triple product method
"""
@brief Multiply three sparse matrices resulting in a dense symmetric matrix
This section demonstrates how to multiply A * B * A', where B is a covariance matrix, 
and only the upper triangular part of the resulting matrix is computed.
"""
result_triple_sym_dense = sparse_matrix_multiply(matrix_c, matrix_a, output_format='sparse', use_triple_product=True, compute_full_matrix=None)
print("\nResult of sparse symmetric C * A * C' (upper triangular):")
print(result_triple_sym_dense)

print("---------------------------------------")

# Multiply using NumPy for comparison
numpy_result = C_matrix @ A_matrix @ C_matrix.T
print("Result of NumPy C * A * C':")
print("---------------------------------------")
print(numpy_result)

# Check if the upper triangular part of the results match
if np.allclose(np.triu(numpy_result), result_triple_sym_dense):
    print("\nThe results from NumPy and the custom routine match!")
else:
    print("\nThere are differences between the NumPy result and the custom routine result.")


