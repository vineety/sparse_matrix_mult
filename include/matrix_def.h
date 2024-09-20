#ifndef MATRIX_DEFINITIONS_H
#define MATRIX_DEFINITIONS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>  // Include OpenMP headers if parallel processing is enabled
#endif

// Structure definition for a sparse matrix in Compressed Sparse Row (CSR) format
struct sparsemat {
    size_t nzmax;    // Maximum number of non-zero elements in the sparse matrix
    size_t rows;     // Number of rows in the sparse matrix
    size_t cols;     // Number of columns in the sparse matrix
    int* rowPtr;  // Array that holds cumulative counts of non-zero elements in each row
    int* colInd;  // Array that stores the column indices for each non-zero element
    double* values; // Array that holds the actual non-zero values of the matrix
};

// Structure definition for a dense matrix stored as a double array
struct darray {
    double* array;  // Pointer to a 1D array storing the matrix elements
    size_t rows;       // Number of rows in the dense matrix
    size_t cols;       // Number of columns in the dense matrix
};

// Structure definition for an integer array, which can be used for storing row/column indices or counts
struct iarray {
    int* array;  // Pointer to a 1D array storing the integer values
    size_t rows;    // Number of rows in the array (used for 2D arrays)
    size_t cols;    // Number of columns in the array (used for 2D arrays)
};

#ifdef __cplusplus
}
#endif

#endif // MATRIX_DEFINITIONS_H
