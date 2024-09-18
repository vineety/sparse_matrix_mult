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
    int nzmax;    // Maximum number of non-zero elements in the sparse matrix
    int rows;     // Number of rows in the sparse matrix
    int cols;     // Number of columns in the sparse matrix
    int* rowPtr;  // Array that holds cumulative counts of non-zero elements in each row
    int* colInd;  // Array that stores the column indices for each non-zero element
    double* values; // Array that holds the actual non-zero values of the matrix
};

// Structure definition for a dense matrix stored as a double array
struct darray {
    double* array;  // Pointer to a 1D array storing the matrix elements
    int rows;       // Number of rows in the dense matrix
    int cols;       // Number of columns in the dense matrix
};

// Structure definition for an integer array, which can be used for storing row/column indices or counts
struct iarray {
    int* array;  // Pointer to a 1D array storing the integer values
    int rows;    // Number of rows in the array (used for 2D arrays)
    int cols;    // Number of columns in the array (used for 2D arrays)
};

#ifdef __cplusplus
}
#endif

#endif // MATRIX_DEFINITIONS_H
