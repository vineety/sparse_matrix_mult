#ifndef MATRIX_DEFINITIONS_H
#define MATRIX_DEFINITIONS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

	// Structure for sparse matrix in CSR format
	struct sparsemat {
		int nzmax;    // number of non-zeros in the sparse matrix
		int rows;     // number of rows in the sparse matrix
		int cols;     // number of columns in the sparse matrix
		int* rowPtr;  // cumulative number of non-zero entries in each row
		int* colInd;  // column indices for non-zero entries
		double* values; // non-zero entries in the sparse matrix
	};

	// Structure for dense double array
	struct darray {
		double* array;
		int rows;
		int cols;
	};

	// Structure for dense integer array
	struct iarray {
		int* array;
		int rows;
		int cols;
	};

#ifdef __cplusplus
}
#endif

#endif // MATRIX_DEFINITIONS_H
