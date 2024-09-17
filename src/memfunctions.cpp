#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include "matrix_def.h"
#include "functions.h"  // Include the header file we just created

// Implementation of functions declared in the header

void destroy_sparsemat(struct sparsemat* matrix) {
    if (matrix != NULL) {
        free(matrix->rowPtr);
        free(matrix->colInd);
        free(matrix->values);
        matrix->rowPtr = matrix->colInd = NULL;
        matrix->values = NULL;
        matrix->nzmax = matrix->rows = matrix->cols = 0;
    }
}

void destroy_iarray(struct iarray* matrix) {
    if (matrix != NULL) {
        free(matrix->array);
        matrix->array = NULL;
        matrix->rows = matrix->cols = 0;
    }
}

void destroy_darray(struct darray* matrix) {
    if (matrix != NULL) {
        free(matrix->array);
        matrix->array = NULL;
        matrix->rows = matrix->cols = 0;
    }
}

void modifyalloc(struct sparsemat* matrix, int new_size) {
    if (new_size <= 0) {
        // Free the memory if new_size is <= 0
        free(matrix->colInd);
        free(matrix->values);
        matrix->colInd = NULL;
        matrix->values = NULL;
        return;
    }

    // Attempt to reallocate colInd and values
    int* new_colInd = (int*)realloc(matrix->colInd, new_size * sizeof(int));
    double* new_values = (double*)realloc(matrix->values, new_size * sizeof(double));

    // Check if realloc failed
    if (new_colInd == NULL || new_values == NULL) {
        fprintf(stderr, "Reallocation failed.\n");
        // Free existing memory to avoid memory leak
        free(new_colInd); // Use new_colInd in case realloc moved the data
        free(new_values);
        return;
    }

    // Update the pointers if reallocation succeeded
    matrix->colInd = new_colInd;
    matrix->values = new_values;
}


struct sparsemat* create_sparsemat(int rows, int cols, int nzmax) {
    struct sparsemat* matrix = (struct sparsemat*)calloc(1, sizeof(struct sparsemat));
    if (matrix != NULL) {
        matrix->rows = rows;
        matrix->cols = cols;
        matrix->nzmax = nzmax;
        matrix->colInd = (int*)calloc((size_t)nzmax, sizeof(int));
        matrix->rowPtr = (int*)calloc((size_t)rows + 1, sizeof(int));
        matrix->values = (double*)calloc((size_t)nzmax, sizeof(double));
    }
    return matrix;
}

struct darray* create_darray(int rows, int cols) {
    struct darray* matrix = (struct darray*)calloc(1, sizeof(struct darray));
    if (matrix != NULL) {
        matrix->rows = rows;
        matrix->cols = cols;
        matrix->array = (double*)calloc((size_t)rows * (size_t)cols, sizeof(double));
    }
    return matrix;
}

// Implement the rest of the functions declared in the header...
