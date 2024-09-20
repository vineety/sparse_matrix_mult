#include <stdio.h>       // Standard I/O functions
#include <stdlib.h>      // Standard library functions for memory allocation and free
#include <string.h>      // String manipulation functions
#include <math.h>        // Math library functions
#ifdef _OPENMP         // If OpenMP is enabled, include OpenMP header
#include <omp.h>
#endif
#include "matrix_def.h"   // Custom header file that defines matrix structures
#include "functions.h"    // Custom header file where function declarations are made

// Implementation of functions declared in the header

/**
 * @brief Frees the memory associated with a sparse matrix (sparsemat structure).
 *
 * This function frees the dynamically allocated arrays within the sparse matrix
 * structure and resets the pointers to NULL. It also resets the matrix dimensions
 * and nzmax (number of non-zero elements) to 0.
 *
 * @param matrix Pointer to the sparsemat structure that needs to be destroyed.
 */
void destroy_sparsemat(struct sparsemat* matrix) {
    if (matrix != NULL) {    // Check if the matrix pointer is valid (not NULL)
        free(matrix->rowPtr); // Free the row pointer array
        free(matrix->colInd); // Free the column indices array
        free(matrix->values); // Free the values array (non-zero elements)
        
        // Reset all pointers to NULL and set dimensions to zero
        matrix->rowPtr = matrix->colInd = NULL;
        matrix->values = NULL;
        matrix->nzmax = matrix->rows = matrix->cols = 0;
    }
}

/**
 * @brief Frees the memory associated with an integer array (iarray structure).
 *
 * Similar to destroy_sparsemat, this function frees the dynamically allocated
 * array inside the iarray structure and resets the dimensions and array pointer to NULL.
 *
 * @param matrix Pointer to the iarray structure that needs to be destroyed.
 */
void destroy_iarray(struct iarray* matrix) {
    if (matrix != NULL) {     // Check if the pointer is valid
        free(matrix->array);  // Free the array holding the integer elements
        matrix->array = NULL; // Reset pointer to NULL
        matrix->rows = matrix->cols = 0; // Reset dimensions to zero
    }
}

/**
 * @brief Frees the memory associated with a double array (darray structure).
 *
 * Similar to destroy_iarray, this function frees the dynamically allocated
 * array inside the darray structure and resets the dimensions and array pointer to NULL.
 *
 * @param matrix Pointer to the darray structure that needs to be destroyed.
 */
void destroy_darray(struct darray* matrix) {
    if (matrix != NULL) {      // Check if the pointer is valid
        free(matrix->array);   // Free the array holding the double elements
        matrix->array = NULL;  // Reset pointer to NULL
        matrix->rows = matrix->cols = 0; // Reset dimensions to zero
    }
}

/**
 * @brief Modifies the allocated size of the sparse matrix structure.
 *
 * This function reallocates memory for the column indices and values arrays to match the new size
 * specified by the user. If the new size is less than or equal to 0, the function frees the
 * existing memory instead of reallocating.
 *
 * @param matrix Pointer to the sparsemat structure that needs reallocation.
 * @param new_size The new size for the column indices and values arrays.
 */
void modifyalloc(struct sparsemat* matrix, int new_size) {
    if (new_size <= 0) {
        // If new_size is <= 0, free the memory instead of reallocating
        free(matrix->colInd); // Free column indices array
        free(matrix->values); // Free values array
        matrix->colInd = NULL; // Set pointers to NULL
        matrix->values = NULL;
        return; // Exit function
    }

    // Reallocate memory for column indices and values arrays
    int* new_colInd = (int*)realloc(matrix->colInd, new_size * sizeof(int)); // Resize colInd
    double* new_values = (double*)realloc(matrix->values, new_size * sizeof(double)); // Resize values

    // Check if reallocation failed
    if (new_colInd == NULL || new_values == NULL) {
        fprintf(stderr, "Reallocation failed.\n"); // Print error message
        // Free newly allocated memory in case realloc moved the data
        free(new_colInd);
        free(new_values);
        return; // Exit function as reallocation failed
    }

    // Update the pointers in the sparsemat structure
    matrix->colInd = new_colInd;
    matrix->values = new_values;
}

/**
 * @brief Creates a sparse matrix structure and allocates memory for its internal arrays.
 *
 * This function initializes a sparsemat structure with the specified number of rows,
 * columns, and non-zero elements (nzmax). It allocates memory for the row pointer, column
 * indices, and values arrays.
 *
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 * @param nzmax The maximum number of non-zero elements in the matrix.
 * @return A pointer to the allocated sparsemat structure or NULL if allocation fails.
 */
struct sparsemat* create_sparsemat(int rows, int cols, int nzmax) {
    // Allocate memory for the sparsemat structure
    struct sparsemat* matrix = (struct sparsemat*)calloc(1, sizeof(struct sparsemat));
    if (matrix != NULL) { // Check if allocation succeeded
        matrix->rows = rows;  // Set the number of rows
        matrix->cols = cols;  // Set the number of columns
        matrix->nzmax = nzmax;  // Set the maximum number of non-zero elements
        
        // Allocate memory for row pointers, column indices, and values arrays
        matrix->colInd = (int*)calloc((size_t)nzmax, sizeof(int));   // Allocate column indices array
        matrix->rowPtr = (int*)calloc((size_t)rows + 1, sizeof(int)); // Allocate row pointer array
        matrix->values = (double*)calloc((size_t)nzmax, sizeof(double)); // Allocate values array
    }
    return matrix; // Return pointer to the newly created sparse matrix
}

/**
 * @brief Creates a darray structure (dense matrix) and allocates memory for its internal array.
 *
 * This function initializes a darray structure with the specified number of rows and columns.
 * It allocates memory for the array that stores the elements of the dense matrix.
 *
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 * @return A pointer to the allocated darray structure or NULL if allocation fails.
 */
struct darray* create_darray(int rows, int cols) {
    // Allocate memory for the darray structure
    struct darray* matrix = (struct darray*)calloc(1, sizeof(struct darray));
    if (matrix != NULL) { // Check if allocation succeeded
        matrix->rows = rows;  // Set the number of rows
        matrix->cols = cols;  // Set the number of columns
        
        // Allocate memory for the array that holds the matrix elements
        matrix->array = (double*)calloc((size_t)rows * (size_t)cols, sizeof(double));
    }
    return matrix; // Return pointer to the newly created dense matrix
}

// More functions can be implemented here as required...

