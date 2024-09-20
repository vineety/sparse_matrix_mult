#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>  // Include OpenMP for parallelization
#endif
#include "matrix_def.h"  // Include custom matrix definitions
#include "functions.h"  // Include custom functions

// Function to multiply two sparse matrices in CSR format and produce a dense symmetric upper triangular matrix
// Optimized function to multiply two sparse matrices in CSR format and produce a dense symmetric upper triangular matrix
void dense_sym(const struct sparsemat* const matrixa, const struct sparsemat* const matrixb,
               struct darray* const matrixc)
{
    // Variables for loop indices and calculations
    int i, j, k;
    int col_num_a, col_num_b;
    double value;

    // Set matrixc dimensions to match the result of multiplying matrixa and matrixb
    matrixc->rows = matrixa->rows;
    matrixc->cols = matrixb->cols;

    // Calculate the total number of elements in the result matrix
    size_t matrix_size = (size_t)matrixc->rows * matrixc->cols;

    // Allocate memory for the dense result matrix (flattened array)
    // Using calloc to initialize all elements to zero
    matrixc->array = (double*)calloc(matrix_size, sizeof(double));
    if (!matrixc->array) {
        fprintf(stderr, "Memory allocation failed for result matrix\n");
        return;
    }

    // OpenMP parallelization over the outer loop (rows of matrixa)
    #ifdef _OPENMP
    #pragma omp parallel for private(i, j, k, value, col_num_a, col_num_b) schedule(static)
    #endif
    for (i = 0; i < matrixa->rows; i++) {
        // Compute the starting index for row i in the flattened result matrix
        size_t row_offset = (size_t)i * matrixc->cols;

        // Loop through non-zero elements in row i of matrixa
        for (j = matrixa->rowPtr[i]; j < matrixa->rowPtr[i + 1]; j++) {
            value = matrixa->values[j];       // Value of the current element in matrixa
            col_num_a = matrixa->colInd[j];   // Column index of the current element in matrixa

            // Access the non-zero elements in row col_num_a of matrixb
            int row_start_b = matrixb->rowPtr[col_num_a];
            int row_end_b = matrixb->rowPtr[col_num_a + 1];

            // Loop through non-zero elements in the corresponding row of matrixb
            for (k = row_start_b; k < row_end_b; k++) {
                col_num_b = matrixb->colInd[k];    // Column index in matrixb
                double product = value * matrixb->values[k]; // Compute the product once

                // Since the result matrix is symmetric, we only need to compute the upper triangular part
                if (i <= col_num_b) {
                    // Update the result matrix at position (i, col_num_b)
                    matrixc->array[row_offset + col_num_b] += product;
                }
                // Optional: If you need the full symmetric matrix, uncomment the following lines
                /*
                else {
                    // Update the symmetric position in the lower triangular part
                    size_t sym_index = (size_t)col_num_b * matrixc->cols + i;
                    matrixc->array[sym_index] += product;
                }
                */
            }
        }
    }
}



// Function to multiply two sparse matrices in CSR format and produce a non-symmetric dense matrix
void dense_nosym(const struct sparsemat* const matrixa, const struct sparsemat* const matrixb,
                 struct darray* const matrixc)
{
    // Check for matrix dimension compatibility for multiplication
    if (matrixa->cols != matrixb->rows) {
        fprintf(stderr, "Error: Matrix dimensions are incompatible for multiplication.\n");
        return;
    }
  
    // Set matrixc dimensions to match the result of multiplying matrixa and matrixb
    matrixc->rows = matrixa->rows;
    matrixc->cols = matrixb->cols;
    
    // Calculate the total number of elements in the result matrix
    size_t matrix_size = (size_t)matrixc->rows * (size_t)matrixc->cols;

    // Allocate memory for the dense result matrix (flattened array)
    // Using calloc to initialize all elements to zero
    matrixc->array = (double*)calloc(matrix_size, sizeof(double));
    if (matrixc->array == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for matrixc->array.\n");
        return;
    }

    // OpenMP parallelization over the outer loop (rows of matrixa)
    // Each thread works on a separate row to avoid race conditions
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < matrixa->rows; i++) {
        // Compute the starting index for row i in the flattened result matrix
        size_t row_offset = (size_t)i * matrixc->cols;

        // Loop through non-zero elements in row i of matrixa
        for (int j = matrixa->rowPtr[i]; j < matrixa->rowPtr[i + 1]; j++) {
            double value = matrixa->values[j];     // Value of the current element in matrixa
            int col_num_a = matrixa->colInd[j];    // Column index of the current element in matrixa

            // Access the non-zero elements in row col_num_a of matrixb
            int row_start_b = matrixb->rowPtr[col_num_a];
            int row_end_b = matrixb->rowPtr[col_num_a + 1];

            // Loop through non-zero elements in the corresponding row of matrixb
            for (int k = row_start_b; k < row_end_b; k++) {
                int col_num_b = matrixb->colInd[k];     // Column index in matrixb
                double product = value * matrixb->values[k]; // Compute the product

                // Update the result matrix at position (i, col_num_b)
                matrixc->array[row_offset + col_num_b] += product;
            }
        }
    }
}


/*
 * Function: triple_product
 * ------------------------
 * This function performs the triple matrix product H * Q * H' (H-transposed).
 * The matrices H and Q are sparse matrices in CSR (Compressed Sparse Row) format,
 * and the result is stored in C, a dense matrix.*/
 
void triple_product(struct sparsemat* H, struct sparsemat* Q, struct darray* C, int compute_full_matrix) {
  int n = H->rows;  // Number of rows in H (and also the resulting matrix C)
  int k = H->cols;  // Number of columns in H (and rows in Q)
  
  // Set the dimensions of the output matrix C (n x n)
  C->rows = n;
  C->cols = n;
  
  // Pre-allocate memory for C->array (dense matrix, n x n)
  C->array = (double*)calloc(C->rows * C->cols, sizeof(double));
  if (C->array == NULL) {
    fprintf(stderr, "Memory allocation failed for C->array\n");
    return;
  }
  
  // Get the number of available threads for parallel execution
#ifdef _OPENMP
  int num_threads = omp_get_max_threads();
#else
  int num_threads = 1;  // Single thread if OpenMP is not enabled
#endif
  
  // Allocate a thread-local version of matrix C for each thread to avoid race conditions
  double** thread_local_C = (double**)malloc(num_threads * sizeof(double*));
  for (int t = 0; t < num_threads; ++t) {
    thread_local_C[t] = (double*)calloc(n * n, sizeof(double));  // Allocate local result matrix for each thread
  }

#ifdef _OPENMP
#pragma omp parallel
#endif
{
#ifdef _OPENMP
  int thread_id = omp_get_thread_num();  // Get the thread ID for this thread
#else
  int thread_id = 0;  // Single thread if OpenMP is not enabled
#endif
  double* temp_values = (double*)calloc(k, sizeof(double));  // Temporary array for storing intermediate results
  double* local_C = thread_local_C[thread_id];  // Thread-local matrix C
  
  // Iterate over each row i of matrix H
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
  for (int i = 0; i < n; i++) {
    // Loop over non-zero elements of row i of matrix H
    for (int jp = H->rowPtr[i]; jp < H->rowPtr[i + 1]; jp++) {
      int j = H->colInd[jp];  // Column index for H
      double h_ij = H->values[jp];  // Value at H(i, j)
      
      // Multiply H(i, j) with corresponding row in Q
#ifdef _OPENMP
#pragma omp simd
#endif
      for (int kp = Q->rowPtr[j]; kp < Q->rowPtr[j + 1]; kp++) {
        temp_values[Q->colInd[kp]] += h_ij * Q->values[kp];  // Accumulate intermediate results
      }
    }
    
    // Process each column of the result matrix C
    int start_k = compute_full_matrix ? 0 : i;  // For upper triangular part, start from diagonal
    for (int k = start_k; k < n; k++) {
      double sum = 0.0;
      
      // Multiply temp_values by the corresponding row of H and accumulate into local_C
#ifdef _OPENMP
#pragma omp simd reduction(+:sum)
#endif
      for (int jp = H->rowPtr[k]; jp < H->rowPtr[k + 1]; jp++) {
        sum += temp_values[H->colInd[jp]] * H->values[jp];
      }
      local_C[i * n + k] += sum;  // Store result in thread-local C
      if (compute_full_matrix && i != k) {
        local_C[k * n + i] += sum;  // Fill symmetric entry in case full matrix is computed
      }
    }
    
    // Reset the temp_values array for the next row
    memset(temp_values, 0, k * sizeof(double));
  }
  
  free(temp_values);  // Free temp_values for this thread
}

// Reduction step: Combine the thread-local matrices into the final result matrix C
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
for (int i = 0; i < C->rows; i++) {
  for (int j = 0; j < C->cols; j++) {
    double sum = 0.0;
    
    // Sum the corresponding entries from all thread-local matrices
#ifdef _OPENMP
#pragma omp simd reduction(+:sum)
#endif
    for (int t = 0; t < num_threads; ++t) {
      sum += thread_local_C[t][i * C->cols + j];
    }
    C->array[i * C->cols + j] = sum;  // Store the final result in C
  }
}

// Free the thread-local matrices
for (int t = 0; t < num_threads; ++t) {
  free(thread_local_C[t]);
}
free(thread_local_C);
}



