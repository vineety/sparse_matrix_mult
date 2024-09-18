#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef USE_OPENMP
#include <omp.h>  // Include OpenMP for parallelization
#endif
#include "matrix_def.h"  // Include custom matrix definitions
#include "functions.h"  // Include custom functions

// Function to multiply two sparse matrices in CSR format and produce a dense symmetric upper triangular matrix
void dense_sym(const struct sparsemat* const matrixa, const struct sparsemat* const matrixb, struct darray* const matrixc)
/* This routine multiplies two sparse matrices in CSR format
with zero-based indexing and produces a dense upper triangular
matrix based on the assumption that multiplication of two sparse
matrices results in a symmetric matrix */
{
    int i, j, k, col_num_a, col_num_b, konstant;  // Declare variables for loop indices, column indices, and constants
    double value;  // Temporary variable to hold matrix values during computation
    
    // Set matrixc dimensions to match the result of multiplying matrixa and matrixb
    matrixc->rows = matrixa->rows;
    matrixc->cols = matrixb->cols;
    
    // Allocate memory for the dense result matrix (upper triangular matrix)
    matrixc->array = (double*)calloc(matrixa->rows * matrixb->cols, sizeof(double));
    
#ifdef USE_OPENMP
# pragma omp parallel for private (i,j,k,value,col_num_a,col_num_b,konstant)  // Parallelize the outer loop using OpenMP
#endif
    // Iterate over the rows of matrixa
    for (i = 0; i < matrixa->rows; i++)
    {
        konstant = i * matrixa->rows;  // Calculate the starting index in the dense matrix for row i
        
        // Loop through the non-zero elements of row i in matrixa
        for (j = matrixa->rowPtr[i]; j <= matrixa->rowPtr[i + 1] - 1; j++)
        {
            value = matrixa->values[j];  // Get the value of the current element in matrixa
            col_num_a = matrixa->colInd[j];  // Get the column index of the current element in matrixa
            
            // Multiply the current element of matrixa with the corresponding elements of matrixb
            for (k = matrixb->rowPtr[col_num_a]; k <= matrixb->rowPtr[col_num_a + 1] - 1; k++)
            {
                col_num_b = matrixb->colInd[k];  // Get the column index of the current element in matrixb
                
                // Update the upper triangular part of the result matrix only if i <= col_num_b
                if (i <= col_num_b)
                    matrixc->array[konstant + col_num_b] += value * matrixb->values[k];
            }
        }
    }
}

// Function to multiply two sparse matrices in CSR format and produce a non-symmetric dense matrix
void dense_nosym(const struct sparsemat* const matrixa, const struct sparsemat* const matrixb, struct darray* const matrixc) {
    // Check for matrix dimension compatibility for multiplication
    if (matrixa->cols != matrixb->rows) {
        fprintf(stderr, "Error: Matrix dimensions are incompatible for multiplication.\n");
        return;
    }
  
    // Set matrixc dimensions to match the result of multiplying matrixa and matrixb
    matrixc->rows = matrixa->rows;
    matrixc->cols = matrixb->cols;
    
    // Allocate memory for the dense result matrix
    matrixc->array = (double*)calloc((size_t)matrixc->rows * (size_t)matrixc->cols, sizeof(double));
    if (matrixc->array == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for matrixc->array.\n");
        return;
    }
    
    int memory_error_flag = 0;  // Flag to track memory allocation errors for thread-local arrays
#ifdef USE_OPENMP
#pragma omp parallel shared(memory_error_flag)  // Start parallel region with shared memory_error_flag
#endif
{
    // Allocate a local result array for each thread to avoid race conditions
    double* local_result = (double*)calloc(matrixc->cols, sizeof(double));
    if (local_result == NULL) {
        // Handle memory allocation error for local_result
#ifdef USE_OPENMP
        fprintf(stderr, "Error: Memory allocation for local result failed in thread %d.\n", omp_get_thread_num());
#else
        fprintf(stderr, "Error: Memory allocation for result failed");
#endif
        memory_error_flag = 1;  // Set the memory error flag
    }
    else {
        int i, j, k, col_num_a, col_num_b;  // Declare variables for loop indices and column indices
        double value;  // Temporary variable to hold matrix values during computation

#ifdef USE_OPENMP
#pragma omp for schedule(guided)  // Use guided scheduling for the parallel loop
#endif
        // Iterate over the rows of matrixa
        for (i = 0; i < matrixa->rows; i++) {
            if (!memory_error_flag) {  // Only execute if no memory error has occurred
                memset(local_result, 0, matrixc->cols * sizeof(double));  // Initialize the local result array for the current row
            
                // Loop through the non-zero elements of row i in matrixa
                for (j = matrixa->rowPtr[i]; j < matrixa->rowPtr[i + 1]; j++) {
                    value = matrixa->values[j];  // Get the value of the current element in matrixa
                    col_num_a = matrixa->colInd[j];  // Get the column index of the current element in matrixa

#ifdef USE_OPENMP
#pragma omp simd  // SIMD parallelization for the inner loop (vectorization)
#endif
                    // Multiply the current element of matrixa with the corresponding elements of matrixb
                    for (k = matrixb->rowPtr[col_num_a]; k < matrixb->rowPtr[col_num_a + 1]; k++) {
                        col_num_b = matrixb->colInd[k];  // Get the column index of the current element in matrixb
                        local_result[col_num_b] += value * matrixb->values[k];  // Update the local result array
                    }
                }
            
                // Copy the local result for the current row into the global result array (matrixc)
                memcpy(&matrixc->array[i * matrixc->cols], local_result, matrixc->cols * sizeof(double));
            }
        }
    
        free(local_result);  // Free the memory allocated for the local result array
    }
}

// Handle memory error by freeing the allocated memory for matrixc and setting its array to NULL
if (memory_error_flag) {
    free(matrixc->array);
    matrixc->array = NULL;
    return;
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
#ifdef USE_OPENMP
  int num_threads = omp_get_max_threads();
#else
  int num_threads = 1;  // Single thread if OpenMP is not enabled
#endif
  
  // Allocate a thread-local version of matrix C for each thread to avoid race conditions
  double** thread_local_C = (double**)malloc(num_threads * sizeof(double*));
  for (int t = 0; t < num_threads; ++t) {
    thread_local_C[t] = (double*)calloc(n * n, sizeof(double));  // Allocate local result matrix for each thread
  }

#ifdef USE_OPENMP
#pragma omp parallel
#endif
{
#ifdef USE_OPENMP
  int thread_id = omp_get_thread_num();  // Get the thread ID for this thread
#else
  int thread_id = 0;  // Single thread if OpenMP is not enabled
#endif
  double* temp_values = (double*)calloc(k, sizeof(double));  // Temporary array for storing intermediate results
  double* local_C = thread_local_C[thread_id];  // Thread-local matrix C
  
  // Iterate over each row i of matrix H
#ifdef USE_OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
  for (int i = 0; i < n; i++) {
    // Loop over non-zero elements of row i of matrix H
    for (int jp = H->rowPtr[i]; jp < H->rowPtr[i + 1]; jp++) {
      int j = H->colInd[jp];  // Column index for H
      double h_ij = H->values[jp];  // Value at H(i, j)
      
      // Multiply H(i, j) with corresponding row in Q
#ifdef USE_OPENMP
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
#ifdef USE_OPENMP
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
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
for (int i = 0; i < C->rows; i++) {
  for (int j = 0; j < C->cols; j++) {
    double sum = 0.0;
    
    // Sum the corresponding entries from all thread-local matrices
#ifdef USE_OPENMP
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



