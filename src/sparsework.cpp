#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>  // Include OpenMP for parallel processing
#endif
#include "matrix_def.h"  // Include matrix structure definitions
#include "functions.h"  // Include custom matrix functions

// Function for non-symmetric sparse matrix multiplication between matrixa and matrixb
// Result is stored in matrixc, which is a sub-matrix corresponding to the row range [startIndex, endIndex]
// memIncrease specifies the initial memory allocation size for matrixc
void sparsework_nosym(const struct sparsemat* const matrixa, const struct sparsemat* const matrixb, struct sparsemat* const matrixc, const int startIndex, const int endIndex,
                      int memIncrease)
{
  int i, j, k, col_num_a;  // Loop counters and column number variable for matrixa
  double value;  // Variable to hold the current value during multiplication
  int temp = 0;  // Temporary variable to track the number of non-zero elements
  int* workArray = NULL;  // Work array used to track non-zero values in matrixc for efficient insertion
  
  // Set the size of matrixc based on the row range and the number of columns in matrixb
  matrixc->rows = endIndex - startIndex + 1;  // Number of rows in the smaller sub-matrix
  matrixc->cols = matrixb->cols;  // Columns in the sub-matrix
  matrixc->nzmax = 0;  // Initialize the number of non-zero elements to zero

  // Allocate memory for the row pointers, column indices, and values of the resulting sparse matrix
  matrixc->rowPtr = (int*)calloc((size_t)matrixc->rows, sizeof(int));
  matrixc->colInd = (int*)calloc((size_t)memIncrease, sizeof(int));
  matrixc->values = (double*)calloc((size_t)memIncrease, sizeof(double));

  // Allocate memory for a temporary work array to store intermediate results
  workArray = (int*)calloc((size_t)matrixb->cols, sizeof(int));
  
  // Check for memory allocation failures and handle them appropriately
  if (workArray == NULL || matrixc->rowPtr == NULL || matrixc->colInd == NULL || matrixc->values == NULL) {
    fprintf(stderr, "Memory allocation failed in sparsework_nosym\n");
    free(workArray);
    free(matrixc->rowPtr);
    free(matrixc->colInd);
    free(matrixc->values);
    return;
  }
  
  // Initialize the work array with -1 to indicate that no values have been inserted yet
  memset(workArray, -1, (size_t)matrixb->cols * sizeof(int));

  // Main loop to process rows from startIndex to endIndex in matrixa
  for (i = startIndex; i <= endIndex; ++i)
  {
    // Loop through the non-zero entries of row i in matrixa
    for (j = matrixa->rowPtr[i]; j < matrixa->rowPtr[i + 1]; ++j)
    {
      value = matrixa->values[j];  // Get the value of the current non-zero element in matrixa
      col_num_a = matrixa->colInd[j];  // Get the column index for the current element in matrixa
      
      // Loop through the non-zero entries in the corresponding row of matrixb
      for (k = matrixb->rowPtr[col_num_a]; k < matrixb->rowPtr[col_num_a + 1]; ++k)
      {
        // If the column index already exists in the result matrixc, update the value
        if (workArray[matrixb->colInd[k]] != -1)
        {
          matrixc->values[workArray[matrixb->colInd[k]]] += value * matrixb->values[k];
        }
        // Otherwise, insert a new non-zero entry in matrixc
        else
        {
          matrixc->colInd[matrixc->nzmax] = matrixb->colInd[k];
          workArray[matrixb->colInd[k]] = matrixc->nzmax;
          matrixc->values[matrixc->nzmax] = value * matrixb->values[k];
          matrixc->nzmax += 1;
        }
      }
    }

    // Reset the work array for the next row
    for (k = temp; k < matrixc->nzmax; ++k)
    {
      workArray[matrixc->colInd[k]] = -1;
    }

    // Check if memory needs to be reallocated if the non-zero entries exceed the initial size
    if (matrixc->nzmax + matrixb->cols >= memIncrease)
    {
      modifyalloc(matrixc, matrixc->nzmax + memIncrease);  // Reallocate more memory for matrixc
      memIncrease = matrixc->nzmax + memIncrease;
    }

    // Update the row pointers for the sub-matrix
    matrixc->rowPtr[i - startIndex] = matrixc->nzmax - temp;
    temp = matrixc->nzmax;  // Update temp to reflect the new number of non-zero elements
  }

  // Free the work array after processing is complete
  free(workArray);
  workArray = NULL;

  // Reduce the memory size to match the actual number of non-zero elements in matrixc
  modifyalloc(matrixc, matrixc->nzmax);
}


// Function for symmetric sparse matrix multiplication between matrixa and matrixb
// Similar to the non-symmetric case, but it only considers the upper triangular part of the resulting matrix
void sparsework_sym(const struct sparsemat* const matrixa, const struct sparsemat* const matrixb, struct sparsemat* const matrixc, const int startIndex, const int endIndex,
                    int memIncrease)
{
  int i, j, k, col_num_a, col_num_b;  // Loop counters and column numbers for matrixa and matrixb
  double value;  // Temporary variable for storing values during computation
  int temp = 0;  // Temporary variable to track the number of non-zero elements
  int* workArray = NULL;  // Work array to store intermediate results
  
  // Set the size of matrixc based on the row range and the number of columns in matrixb
  matrixc->rows = endIndex - startIndex + 1;
  matrixc->cols = matrixb->cols;
  matrixc->nzmax = 0;

  // Allocate memory for the row pointers, column indices, and values of the resulting sparse matrix
  matrixc->rowPtr = (int*)calloc(matrixc->rows, sizeof(int));
  matrixc->colInd = (int*)calloc(memIncrease, sizeof(int));
  matrixc->values = (double*)calloc(memIncrease, sizeof(double));

  // Allocate memory for the work array
  workArray = (int*)calloc((size_t)matrixb->cols, sizeof(int));
  
  // Check for memory allocation failures and handle them
  if (workArray == NULL || matrixc->rowPtr == NULL || matrixc->colInd == NULL || matrixc->values == NULL) {
    fprintf(stderr, "Memory allocation failed in sparsework_sym\n");
    free(workArray);
    free(matrixc->rowPtr);
    free(matrixc->colInd);
    free(matrixc->values);
    return;
  }

  // Initialize the work array with -1
  memset(workArray, -1, matrixb->cols * sizeof(int));

  // Main loop to process rows from startIndex to endIndex in matrixa
  for (i = startIndex; i <= endIndex; ++i)
  {
    // Loop through the non-zero entries of row i in matrixa
    for (j = matrixa->rowPtr[i]; j < matrixa->rowPtr[i + 1]; ++j)
    {
      value = matrixa->values[j];  // Get the value of the current element in matrixa
      col_num_a = matrixa->colInd[j];  // Get the column index for the current element in matrixa

      // Loop through the non-zero entries in the corresponding row of matrixb
      for (k = matrixb->rowPtr[col_num_a]; k < matrixb->rowPtr[col_num_a + 1]; ++k)
      {
        col_num_b = matrixb->colInd[k];  // Get the column index for the current element in matrixb

        // Only consider entries in the upper triangular part (i <= col_num_b)
        if (i <= col_num_b)
        {
          if (workArray[col_num_b] != -1)
          {
            matrixc->values[workArray[col_num_b]] += value * matrixb->values[k];
          }
          else
          {
            matrixc->colInd[matrixc->nzmax] = col_num_b;
            workArray[col_num_b] = matrixc->nzmax;
            matrixc->values[matrixc->nzmax] = value * matrixb->values[k];
            matrixc->nzmax += 1;
          }
        }
      }
    }

    // Reset the work array for the next row
    for (k = temp; k < matrixc->nzmax; ++k)
    {
      workArray[matrixc->colInd[k]] = -1;
    }

    // Check if memory needs to be reallocated
    if (matrixc->nzmax + matrixb->cols >= memIncrease)
    {
      modifyalloc(matrixc, matrixc->nzmax + memIncrease);  // Reallocate more memory for matrixc
      memIncrease = matrixc->nzmax + memIncrease;
    }

    // Update the row pointers for the sub-matrix
    matrixc->rowPtr[i - startIndex] = matrixc->nzmax - temp;
    temp = matrixc->nzmax;  // Update temp for the next iteration
  }

  // Free the work array after processing is complete
  free(workArray);
  workArray = NULL;

  // Reduce the memory size to match the actual number of non-zero elements in matrixc
  modifyalloc(matrixc, matrixc->nzmax);
}
