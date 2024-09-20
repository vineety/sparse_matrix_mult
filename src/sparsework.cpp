#include <string.h>
#include <math.h>
#ifdef _OPENMP         // If OpenMP is enabled, include OpenMP header
#include <omp.h>
#endif
#include "matrix_def.h"  // Include matrix structure definitions
#include "functions.h"  // Include custom matrix functions

// Function for non-symmetric sparse matrix multiplication between matrixa and matrixb
// Result is stored in matrixc, which is a sub-matrix corresponding to the row range [startIndex, endIndex]
// memIncrease specifies the initial memory allocation size for matrixc
void sparsework_nosym(const struct sparsemat* const matrixa, const struct sparsemat* const matrixb,
                      struct sparsemat* const matrixc, const int startIndex, const int endIndex,
                      int memIncrease)
{
    int local_rows = endIndex - startIndex + 1; // Number of rows to process

    // Initialize dimensions and non-zero count for the result matrix
    matrixc->rows = local_rows;
    matrixc->cols = matrixb->cols;
    matrixc->nzmax = 0; // Initialize the number of non-zero elements to zero

    // Estimate the initial number of non-zero elements (nzmax) for matrixc
    // Here, we use memIncrease as the initial estimate, which can be adjusted as needed
    size_t estimated_nzmax = (size_t)memIncrease;

    // Calculate the total memory required for matrixc components
    size_t total_memory = (size_t)(local_rows + 1) * sizeof(int) + // For rowPtr
                          estimated_nzmax * (sizeof(int) + sizeof(double)); // For colInd and values

    // Allocate a contiguous block of memory for matrixc
    char* memory_pool = (char*)malloc(total_memory);
    if (memory_pool == NULL) {
        fprintf(stderr, "Memory allocation failed in sparsework_nosym\n");
        return;
    }

    // Set pointers within the memory pool for matrixc components
    matrixc->rowPtr = (int*)memory_pool;
    matrixc->colInd = (int*)(memory_pool + (size_t)(local_rows + 1) * sizeof(int));
    matrixc->values = (double*)(memory_pool + (size_t)(local_rows + 1) * sizeof(int) +
                                estimated_nzmax * sizeof(int));

    // Allocate and initialize the work array to keep track of column indices
    int* workArray = (int*)calloc((size_t)matrixb->cols, sizeof(int));
    if (workArray == NULL) {
        fprintf(stderr, "Memory allocation failed for workArray in sparsework_nosym\n");
        free(memory_pool);
        return;
    }

    // Initialize the starting index for the current row in matrixc
    size_t temp = 0;

    // Iterate over the rows of matrixa from startIndex to endIndex
    for (int i = startIndex; i <= endIndex; ++i)
    {
        // Record the starting point of non-zero elements for the current row
        size_t row_start = matrixc->nzmax;

        // Iterate over the non-zero elements in row i of matrixa
        for (int j = matrixa->rowPtr[i]; j < matrixa->rowPtr[i + 1]; ++j)
        {
            double value = matrixa->values[j];    // Value of the non-zero element
            int col_num_a = matrixa->colInd[j];   // Column index in matrixa

            // Iterate over the non-zero elements in the corresponding row of matrixb
            for (int k = matrixb->rowPtr[col_num_a]; k < matrixb->rowPtr[col_num_a + 1]; ++k)
            {
                int col_b = matrixb->colInd[k]; // Column index in matrixb

                // Check if this column index has been encountered in the current row
                if (workArray[col_b] >= (int)row_start)
                {
                    // Update the existing value in matrixc
                    matrixc->values[workArray[col_b]] += value * matrixb->values[k];
                }
                else
                {
                    // Check if we need to reallocate memory for matrixc
                    if ((size_t)matrixc->nzmax >= (size_t)estimated_nzmax)
                    {
                        // Increase estimated_nzmax to reduce the frequency of reallocations
                        estimated_nzmax *= 2;
                        size_t new_size = (size_t)(local_rows + 1) * sizeof(int) +
                                          estimated_nzmax * (sizeof(int) + sizeof(double));

                        // Reallocate the memory pool to accommodate more non-zero elements
                        char* new_memory = (char*)realloc(memory_pool, new_size);
                        if (new_memory == NULL) {
                            fprintf(stderr, "Memory reallocation failed in sparsework_nosym\n");
                            free(memory_pool);
                            free(workArray);
                            return;
                        }

                        // Update the memory pool and pointers after reallocation
                        memory_pool = new_memory;
                        matrixc->rowPtr = (int*)memory_pool;
                        matrixc->colInd = (int*)(memory_pool + (size_t)(local_rows + 1) * sizeof(int));
                        matrixc->values = (double*)(memory_pool + (size_t)(local_rows + 1) * sizeof(int) +
                                                    estimated_nzmax * sizeof(int));
                    }

                    // Insert the new non-zero element into matrixc
                    size_t index = matrixc->nzmax;
                    matrixc->colInd[index] = col_b;
                    matrixc->values[index] = value * matrixb->values[k];
                    workArray[col_b] = (int)index; // Record the position in workArray
                    matrixc->nzmax++;
                }
            }
        }

        // Record the number of non-zero elements added in this row
        matrixc->rowPtr[i - startIndex] = (int)(matrixc->nzmax - temp);
        temp = matrixc->nzmax;

        // Reset the workArray entries used in this row
        if ((matrixc->nzmax - row_start) > 32) {
            // If many columns were updated, reset the entire workArray
            memset(workArray, -1, matrixb->cols * sizeof(int));
        } else {
            // Otherwise, reset only the entries that were used
            for (size_t k = row_start; k < (size_t)matrixc->nzmax; ++k) {
                workArray[matrixc->colInd[k]] = -1;
            }
        }
    }

    // Free the workArray as it's no longer needed
    free(workArray);

    // Adjust the final size of the memory pool to fit the actual data
    size_t final_size = (size_t)(local_rows + 1) * sizeof(int) +
                        matrixc->nzmax * (sizeof(int) + sizeof(double));
    char* final_memory = (char*)realloc(memory_pool, final_size);
    if (final_memory == NULL) {
        fprintf(stderr, "Final memory reallocation failed in sparsework_nosym\n");
        // If realloc fails, we proceed with the existing memory pool
    } else {
        // Update the memory pool and pointers to the shrunk memory
        memory_pool = final_memory;
        matrixc->rowPtr = (int*)memory_pool;
        matrixc->colInd = (int*)(memory_pool + (size_t)(local_rows + 1) * sizeof(int));
        matrixc->values = (double*)(memory_pool + (size_t)(local_rows + 1) * sizeof(int) +
                                    matrixc->nzmax * sizeof(int));
    }
}



// Function for symmetric sparse matrix multiplication between matrixa and matrixb
// Similar to the non-symmetric case, but it only considers the upper triangular part of the resulting matrix

void sparsework_sym(const struct sparsemat* const matrixa, const struct sparsemat* const matrixb,
                    struct sparsemat* const matrixc, const int startIndex, const int endIndex,
                    int memIncrease)
{
    // Calculate the number of local rows to process
    int local_rows = endIndex - startIndex + 1;

    // Initialize dimensions and non-zero count for the result matrix
    matrixc->rows = local_rows;
    matrixc->cols = matrixb->cols;
    matrixc->nzmax = 0; // Initialize the number of non-zero elements to zero

    // Estimate the initial number of non-zero elements (nzmax) for matrixc
    // Add a 20% buffer to accommodate potential additional non-zero elements
    size_t estimated_nzmax = (size_t)((double)matrixa->nzmax / matrixa->rows * local_rows * 1.2);

    // Calculate the total memory required for matrixc components
    size_t total_memory = (size_t)(local_rows + 1) * sizeof(int) + // For rowPtr
                          estimated_nzmax * (sizeof(int) + sizeof(double)); // For colInd and values

    // Allocate a contiguous block of memory for matrixc
    char* memory_pool = (char*)malloc(total_memory);
    if (memory_pool == NULL) {
        fprintf(stderr, "Memory allocation failed in sparsework_sym\n");
        return;
    }

    // Set pointers within the memory pool for matrixc components
    matrixc->rowPtr = (int*)memory_pool;
    matrixc->colInd = (int*)(memory_pool + (size_t)(local_rows + 1) * sizeof(int));
    matrixc->values = (double*)(memory_pool + (size_t)(local_rows + 1) * sizeof(int) +
                                estimated_nzmax * sizeof(int));

    // Allocate and initialize the work array to keep track of column indices
    int* workArray = (int*)calloc((size_t)matrixb->cols, sizeof(int));
    if (workArray == NULL) {
        fprintf(stderr, "Memory allocation failed for workArray in sparsework_sym\n");
        free(memory_pool);
        return;
    }

    // Initialize the starting index for the current row in matrixc
    size_t row_start = 0;

    // Iterate over the rows of matrixa from startIndex to endIndex
    for (int i = startIndex; i <= endIndex; ++i)
    {
        size_t local_nzmax = 0; // Number of non-zero elements in the current row

        // Iterate over the non-zero elements in row i of matrixa
        for (int j = matrixa->rowPtr[i]; j < matrixa->rowPtr[i + 1]; ++j)
        {
            double value = matrixa->values[j];    // Value of the non-zero element
            int col_num_a = matrixa->colInd[j];   // Column index in matrixa

            // Iterate over the non-zero elements in the corresponding row of matrixb
            for (int k = matrixb->rowPtr[col_num_a]; k < matrixb->rowPtr[col_num_a + 1]; ++k)
            {
                int col_num_b = matrixb->colInd[k]; // Column index in matrixb

                // Since the matrix is symmetric, process only the upper triangle (i <= col_num_b)
                if (i <= col_num_b)
                {
                    // Check if this column index has been encountered in the current row
                    if (workArray[col_num_b] >= (int)row_start)
                    {
                        // Update the existing value in matrixc
                        matrixc->values[workArray[col_num_b]] += value * matrixb->values[k];
                    }
                    else
                    {
                        // Check if we need to reallocate memory for matrixc
                        if ((size_t)matrixc->nzmax >= estimated_nzmax)
                        {
                            // Double the estimated nzmax to reduce the frequency of reallocations
                            estimated_nzmax *= 2;
                            size_t new_size = (size_t)(local_rows + 1) * sizeof(int) +
                                              estimated_nzmax * (sizeof(int) + sizeof(double));

                            // Reallocate the memory pool to accommodate more non-zero elements
                            char* new_memory = (char*)realloc(memory_pool, new_size);
                            if (new_memory == NULL) {
                                fprintf(stderr, "Memory reallocation failed in sparsework_sym\n");
                                free(memory_pool);
                                free(workArray);
                                return;
                            }

                            // Update the memory pool and pointers after reallocation
                            memory_pool = new_memory;
                            matrixc->rowPtr = (int*)memory_pool;
                            matrixc->colInd = (int*)(memory_pool + (size_t)(local_rows + 1) * sizeof(int));
                            matrixc->values = (double*)(memory_pool + (size_t)(local_rows + 1) * sizeof(int) +
                                                        estimated_nzmax * sizeof(int));
                        }

                        // Insert the new non-zero element into matrixc
                        size_t index = matrixc->nzmax;
                        matrixc->colInd[index] = col_num_b;
                        matrixc->values[index] = value * matrixb->values[k];
                        workArray[col_num_b] = (int)index; // Record the position in workArray
                        matrixc->nzmax++;
                        local_nzmax++;
                    }
                }
            }
        }

        // Record the number of non-zero elements in the current row
        matrixc->rowPtr[i - startIndex] = (int)local_nzmax;

        // Reset the workArray for the next row
        if (local_nzmax > 32) {
            // If many columns were updated, reset the entire workArray
            memset(workArray, 0, matrixb->cols * sizeof(int));
        } else {
            // Otherwise, reset only the entries that were used
            for (size_t k = 0; k < local_nzmax; ++k) {
                workArray[matrixc->colInd[row_start + k]] = 0;
            }
        }

        // Update the starting index for the next row
        row_start += local_nzmax;
    }

    // Free the workArray as it's no longer needed
    free(workArray);

    // Adjust the final size of the memory pool to fit the actual data
    size_t final_size = (size_t)(local_rows + 1) * sizeof(int) +
                        matrixc->nzmax * (sizeof(int) + sizeof(double));
    char* final_memory = (char*)realloc(memory_pool, final_size);
    if (final_memory == NULL) {
        fprintf(stderr, "Final memory reallocation failed in sparsework_sym\n");
        // If realloc fails, we proceed with the existing memory pool
    } else {
        // Update the memory pool and pointers to the shrunk memory
        memory_pool = final_memory;
        matrixc->rowPtr = (int*)memory_pool;
        matrixc->colInd = (int*)(memory_pool + (size_t)(local_rows + 1) * sizeof(int));
        matrixc->values = (double*)(memory_pool + (size_t)(local_rows + 1) * sizeof(int) +
                                    matrixc->nzmax * sizeof(int));
    }
}
