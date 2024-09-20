#include <stdio.h>       // Standard I/O functions
#include <stdlib.h>      // Standard library for memory allocation and management
#include <string.h>      // String manipulation functions
#include <math.h>        // Math library for functions like ceil
#ifdef _OPENMP         // If OpenMP is enabled, include OpenMP header
#include <omp.h>
#endif
#include "matrix_def.h"   // Custom header file for matrix structure definitions
#include "functions.h"    // Custom header file containing function declarations

/**
 * @brief Initializes a sparse matrix structure with all values set to zero.
 *
 * This function sets the non-zero maximum (`nzmax`) to 0 and assigns the number of rows and columns to the provided values.
 * It allocates memory for the row pointer (`rowPtr`), while column indices (`colInd`) and values are not allocated in this function.
 *
 * @param matrixc Pointer to the sparsemat structure.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 */
static void sparseMatrix0(struct sparsemat* const matrixc, const int rows, const int cols)
{
  matrixc->nzmax = 0; // Initialize number of non-zero elements to zero
  matrixc->rows = rows; // Set number of rows
  matrixc->cols = cols; // Set number of columns
  matrixc->rowPtr = (int*)calloc(((size_t)matrixc->rows + 1), sizeof(int)); // Allocate memory for row pointer array
}

/**
 * @brief Performs sparse matrix multiplication assuming symmetric result.
 *
 * This function computes the product of two sparse matrices `matrixA` and `matrixB` and stores the result in `matrixC`.
 * It assumes that the result is symmetric, so it only computes the upper triangular part of the result.
 * The function uses OpenMP for parallel processing if enabled.
 *
 * @param matrixA Pointer to the first sparse matrix.
 * @param matrixB Pointer to the second sparse matrix.
 * @param matrixC Pointer to the resulting sparse matrix.
 * @param imemSize Intermediate memory size hint for the operation.
 */
void sparse_sym(const struct sparsemat* const matrixA, const struct sparsemat* const matrixB, struct sparsemat* const matrixC, int imemSize)
{
  int i, j, scratch, counter, threads, subDivision;
  struct iarray rowDistribute = { 0 }; // Struct to store row distribution for parallelism
  struct sparsemat constant = { 0, 0, 0, NULL, NULL, NULL }; // A constant placeholder matrix
  struct sparsemat* dimensions = NULL; // Array of sparse matrices for intermediate results
  
  // If either matrixA or matrixB is empty (i.e., no non-zero elements), initialize matrixC with zero values
  if (matrixA->nzmax == 0 || matrixB->nzmax == 0)
  {
    sparseMatrix0(matrixC, matrixA->rows, matrixB->cols); // Initialize matrixC as an empty matrix
    return;
  }

#ifdef _OPENMP
  threads = omp_get_max_threads(); // Get the number of available threads for parallelism
#else
  threads = 1; // If OpenMP is not enabled, set to a single thread
#endif

  subDivision = threads; // Set the number of subdivisions equal to the number of threads
  limits(matrixA->rows, subDivision, &rowDistribute); // Divide rows among threads
  subDivision = rowDistribute.rows; // Update subDivision with the number of rows assigned to each thread
  
  matrixC->rows = matrixA->rows; // Set the number of rows in matrixC
  matrixC->cols = matrixB->cols; // Set the number of columns in matrixC
  
  // Determine intermediate memory size (imemSize) based on matrix dimensions and thread count
  if (imemSize > 10 || imemSize < 0)
  {
    imemSize = (int)ceil((double)(matrixA->rows * (size_t)matrixB->cols * 0.1) / threads); // Default to 10% of the matrix size
  }
  else
  {
    imemSize = (int)ceil((double)(matrixA->rows * (size_t)matrixB->cols * ((double)imemSize / 100.0)) / threads); // Use user-defined percentage
  }
  if (imemSize < matrixB->cols) // Ensure that imemSize is at least as large as the number of columns
  {
    imemSize = matrixB->cols;
  }
  
  dimensions = (struct sparsemat*)calloc(subDivision, sizeof(struct sparsemat)); // Allocate intermediate matrices for each subdivision
  if (dimensions == NULL)
  {
    fprintf(stderr, "Memory allocation failed for dimensions\n");
    return;
  }
  
  // Parallel computation block using OpenMP
#ifdef _OPENMP
#pragma omp parallel firstprivate(imemSize) private(i)
#endif
{
#ifdef USE_OPENMP
#pragma omp for schedule(dynamic)
#endif
  for (i = 0; i < subDivision; ++i) // Loop over each thread's subdivision of rows
  {
    dimensions[i] = constant; // Initialize local matrix to zero values
    
    // Perform the actual multiplication for this subdivision
    sparsework_sym(matrixA, matrixB, &dimensions[i], rowDistribute.array[i], rowDistribute.array[i + rowDistribute.rows], imemSize);
    
    // Atomically update the total number of non-zero elements in matrixC
#ifdef _OPENMP
#pragma omp atomic
#endif
    matrixC->nzmax += dimensions[i].nzmax;
  }
}

// Allocate memory for matrixC based on the total number of non-zero elements (nzmax)
matrixC->rowPtr = (int*)calloc(((size_t)matrixC->rows + 1), sizeof(int));
matrixC->colInd = (int*)calloc((size_t)matrixC->nzmax, sizeof(int));
matrixC->values = (double*)calloc((size_t)matrixC->nzmax, sizeof(double));

// Memory allocation failure check
if (matrixC->rowPtr == NULL || matrixC->colInd == NULL || matrixC->values == NULL)
{
  fprintf(stderr, "Memory allocation failed for matrixC\n");
  free(dimensions);
  return;
}

// Initialize variables for final population of matrixC
scratch = 0;
counter = 1;

for (i = 0; i < subDivision; ++i)
{
  // Update rowPtr for each sub-division
  for (int j = 0; j < rowDistribute.array[i + rowDistribute.rows] - rowDistribute.array[i] + 1; ++j)
  {
    matrixC->rowPtr[counter] = matrixC->rowPtr[counter - 1] + dimensions[i].rowPtr[j];
    counter += 1;
  }
  
  // Populate colInd and values arrays from the sub-division results
  if (dimensions[i].colInd != NULL && dimensions[i].values != NULL)
  {
    for (j = 0; j < dimensions[i].nzmax; ++j)
    {
      matrixC->colInd[scratch] = dimensions[i].colInd[j]; // Copy column indices
      matrixC->values[scratch] = dimensions[i].values[j]; // Copy values
      scratch += 1;
    }
  }
  // Clean up memory for the thread-local sparse matrix
  destroy_sparsemat(&dimensions[i]);
}

destroy_iarray(&rowDistribute); // Free the row distribution array
free(dimensions);               // Free the intermediate matrices
dimensions = NULL;              // Set dimensions pointer to NULL for safety
}



/**
 * @brief Performs sparse matrix multiplication for non-symmetric matrices.
 *
 * This function computes the product of two sparse matrices `matrixA` and `matrixB` and stores the result in `matrixC`.
 * It does not assume any symmetry in the result matrix, unlike `sparse_sym`.
 * The function also uses OpenMP for parallel processing if enabled.
 *
 * @param matrixA Pointer to the first sparse matrix.
 * @param matrixB Pointer to the second sparse matrix.
 * @param matrixC Pointer to the resulting sparse matrix.
 * @param imemSize Intermediate memory size hint for the operation.
 */

void sparse_nosym(const struct sparsemat* const matrixA, const struct sparsemat* const matrixB, struct sparsemat* const matrixC, int imemSize)
{
    // Declare variables for iteration, scratch space, counters, number of threads, and subdivision for parallel execution
    int i, j, scratch, counter, threads, subDivision;
    struct iarray rowDistribute = { 0 };  // Structure to hold row distribution for parallel processing
    struct sparsemat constant = { 0, 0, 0, NULL, NULL, NULL };  // Initialize a constant sparse matrix
    struct sparsemat* dimensions = NULL;  // Pointer to hold sparse matrix partitions for each thread
    
    // Check if either matrixA or matrixB has no non-zero elements, and directly initialize matrixC to zero if true
    if (matrixA->nzmax == 0 || matrixB->nzmax == 0)
    {
        sparseMatrix0(matrixC, matrixA->rows, matrixB->cols);  // Set matrixC as a zero matrix
        return;
    }
    
    // Determine the number of threads to use, either OpenMP parallel threads or 1 if OpenMP is not enabled
#ifdef _OPENMP
    threads = omp_get_max_threads();  // Get the number of available threads
#else
    threads = 1;  // Single thread if OpenMP is not enabled
#endif

    // Set subdivision of rows for parallel processing based on the number of threads
    subDivision = threads;
    limits(matrixA->rows, subDivision, &rowDistribute);  // Calculate row limits for each thread
    subDivision = rowDistribute.rows;

    // Initialize matrixC with the same dimensions as the product of matrixA and matrixB
    matrixC->rows = matrixA->rows;
    matrixC->cols = matrixB->cols;

    // Adjust memory size for intermediate computation based on the input or default values
    if (imemSize > 10 || imemSize < 0)
    {
        imemSize = (int)ceil((double)(matrixA->rows * (size_t)matrixB->cols * 0.1) / threads);  // Default to 10% of matrix size
    }
    else
    {
        imemSize = (int)ceil((double)(matrixA->rows * (size_t)matrixB->cols * ((double)imemSize / 100.0)) / threads);  // Use user-defined percentage
    }

    // Ensure imemSize is at least as large as matrixB's column size
    if (imemSize < matrixB->cols)
    {
        imemSize = matrixB->cols;
    }

    // Allocate memory for the sparse matrix partitions for each thread
    dimensions = (struct sparsemat*)calloc(subDivision, sizeof(struct sparsemat));
    if (dimensions == NULL)
    {
        fprintf(stderr, "Memory allocation failed for dimensions\n");
        return;
    }

    // Parallel block to perform the actual sparse matrix multiplication using OpenMP
#ifdef _OPENMP
#pragma omp parallel firstprivate(imemSize) private(i)  // Parallel region with imemSize as firstprivate and i as private
#endif
    {
#ifdef _OPENMP
#pragma omp for schedule(dynamic)  // Distribute iterations dynamically across threads
#endif
        for (i = 0; i < subDivision; ++i)
        {
            // Initialize the sparse matrix partition for each thread
            dimensions[i] = constant;

            // Perform the actual multiplication for the assigned rows
            sparsework_nosym(matrixA, matrixB, &dimensions[i], rowDistribute.array[i], rowDistribute.array[i + rowDistribute.rows], imemSize);

            // Atomically update the total number of non-zero elements in matrixC
#ifdef _OPENMP
#pragma omp atomic
#endif
            matrixC->nzmax += dimensions[i].nzmax;
        }
    }

    // Allocate memory for matrixC's row pointer, column index, and value arrays based on total nzmax
    matrixC->rowPtr = (int*)calloc(((size_t)matrixC->rows + 1), sizeof(int));
    matrixC->colInd = (int*)calloc((size_t)matrixC->nzmax, sizeof(int));
    matrixC->values = (double*)calloc((size_t)matrixC->nzmax, sizeof(double));

    // Check if memory allocation failed for matrixC
    if (matrixC->rowPtr == NULL || matrixC->colInd == NULL || matrixC->values == NULL)
    {
        fprintf(stderr, "Memory allocation failed for matrixC\n");
        free(dimensions);
        return;
    }

    // Initialize variables for final population of matrixC's rowPtr, colInd, and values
    scratch = 0;
    counter = 1;

    // Loop through each sub-division and populate matrixC
    for (i = 0; i < subDivision; ++i)
    {
        // Update matrixC's rowPtr for each sub-division
        for (j = 0; j < rowDistribute.array[i + rowDistribute.rows] - rowDistribute.array[i] + 1; ++j)
        {
            matrixC->rowPtr[counter] = matrixC->rowPtr[counter - 1] + dimensions[i].rowPtr[j];
            counter += 1;
        }

        // Update matrixC's colInd and values for each sub-division
        if (dimensions[i].colInd != NULL && dimensions[i].values != NULL)
        {
            for (j = 0; j < dimensions[i].nzmax; ++j)
            {
                matrixC->colInd[scratch] = dimensions[i].colInd[j];
                matrixC->values[scratch] = dimensions[i].values[j];
                scratch += 1;
            }
        }

        // Clean up each thread's sparse matrix partition
        destroy_sparsemat(&dimensions[i]);
    }

    // Clean up the row distribution array
    destroy_iarray(&rowDistribute);

    // Free allocated memory for dimensions
    free(dimensions);
    dimensions = NULL;
}


