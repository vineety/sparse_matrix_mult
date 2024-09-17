#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include "matrix_def.h"
#include "functions.h"

static void sparseMatrix0(struct sparsemat* const matrixc, const int rows, const int cols)
{
  matrixc->nzmax = 0;
  matrixc->rows = rows;
  matrixc->cols = cols;
  matrixc->rowPtr = (int*)calloc(((size_t)matrixc->rows + 1), sizeof(int));
}

void sparse_sym(const struct sparsemat* const matrixA, const struct sparsemat* const matrixB, struct sparsemat* const matrixC, int imemSize)
{
  int i, j, scratch, counter, threads, subDivision;
  struct iarray rowDistribute = { 0 };
  struct sparsemat constant = { 0, 0, 0, NULL, NULL, NULL };
  struct sparsemat* dimensions = NULL;
  
  if (matrixA->nzmax == 0 || matrixB->nzmax == 0)
  {
    sparseMatrix0(matrixC, matrixA->rows, matrixB->cols);
    return;
  }
#ifdef USE_OPENMP
  threads = omp_get_max_threads();
#else
  threads = 1;
#endif

  subDivision = threads;
  limits(matrixA->rows, subDivision, &rowDistribute);
  subDivision = rowDistribute.rows;
  
  matrixC->rows = matrixA->rows;
  matrixC->cols = matrixB->cols;
  
  if (imemSize > 10 || imemSize < 0)
  {
    imemSize = (int)ceil((double)(matrixA->rows * (size_t)matrixB->cols * 0.1) / threads);
  }
  else
  {
    imemSize = (int)ceil((double)(matrixA->rows * (size_t)matrixB->cols * ((double)imemSize / 100.0)) / threads);
  }
  if (imemSize < matrixB->cols)
  {
    imemSize = matrixB->cols;
  }
  
  dimensions = (struct sparsemat*)calloc(subDivision, sizeof(struct sparsemat));
  if (dimensions == NULL)
  {
    fprintf(stderr, "Memory allocation failed for dimensions\n");
    return;
  }
  
  // Parallel block for computation
#ifdef USE_OPENMP
#pragma omp parallel firstprivate(imemSize) private(i)
#endif
{
#ifdef USE_OPENMP
#pragma omp for schedule(dynamic)
#endif
  for (i = 0; i < subDivision; ++i)
  {
    dimensions[i] = constant;
    
    
    sparsework_sym(matrixA, matrixB, &dimensions[i], rowDistribute.array[i], rowDistribute.array[i + rowDistribute.rows], imemSize);
    
    
    // Update nzmax atomically
#ifdef USE_OPENMP
#pragma omp atomic
#endif
    matrixC->nzmax += dimensions[i].nzmax;
  }
}

// Allocate memory for matrixC based on the total nzmax
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

// Initialize variables for final population
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
  
  // Update colInd and values for each sub-division
  if (dimensions[i].colInd != NULL && dimensions[i].values != NULL)
  {
    for (j = 0; j < dimensions[i].nzmax; ++j)
    {
      matrixC->colInd[scratch] = dimensions[i].colInd[j];
      matrixC->values[scratch] = dimensions[i].values[j];
      scratch += 1;
    }
  }
  // Clean up the thread-local sparse matrix
  destroy_sparsemat(&dimensions[i]);
}

destroy_iarray(&rowDistribute);
free(dimensions);
dimensions = NULL;
}

void sparse_nosym(const struct sparsemat* const matrixA, const struct sparsemat* const matrixB, struct sparsemat* const matrixC, int imemSize)
{
  int i, j, scratch, counter, threads, subDivision;
  struct iarray rowDistribute = { 0 };
  struct sparsemat constant = { 0, 0, 0, NULL, NULL, NULL };
  struct sparsemat* dimensions = NULL;
  
  if (matrixA->nzmax == 0 || matrixB->nzmax == 0)
  {
    sparseMatrix0(matrixC, matrixA->rows, matrixB->cols);
    return;
  }
  
#ifdef USE_OPENMP
  threads = omp_get_max_threads();
#else
  threads = 1;
#endif
  subDivision = threads;
  limits(matrixA->rows, subDivision,&rowDistribute);
  subDivision = rowDistribute.rows;
  
  matrixC->rows = matrixA->rows;
  matrixC->cols = matrixB->cols;
  
  if (imemSize > 10 || imemSize < 0)
  {
    imemSize = (int)ceil((double)(matrixA->rows * (size_t)matrixB->cols * 0.1) / threads);
  }
  else
  {
    imemSize = (int)ceil((double)(matrixA->rows * (size_t)matrixB->cols * ((double)imemSize / 100.0)) / threads);
  }
  if (imemSize < matrixB->cols)
  {
    imemSize = matrixB->cols;
  }
  
  dimensions = (struct sparsemat*)calloc(subDivision, sizeof(struct sparsemat));
  if (dimensions == NULL)
  {
    fprintf(stderr, "Memory allocation failed for dimensions\n");
    return;
  }
  
  // Parallel block for computation
#ifdef USE_OPENMP
#pragma omp parallel firstprivate(imemSize) private(i)
#endif
{
#ifdef USE_OPENMP
#pragma omp for schedule(dynamic)
#endif
  for (i = 0; i < subDivision; ++i)
  {
    dimensions[i] = constant;
    
    
    sparsework_nosym(matrixA, matrixB, &dimensions[i], rowDistribute.array[i], rowDistribute.array[i + rowDistribute.rows], imemSize);
    
    
    // Update nzmax atomically
#ifdef USE_OPENMP
#pragma omp atomic
#endif
    matrixC->nzmax += dimensions[i].nzmax;
  }
}
// Allocate memory for matrixC based on the total nzmax
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

// Initialize variables for final population
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
  
  // Update colInd and values for each sub-division
  if (dimensions[i].colInd != NULL && dimensions[i].values != NULL)
  {
    for (j = 0; j < dimensions[i].nzmax; ++j)
    {
      matrixC->colInd[scratch] = dimensions[i].colInd[j];
      matrixC->values[scratch] = dimensions[i].values[j];
      scratch += 1;
    }
  }
  // Clean up the thread-local sparse matrix
  destroy_sparsemat(&dimensions[i]);
}

destroy_iarray(&rowDistribute);
free(dimensions);
dimensions = NULL;
}

