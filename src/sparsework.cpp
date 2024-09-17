#include <string.h>
#include <math.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include "matrix_def.h"
#include "functions.h"



void sparsework_nosym(const struct sparsemat* const matrixa, const struct sparsemat* const matrixb, struct sparsemat* const matrixc, const int startIndex, const int endIndex,
                      int memIncrease)
{
  /* Loop counters and scratch variables */
  int i, j, k, col_num_a;
  double value;
  int temp = 0;
  int* workArray = NULL;
  
  matrixc->rows = endIndex - startIndex + 1; // size of smaller sub-matrix
  matrixc->cols = matrixb->cols; // columns in smaller sub-matrix
  matrixc->nzmax = 0;
  
  // Memory would be allocated based on initial size specified by the user
  matrixc->rowPtr = (int*)calloc((size_t)matrixc->rows, sizeof(int));
  matrixc->colInd = (int*)calloc((size_t)memIncrease, sizeof(int));
  matrixc->values = (double*)calloc((size_t)memIncrease, sizeof(double));
  
  // Workarray for collecting results in a temporary array
  workArray = (int*)calloc((size_t)matrixb->cols, sizeof(int));
  if (workArray == NULL || matrixc->rowPtr == NULL || matrixc->colInd == NULL || matrixc->values == NULL) {
    fprintf(stderr, "Memory allocation failed in sparsework_nosym\n");
    free(workArray);
    free(matrixc->rowPtr);
    free(matrixc->colInd);
    free(matrixc->values);
    return;
  }
  
  memset(workArray, -1, (size_t)matrixb->cols * sizeof(int));
  
  /* Get column indices and entries of the output matrix for the rows
   from startIndex to endIndex */
  for (i = startIndex; i <= endIndex; ++i)
  {
    for (j = matrixa->rowPtr[i]; j < matrixa->rowPtr[i + 1]; ++j)
    {
      value = matrixa->values[j];
      col_num_a = matrixa->colInd[j];
      for (k = matrixb->rowPtr[col_num_a]; k < matrixb->rowPtr[col_num_a + 1]; ++k)
      {
        /* This would give the column number of the entry in the B matrix */
        if (workArray[matrixb->colInd[k]] != -1)
        {
          matrixc->values[workArray[matrixb->colInd[k]]] += value * matrixb->values[k];
        }
        else
        {
          matrixc->colInd[matrixc->nzmax] = matrixb->colInd[k];
          workArray[matrixb->colInd[k]] = matrixc->nzmax;
          matrixc->values[matrixc->nzmax] = value * matrixb->values[k];
          matrixc->nzmax += 1;
        }
      }
    }
    /* Put results sequentially. Counter computes
     how many entries in a row of the output matrix are non-zeros */
    for (k = temp; k < matrixc->nzmax; ++k)
    {
      workArray[matrixc->colInd[k]] = -1;
    }
    
    if (matrixc->nzmax + matrixb->cols >= memIncrease)
    {
      modifyalloc(matrixc, matrixc->nzmax + memIncrease);
      memIncrease = matrixc->nzmax + memIncrease;
    }
    
    // This collects results for final calculation of row pointers
    matrixc->rowPtr[i - startIndex] = matrixc->nzmax - temp;
    temp = matrixc->nzmax;
  }
  free(workArray);
  workArray = NULL;
  
  // Final reduce the size of memory based on actual nzmax
  modifyalloc(matrixc, matrixc->nzmax);
}


void sparsework_sym(const struct sparsemat* const matrixa, const struct sparsemat* const matrixb, struct sparsemat* const matrixc, const int startIndex, const int endIndex,
                    int memIncrease)
{
  /*loop counters and scratch variables*/
  int i, j, k, col_num_a, col_num_b;
  double value;
  int temp = 0;
  int* workArray = NULL;
  
  matrixc->rows = endIndex - startIndex + 1; // size of smaller sub-matrix
  matrixc->cols = matrixb->cols; // columns in smaller sub-matrix
  matrixc->nzmax = 0;
  // Memory would be allocated based on initial size specified by the user
  matrixc->rowPtr = (int*)calloc(matrixc->rows, sizeof(int));
  matrixc->colInd = (int*)calloc(memIncrease, sizeof(int));
  matrixc->values = (double*)calloc(memIncrease, sizeof(double));
  
  // Workarray for collecting results in a temporary array
  workArray = (int*)calloc((size_t)matrixb->cols, sizeof(int));
  if (workArray == NULL || matrixc->rowPtr == NULL || matrixc->colInd == NULL || matrixc->values == NULL) {
    fprintf(stderr, "Memory allocation failed in sparsework_sym\n");
    free(workArray);
    free(matrixc->rowPtr);
    free(matrixc->colInd);
    free(matrixc->values);
    return;
  }
  
  memset(workArray, -1, matrixb->cols * sizeof(int));
  
  /*Get column indices and entries of the output matrix for the rows
   from startIndex to endIndex*/
  for (i = startIndex; i <= endIndex; ++i)
  {
    for (j = matrixa->rowPtr[i]; j < matrixa->rowPtr[i + 1]; ++j)
    {
      value = matrixa->values[j];
      col_num_a = matrixa->colInd[j];
      for (k = matrixb->rowPtr[col_num_a]; k < matrixb->rowPtr[col_num_a + 1]; ++k)
      {
        /*This would give coloumn no of the entry in the B matrix*/
        col_num_b = matrixb->colInd[k];
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
    /*put results sequentially. Counter computes
     how many entries in a row of the output matrix are non-zeros*/
    for (k = temp; k < matrixc->nzmax; ++k)
    {
      workArray[matrixc->colInd[k]] = -1;
    }
    if (matrixc->nzmax + matrixb->cols >= memIncrease)
    {
      modifyalloc(matrixc, matrixc->nzmax + memIncrease);
      memIncrease = matrixc->nzmax + memIncrease;
    }
    //This collect results for final calculation of row pointers
    matrixc->rowPtr[i - startIndex] = matrixc->nzmax - temp;
    temp = matrixc->nzmax;
  }
  
  free(workArray);
  workArray = NULL;
  // if memory is more than nzmax than reduce the size of memory
  modifyalloc(matrixc, matrixc->nzmax);
}
