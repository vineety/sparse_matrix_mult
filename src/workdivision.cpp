#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include "matrix_def.h"
#include "memory.h"
#include "functions.h"



void limits(int tcov_rows, int numprocs, struct iarray* result)
{
  // Error checking
  if (numprocs <= 0)
  {
    //printf("Error! Number of Processors Cannot be Zero. Program Exiting");
    exit(0);
  }
  if (numprocs > tcov_rows)
  {
    numprocs = tcov_rows;
    //printf("Number of processors greater than number of rows\n");
    //printf("Number of processors are set to number of rows\n");
  }
  
  // Initialize the result struct
  result->rows = numprocs;
  result->cols = 2;
  result->array = (int*)calloc((size_t)numprocs * 2, sizeof(int));
  if (result->array == NULL)
  {
    printf("Error! memory not allocated for start vector, Program Exiting");
    exit(0);
  }
  
  // The rest of the function remains largely the same
  int remainder = tcov_rows % numprocs;
  int pset = (remainder > 0) ? 1 : 0;
  int division = (tcov_rows - remainder) / (numprocs);
  int index = 0;
  
  for (int i = 0; i < numprocs; ++i)
  {
    if (pset == 0)
    {
      result->array[i] = index;
      result->array[i + result->rows] = result->array[i] + division - 1;
      index = result->array[i + result->rows] + 1;
    }
    else if (pset == 1)
    {
      if (remainder > 0)
      {
        result->array[i] = index;
        result->array[i + result->rows] = index + (division - 1) + (remainder - (remainder - 1));
        remainder = remainder - 1;
        index = result->array[i + result->rows] + 1;
      }
      else if (remainder == 0)
      {
        result->array[i] = index;
        result->array[i + result->rows] = index + division - 1;
        index = result->array[i + result->rows] + 1;
      }
    }
  }
  
  // No return statement needed
}
