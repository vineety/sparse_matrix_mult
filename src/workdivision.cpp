#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>  // Include OpenMP for parallelization
#endif
#include "matrix_def.h"  // Include custom matrix definitions
#include "memory.h"  // Include custom memory management functions
#include "functions.h"  // Include additional custom functions

// Function to divide rows of a matrix among processors for parallel computation
// tcov_rows: total number of rows in the matrix
// numprocs: number of processors/threads to divide the rows across
// result: an iarray struct that will store the row limits for each processor
void limits(int tcov_rows, int numprocs, struct iarray* result)
{
    // Error checking to ensure the number of processors is valid
    if (numprocs <= 0)
    {
        // Exit the program if the number of processors is invalid
        exit(0);
    }
    
    // If the number of processors is greater than the number of rows, set numprocs to the number of rows
    if (numprocs > tcov_rows)
    {
        numprocs = tcov_rows;
    }
    
    // Initialize the result struct with the number of processors and columns (2 columns: start and end of row range)
    result->rows = numprocs;
    result->cols = 2;
    
    // Allocate memory for the array in result, which will store the start and end indices of rows for each processor
    result->array = (int*)calloc((size_t)numprocs * 2, sizeof(int));
    
    // Check if memory allocation failed and exit if true
    if (result->array == NULL)
    {
        printf("Error! Memory not allocated for start vector, Program Exiting");
        exit(0);
    }
    
    // Calculate the number of leftover rows (remainder) after dividing rows evenly across processors
    int remainder = tcov_rows % numprocs;
    
    // Set pset flag to 1 if there is a remainder, otherwise set it to 0
    int pset = (remainder > 0) ? 1 : 0;
    
    // Calculate the number of rows per processor (division) without remainder
    int division = (tcov_rows - remainder) / numprocs;
    
    // Initialize index variable to track the starting row index for each processor
    int index = 0;
    
    // Loop over each processor to calculate the row range (start and end indices) for each processor
    for (int i = 0; i < numprocs; ++i)
    {
        // Case when there is no remainder
        if (pset == 0)
        {
            result->array[i] = index;  // Set the start index for the current processor
            result->array[i + result->rows] = result->array[i] + division - 1;  // Set the end index
            index = result->array[i + result->rows] + 1;  // Update index to the next starting row
        }
        // Case when there is a remainder (rows need to be unevenly distributed)
        else if (pset == 1)
        {
            // If remainder is still greater than 0, allocate one extra row to this processor
            if (remainder > 0)
            {
                result->array[i] = index;  // Set the start index
                result->array[i + result->rows] = index + (division - 1) + (remainder - (remainder - 1));  // Set the end index with an extra row
                remainder = remainder - 1;  // Decrease the remainder
                index = result->array[i + result->rows] + 1;  // Update index to the next starting row
            }
            // When remainder becomes zero, assign rows normally
            else if (remainder == 0)
            {
                result->array[i] = index;  // Set the start index
                result->array[i + result->rows] = index + division - 1;  // Set the end index
                index = result->array[i + result->rows] + 1;  // Update index to the next starting row
            }
        }
    }
    
    // No return statement needed, as the function modifies the result struct directly
}
