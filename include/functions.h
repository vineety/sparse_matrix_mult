#ifndef FUNCTIONS_INCLUDED
#define FUNCTIONS_INCLUDED

// Conditional definition for exporting or importing functions in a shared library (DLL) on Windows platforms.
#ifdef _WIN32
    #ifdef SPARSE_LIB_EXPORTS
        // If SPARSE_LIB_EXPORTS is defined, export the functions (for building the DLL).
        #define SPARSE_LIB_API __declspec(dllexport)
    #else
        // If SPARSE_LIB_EXPORTS is not defined, import the functions (for using the DLL).
        #define SPARSE_LIB_API __declspec(dllimport)
    #endif
#else
    // On non-Windows platforms (Linux, macOS), no special directives are needed.
    #define SPARSE_LIB_API
#endif

// Macro defining the maximum chunk size for certain operations (can be used for parallelization or buffer sizes).
#define MAX_CHUNK_SIZE 16

// OpenMP-specific includes and definitions for parallel processing.
#ifdef USE_OPENMP
    #include <omp.h>  // Include OpenMP headers when parallelization is enabled.
    // Macro to get the number of threads available in the OpenMP environment.
    #define OMP_NUM_THREADS omp_get_max_threads()
#else
    // Fallback definition for the number of threads when OpenMP is not enabled (single-threaded).
    #define OMP_NUM_THREADS 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations of structures for sparse and dense matrix representations.
// These are defined elsewhere in the project but declared here for use in function signatures.
struct sparsemat;  // Sparse matrix structure (CSR format).
struct iarray;     // Integer array structure, used for storing row distributions.
struct darray;     // Dense array structure, used for storing dense matrices.

// Function declarations, exposed via the API (SPARSE_LIB_API).
// Functions for destroying matrices and arrays to free memory.
SPARSE_LIB_API void destroy_sparsemat(struct sparsemat* matrix);
SPARSE_LIB_API void destroy_iarray(struct iarray* matrix);
SPARSE_LIB_API void destroy_darray(struct darray* matrix);

// Function to reallocate memory for a sparse matrix based on new memory size.
SPARSE_LIB_API void modifyalloc(struct sparsemat* matrix, int imemSize);

// Function to create a sparse matrix with specified rows, columns, and non-zero elements (nzmax).
SPARSE_LIB_API struct sparsemat* create_sparsemat(int rows, int cols, int nzmax);

// Function to create a dense array (matrix) with specified rows and columns.
SPARSE_LIB_API struct darray* create_darray(int rows, int cols);

// Function to distribute rows of a matrix across processors for parallel computation.
SPARSE_LIB_API void limits(int tcov_rows, int numprocs, struct iarray* result);

// Function to perform non-symmetric sparse matrix multiplication for a sub-matrix.
// Result is stored in matrixc, corresponding to the row range [startIndex, endIndex], with a memory increase factor.
SPARSE_LIB_API void sparsework_nosym(const struct sparsemat* matrixa, const struct sparsemat* matrixb, struct sparsemat* matrixc, int startIndex, int endIndex, int memIncrease);

// Function to perform symmetric sparse matrix multiplication for a sub-matrix.
// Similar to sparsework_nosym, but only processes the upper triangular part of the result matrix.
SPARSE_LIB_API void sparsework_sym(const struct sparsemat* matrixa, const struct sparsemat* matrixb, struct sparsemat* matrixc, int startIndex, int endIndex, int memIncrease);

// Function to perform dense matrix multiplication and store the result in a dense matrix.
// Assumes that matrixa and matrixb are sparse matrices, and the result matrixc is dense and symmetric.
SPARSE_LIB_API void dense_sym(const struct sparsemat* matrixa, const struct sparsemat* matrixb, struct darray* matrixc);

// Function for dense matrix multiplication where the result is not symmetric.
SPARSE_LIB_API void dense_nosym(const struct sparsemat* matrixa, const struct sparsemat* matrixb, struct darray* matrixc);

// Function for sparse matrix multiplication, storing the result in a sparse matrix.
// The multiplication is non-symmetric, and imemSize specifies the memory allocation for intermediate computation.
SPARSE_LIB_API void sparse_nosym(const struct sparsemat* const matrixA, const struct sparsemat* const matrixB, struct sparsemat* const matrixC, int imemSize);

// Function for symmetric sparse matrix multiplication.
// Similar to sparse_nosym, but only the upper triangular part of the result is stored.
SPARSE_LIB_API void sparse_sym(const struct sparsemat* const matrixA, const struct sparsemat* const matrixB, struct sparsemat* const matrixC, int imemSize);

// Function to compute the triple product of sparse matrices (H * Q * H') and store the result in a dense matrix.
// The result matrix can be computed as either the full matrix or just the upper triangular part, based on the parameter compute_full_matrix.
SPARSE_LIB_API void triple_product(struct sparsemat* H, struct sparsemat* Q, struct darray* C, int compute_full_matrix);

#ifdef __cplusplus
}
#endif

#endif // FUNCTIONS_INCLUDED

