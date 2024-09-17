#ifndef FUNCTIONS_INCLUDED
#define FUNCTIONS_INCLUDED

#ifdef _WIN32
    #ifdef SPARSE_LIB_EXPORTS
        #define SPARSE_LIB_API __declspec(dllexport)
    #else
        #define SPARSE_LIB_API __declspec(dllimport)
    #endif
#else
    #define SPARSE_LIB_API
#endif

// Macro definition for maximum chunk size
#define MAX_CHUNK_SIZE 16

// OpenMP-specific includes and definitions
#ifdef USE_OPENMP
    #include <omp.h>
    // You can add any OpenMP-specific macros or typedefs here
    #define OMP_NUM_THREADS omp_get_max_threads()
#else
    // Fallback definitions for non-OpenMP builds
    #define OMP_NUM_THREADS 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations of the structs (if they're not defined in this header)
struct sparsemat;
struct iarray;
struct darray;

SPARSE_LIB_API void destroy_sparsemat(struct sparsemat* matrix);
SPARSE_LIB_API void destroy_iarray(struct iarray* matrix);
SPARSE_LIB_API void destroy_darray(struct darray* matrix);
SPARSE_LIB_API void modifyalloc(struct sparsemat* matrix, int imemSize);
SPARSE_LIB_API struct sparsemat* create_sparsemat(int rows, int cols, int nzmax);
SPARSE_LIB_API struct darray* create_darray(int rows, int cols);
SPARSE_LIB_API void limits(int tcov_rows, int numprocs, struct iarray* result);
SPARSE_LIB_API void sparsework_nosym(const struct sparsemat* matrixa, const struct sparsemat* matrixb, struct sparsemat* matrixc, int startIndex, int endIndex, int memIncrease);
SPARSE_LIB_API void sparsework_sym(const struct sparsemat* matrixa, const struct sparsemat* matrixb, struct sparsemat* matrixc, int startIndex, int endIndex, int memIncrease);
SPARSE_LIB_API void dense_sym(const struct sparsemat* matrixa, const struct sparsemat* matrixb, struct darray* matrixc);
SPARSE_LIB_API void dense_nosym(const struct sparsemat* matrixa, const struct sparsemat* matrixb, struct darray* matrixc);
SPARSE_LIB_API void sparse_nosym(const struct sparsemat* const matrixA, const struct sparsemat* const matrixB, struct sparsemat* const matrixC, int imemSize);
SPARSE_LIB_API void sparse_sym(const struct sparsemat* const matrixA, const struct sparsemat* const matrixB, struct sparsemat* const matrixC, int imemSize);
SPARSE_LIB_API void triple_product(struct sparsemat* H, struct sparsemat* Q, struct darray* C, int compute_full_matrix);

#ifdef __cplusplus
}
#endif

#endif // FUNCTIONS_INCLUDED
