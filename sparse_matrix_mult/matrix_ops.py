"""
@file matrix_ops.py
@brief This module provides a Python interface to C-based matrix operations.

This module defines classes and functions for efficient matrix operations,
particularly for sparse matrices. It interfaces with a C library for
performance-critical operations.
"""

import ctypes
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
import os
import platform


class SparseMat(ctypes.Structure):
    """
    @class SparseMat
    @brief A structure representing a sparse matrix in CSR (Compressed Sparse Row) format.

    This class defines a sparse matrix with attributes that store the number of non-zero elements,
    dimensions of the matrix, row pointer, column index, and the values of non-zero elements.
    """
    _fields_ = [
        ("nzmax", ctypes.c_int),  #/**< Maximum number of non-zero entries in the matrix. */
        ("rows", ctypes.c_int),   #/**< Number of rows in the matrix. */
        ("cols", ctypes.c_int),   #/**< Number of columns in the matrix. */
        ("rowPtr", ctypes.POINTER(ctypes.c_int)), #/**< Pointer to the row pointer array. */
        ("colInd", ctypes.POINTER(ctypes.c_int)), #/**< Pointer to the column index array. */
        ("values", ctypes.POINTER(ctypes.c_double)) #/**< Pointer to the values of non-zero elements. */
    ]


class DArray(ctypes.Structure):
    """
    @class DArray
    @brief A structure representing a dense matrix in double precision format.

    This class defines a dense matrix with attributes that store the number of rows,
    columns, and the data in the form of a 2D array.
    """
    _fields_ = [
        ("array", ctypes.POINTER(ctypes.c_double)), #/**< Pointer to the array of matrix elements. */
        ("rows", ctypes.c_int),  #/**< Number of rows in the matrix. */
        ("cols", ctypes.c_int),  #/**< Number of columns in the matrix. */
    ]


class MatrixOpsLibrary:
    """
    @class MatrixOpsLibrary
    @brief Singleton class to load the C-based matrix operations library and set up function prototypes.

    This class ensures that the C-based matrix operations library is loaded only once,
    and provides a method to access the library. It also defines function prototypes for
    the matrix operations exposed by the C library.
    """
    _instance = None
    _lib = None

    def __new__(cls):
        """
        @brief Create a new instance of the MatrixOpsLibrary singleton or return the existing one.

        @return MatrixOpsLibrary instance.
        """
        if cls._instance is None:
            cls._instance = super(MatrixOpsLibrary, cls).__new__(cls)
            cls._instance._load_library()
        return cls._instance

    def _load_library(self):
        """
        @brief Load the C library based on the platform and architecture.
        This method selects the correct shared library file (.so, .dylib, or .dll)
        based on the operating system and processor architecture.
        """
        if self._lib is not None:
            return

        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == 'linux':
            extension = 'so'
        elif system == 'darwin':
            extension = 'dylib'
        elif system == 'windows':
            extension = 'dll'
        else:
            raise OSError(f"Unsupported operating system: {system}")

        if 'arm' in machine or 'aarch64' in machine:
            arch = 'arm64'
        elif 'x86_64' in machine or 'amd64' in machine:
            arch = 'x64'
        else:
            raise OSError(f"Unsupported architecture: {machine}")

        lib_name = f'libsparse_{arch}.{extension}'
        package_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(package_dir, 'lib', lib_name)

        if not os.path.exists(lib_path):
            raise OSError(f"Library not found: {lib_path}")

        try:
            self._lib = ctypes.CDLL(lib_path)
            print(f"Successfully loaded library from: {lib_path}")
            self._setup_function_prototypes()
        except OSError as e:
            raise OSError(f"Failed to load library: {lib_path}. Error: {e}")

    def _setup_function_prototypes(self):
        """
        @brief Set up the argument and return types for the matrix operations defined in the C library.
        """
        self._lib.create_sparsemat.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self._lib.create_sparsemat.restype = ctypes.POINTER(SparseMat)

        self._lib.create_darray.argtypes = [ctypes.c_int, ctypes.c_int]
        self._lib.create_darray.restype = ctypes.POINTER(DArray)

        self._lib.sparse_nosym.argtypes = [ctypes.POINTER(SparseMat), ctypes.POINTER(SparseMat),
                                           ctypes.POINTER(SparseMat), ctypes.c_int]
        self._lib.sparse_nosym.restype = None

        self._lib.sparse_sym.argtypes = [ctypes.POINTER(SparseMat), ctypes.POINTER(SparseMat),
                                         ctypes.POINTER(SparseMat), ctypes.c_int]
        self._lib.sparse_sym.restype = None

        self._lib.dense_sym.argtypes = [ctypes.POINTER(SparseMat), ctypes.POINTER(SparseMat), ctypes.POINTER(DArray)]
        self._lib.dense_sym.restype = None

        self._lib.dense_nosym.argtypes = [ctypes.POINTER(SparseMat), ctypes.POINTER(SparseMat), ctypes.POINTER(DArray)]
        self._lib.dense_nosym.restype = None

        self._lib.destroy_sparsemat.argtypes = [ctypes.POINTER(SparseMat)]
        self._lib.destroy_sparsemat.restype = None

        self._lib.destroy_darray.argtypes = [ctypes.POINTER(DArray)]
        self._lib.destroy_darray.restype = None

    def get_lib(self):
        """
        @brief Return the loaded C library.

        @return C library instance.
        """
        if self._lib is None:
            self._load_library()
        return self._lib


matrix_ops = MatrixOpsLibrary()


def csr_to_sparsemat(csr):
    """
    @brief Convert a SciPy CSR matrix to a SparseMat structure for use with the C library.

    @param csr SciPy CSR matrix.
    @return SparseMat structure containing the CSR matrix data.
    """
    lib = matrix_ops.get_lib()
    sparsemat = lib.create_sparsemat(csr.shape[0], csr.shape[1], csr.nnz).contents
    rowPtr = np.array(csr.indptr, dtype=np.int32)
    colInd = np.array(csr.indices, dtype=np.int32)
    values = np.array(csr.data, dtype=np.float64)
    ctypes.memmove(sparsemat.rowPtr, rowPtr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), rowPtr.nbytes)
    ctypes.memmove(sparsemat.colInd, colInd.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), colInd.nbytes)
    ctypes.memmove(sparsemat.values, values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), values.nbytes)
    return sparsemat


def sparsemat_to_csr(sparsemat_ptr, symmetric=False):
    """
    @brief Convert a SparseMat structure back to a SciPy CSR matrix.

    @param sparsemat_ptr Pointer to a SparseMat structure.
    @param symmetric Whether the matrix is symmetric.
    @return SciPy CSR matrix equivalent of the SparseMat structure.
    """
    if not sparsemat_ptr:
        raise ValueError("Invalid sparsemat_ptr: Null pointer")

    sparsemat = sparsemat_ptr.contents
    rows = sparsemat.rows
    cols = sparsemat.cols
    nzmax = sparsemat.nzmax

    if nzmax == 0:
        return csr_matrix((rows, cols))

    rowPtr = np.ctypeslib.as_array(sparsemat.rowPtr, shape=(rows + 1,)).copy()
    colInd = np.ctypeslib.as_array(sparsemat.colInd, shape=(nzmax,)).copy()
    values = np.ctypeslib.as_array(sparsemat.values, shape=(nzmax,)).copy()

    return csr_matrix((values, colInd, rowPtr), shape=(rows, cols))


def darray_to_numpy(darray_ptr):
    """
    @brief Convert a DArray structure to a NumPy array.

    @param darray_ptr Pointer to a DArray structure.
    @return NumPy array equivalent of the DArray structure.
    """
    darray = darray_ptr.contents
    numpy_array = np.ctypeslib.as_array(darray.array, shape=(darray.rows, darray.cols)).copy()
    return numpy_array

def create_sparsemat():
    return ctypes.pointer(SparseMat())


def create_darray():
    return ctypes.pointer(DArray())


def sparse_matrix_multiply(matrix_a, matrix_b, output_format='sparse', symmetric=False, imem_size=None,
                           use_triple_product=False, compute_full_matrix=None):
    """
    @brief Perform matrix multiplication of two sparse matrices.

    This function multiplies two sparse matrices using either sparse, dense, or triple-product format.
    
    @param matrix_a First matrix (CSR format or NumPy array).
    @param matrix_b Second matrix (CSR format or NumPy array).
    @param output_format The format of the result ('sparse' or 'dense').
    @param symmetric Whether the result matrix should be symmetric.
    @param imem_size Intermediate memory size for calculations.
    @param use_triple_product Whether to use the triple-product method for dense results.
    @param compute_full_matrix Whether to compute the full matrix or only the upper triangular part.

    @return Resulting matrix (CSR format or NumPy array).
    """
    lib = matrix_ops.get_lib()
    
    # Handle imem_size
    if imem_size is None:
        imem_size = 5
    else:
        try:
            imem_size = int(imem_size)
        except ValueError:
            raise ValueError(f"imem_size must be an integer or None, got {type(imem_size)}")

    # Handle compute_full_matrix
    if compute_full_matrix is None:
        compute_full_matrix = 0
    else:
        if compute_full_matrix not in (0, 1):
            raise ValueError("compute_full_matrix must be None, 0, or 1")
        compute_full_matrix = int(compute_full_matrix)

    if not isspmatrix_csr(matrix_a):
        matrix_a = csr_matrix(matrix_a)
    if not isspmatrix_csr(matrix_b):
        matrix_b = csr_matrix(matrix_b)

    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError("Matrix dimensions are incompatible for multiplication.")

    if matrix_a.nnz == 0 or matrix_b.nnz == 0:
        if output_format == 'sparse':
            return csr_matrix((matrix_a.shape[0], matrix_b.shape[1]))
        else:
            return np.zeros((matrix_a.shape[0], matrix_b.shape[1]))

    if symmetric and (matrix_a.shape[0] != matrix_b.shape[1]):
        raise ValueError("For symmetric output, the resulting matrix must be square.")

    try:
        if use_triple_product:
            result_ptr = create_darray()
            if result_ptr is None:
                raise MemoryError("Failed to create darray for result.")

            spmat_a = csr_to_sparsemat(matrix_a)
            spmat_b = csr_to_sparsemat(matrix_b)

            lib.triple_product(ctypes.byref(spmat_a), ctypes.byref(spmat_b), result_ptr,
                               ctypes.c_int(compute_full_matrix))
            result = darray_to_numpy(result_ptr)
            lib.destroy_darray(result_ptr)

        elif output_format == 'sparse':
            spmat_a = csr_to_sparsemat(matrix_a)
            spmat_b = csr_to_sparsemat(matrix_b)
            result_ptr = create_sparsemat()
            if result_ptr is None:
                raise MemoryError("Failed to create sparsemat for result.")

            if symmetric:
                lib.sparse_sym(ctypes.byref(spmat_a), ctypes.byref(spmat_b), result_ptr, ctypes.c_int(imem_size))
            else:
                lib.sparse_nosym(ctypes.byref(spmat_a), ctypes.byref(spmat_b), result_ptr, ctypes.c_int(imem_size))

            result = sparsemat_to_csr(result_ptr, symmetric)
            lib.destroy_sparsemat(result_ptr)
        elif output_format == 'dense':
            spmat_a = csr_to_sparsemat(matrix_a)
            spmat_b = csr_to_sparsemat(matrix_b)
            result_ptr = create_darray()
            if result_ptr is None:
                raise MemoryError("Failed to create darray for result.")

            if symmetric:
                lib.dense_sym(ctypes.byref(spmat_a), ctypes.byref(spmat_b), result_ptr)
            else:
                lib.dense_nosym(ctypes.byref(spmat_a), ctypes.byref(spmat_b), result_ptr)

            result = darray_to_numpy(result_ptr)
            lib.destroy_darray(result_ptr)

        else:
            raise ValueError("Invalid output_format. Choose 'sparse' or 'dense'.")

        if isinstance(result, np.ndarray) and np.all(result == 0):
            print("Multiplication resulted in a zero matrix.")
        elif isinstance(result, csr_matrix) and result.nnz == 0:
            print("Multiplication resulted in a zero matrix.")

        return result

    except Exception as e:
        print(f"An error occurred during matrix multiplication: {str(e)}")
        print(f"matrix_a shape: {matrix_a.shape}, nnz: {matrix_a.nnz}")
        print(f"matrix_b shape: {matrix_b.shape}, nnz: {matrix_b.nnz}")
        print(f"Parameters: output_format={output_format}, symmetric={symmetric}, "
              f"imem_size={imem_size}, use_triple_product={use_triple_product}, "
              f"compute_full_matrix={compute_full_matrix}")
        if output_format == 'sparse':
            return csr_matrix((matrix_a.shape[0], matrix_b.shape[1]))
        else:
            return np.zeros((matrix_a.shape[0], matrix_b.shape[1]))






