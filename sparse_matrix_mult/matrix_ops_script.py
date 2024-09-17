import ctypes
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
import os
import platform


class SparseMat(ctypes.Structure):
    _fields_ = [
        ("nzmax", ctypes.c_int),
        ("rows", ctypes.c_int),
        ("cols", ctypes.c_int),
        ("rowPtr", ctypes.POINTER(ctypes.c_int)),
        ("colInd", ctypes.POINTER(ctypes.c_int)),
        ("values", ctypes.POINTER(ctypes.c_double))
    ]


class DArray(ctypes.Structure):
    _fields_ = [
        ("array", ctypes.POINTER(ctypes.c_double)),
        ("rows", ctypes.c_int),
        ("cols", ctypes.c_int),
    ]


class MatrixOpsLibrary:
    _instance = None
    _lib = None

    def __new__(cls, library_path=None):
        if cls._instance is None:
            cls._instance = super(MatrixOpsLibrary, cls).__new__(cls)
            cls._instance._load_library(library_path)
        return cls._instance

    def _load_library(self, library_path=None):
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

        # If the user doesn't provide a library path, fall back to the default
        if library_path is None:
            package_dir = os.path.dirname(os.path.abspath(__file__))
            library_path = os.path.join(package_dir, 'lib')

        lib_name = f'libsparse_{arch}.{extension}'
        lib_path = os.path.join(library_path, lib_name)

        if not os.path.exists(lib_path):
            raise OSError(f"Library not found: {lib_path}")

        try:
            self._lib = ctypes.CDLL(lib_path)
            print(f"Successfully loaded library from: {lib_path}")
            self._setup_function_prototypes()
        except OSError as e:
            raise OSError(f"Failed to load library: {lib_path}. Error: {e}")

    def _setup_function_prototypes(self):
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
        if self._lib is None:
            self._load_library()
        return self._lib


# Function to convert csr_matrix to C struct
def csr_to_sparsemat(csr):
    lib = matrix_ops.get_lib()
    sparsemat = lib.create_sparsemat(csr.shape[0], csr.shape[1], csr.nnz).contents
    rowPtr = np.array(csr.indptr, dtype=np.int32)
    colInd = np.array(csr.indices, dtype=np.int32)
    values = np.array(csr.data, dtype=np.float64)
    ctypes.memmove(sparsemat.rowPtr, rowPtr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), rowPtr.nbytes)
    ctypes.memmove(sparsemat.colInd, colInd.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), colInd.nbytes)
    ctypes.memmove(sparsemat.values, values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), values.nbytes)
    return sparsemat


# Convert C struct to csr_matrix
def sparsemat_to_csr(sparsemat_ptr, symmetric=False):
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
    darray = darray_ptr.contents
    numpy_array = np.ctypeslib.as_array(darray.array, shape=(darray.rows, darray.cols)).copy()
    return numpy_array


def sparse_matrix_multiply(matrix_a, matrix_b, output_format='sparse', symmetric=False, imem_size=None,
                           use_triple_product=False, compute_full_matrix=None, library_path=None):
    # Load library from user-specified path or default
    matrix_ops = MatrixOpsLibrary(library_path)
    lib = matrix_ops.get_lib()
    
    # Handle imem_size
    if imem_size is None:
        imem_size = 5
    else:
        imem_size = int(imem_size)

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
            result_ptr = matrix_ops._lib.create_darray()
            spmat_a = csr_to_sparsemat(matrix_a)
            spmat_b = csr_to_sparsemat(matrix_b)
            lib.triple_product(ctypes.byref(spmat_a), ctypes.byref(spmat_b), result_ptr, ctypes.c_int(compute_full_matrix))
            result = darray_to_numpy(result_ptr)
            lib.destroy_darray(result_ptr)

        elif output_format == 'sparse':
            spmat_a = csr_to_sparsemat(matrix_a)
            spmat_b = csr_to_sparsemat(matrix_b)
            result_ptr = matrix_ops._lib.create_sparsemat()
            if symmetric:
                lib.sparse_sym(ctypes.byref(spmat_a), ctypes.byref(spmat_b), result_ptr, ctypes.c_int(imem_size))
            else:
                lib.sparse_nosym(ctypes.byref(spmat_a), ctypes.byref(spmat_b), result_ptr, ctypes.c_int(imem_size))
            result = sparsemat_to_csr(result_ptr, symmetric)
            lib.destroy_sparsemat(result_ptr)

        elif output_format == 'dense':
            spmat_a = csr_to_sparsemat(matrix_a)
            spmat_b = csr_to_sparsemat(matrix_b)
            result_ptr = matrix_ops._lib.create_darray()
            if symmetric:
                lib.dense_sym(ctypes.byref(spmat_a), ctypes.byref(spmat_b), result_ptr)
            else:
                lib.dense_nosym(ctypes.byref(spmat_a), ctypes.byref(spmat_b), result_ptr)
            result = darray_to_numpy(result_ptr)
            lib.destroy_darray(result_ptr)

        else:
            raise ValueError("Invalid output_format. Choose 'sparse' or 'dense'.")

        return result

    except Exception as e:
        print(f"Error during matrix multiplication: {str(e)}")
        return None


# Example script to call the library
if __name__ == "__main__":
    # User can pass a path to the library
    library_dir = "/path/to/custom/library"
    A_matrix = np.array([[1, 2], [3, 4]])
    B_matrix = np.array([[5, 6], [7, 8]])

    try:
        result = sparse_matrix_multiply(A_matrix, B_matrix, output_format='dense', symmetric=False, imem_size=10, library_path=library_dir)
        print(result)
    except ValueError as e:
        print(f"Error: {e}")

