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
    
matrix_ops = MatrixOpsLibrary()

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
    print("Sparse Matrix Multiplication Examples")
    print("=====================================")

    # Define demonstration matrices
    A_matrix = np.array([
        [0.64, 0.99, 0.89, 0.72],
        [0,    0.67, 0.54, 0   ],
        [0,    0.32, 0,    0   ],
        [0.1,  0,    0,    0   ]
    ])
    
    B_matrix = np.array([
        [0.23, 0,    0,    0.51],
        [0,    0.72, 0,    0   ],
        [0,    0,    0.99, 0   ],
        [0,    0.76, 0.87, 0.97]
    ])

    C_matrix = np.array([
        [1, 2, 3, 4],
        [0, 5, 6, 0],
        [0, 0, 7, 8],
        [9, 0, 0, 10]
    ])

    D_matrix = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8]
    ])

    print("\nDemonstration Matrices:")
    print("A_matrix:")
    print(A_matrix)
    print("\nB_matrix:")
    print(B_matrix)
    print("\nC_matrix:")
    print(C_matrix)
    print("\nD_matrix:")
    print(D_matrix)

    # Example 1: Using the installed package
    print("\nExample 1: Using the installed package")
    print("---------------------------------------")

    try:
        # Sparse non-symmetric multiplication
        result_sparse = sparse_matrix_multiply(A_matrix, B_matrix, output_format='sparse', symmetric=False)
        print("Result of sparse non-symmetric A * B:")
        print(result_sparse.toarray())

        # Dense non-symmetric multiplication
        result_dense = sparse_matrix_multiply(A_matrix, B_matrix, output_format='dense', symmetric=False)
        print("\nResult of dense non-symmetric A * B:")
        print(result_dense)

        # Dense symmetric multiplication
        result_sym_dense = sparse_matrix_multiply(C_matrix, C_matrix.T, output_format='dense', symmetric=True)
        print("\nResult of dense symmetric C * C^T (upper triangular):")
        print(np.triu(result_sym_dense))

        # Sparse symmetric multiplication
        result_sym_sparse = sparse_matrix_multiply(C_matrix, C_matrix.T, output_format='sparse', symmetric=True)
        print("\nResult of sparse symmetric C * C^T (upper triangular):")
        print(np.triu(result_sym_sparse.toarray()))

    except ValueError as e:
        print(f"Error using installed package: {e}")

    # Example 2: Using manually compiled library
    print("\nExample 2: Using manually compiled library")
    print("-------------------------------------------")
    print("If you've manually compiled the library using the Makefile, you can use it as follows:")
    
    # Trying to load the library from common locations
    library_paths = [
        "./sparse_matrix_mult/lib",  # Current directory
        "/usr/local/lib",            # Common system-wide location
        "/usr/lib",                  # Another common system-wide location
    ]
    
    lib = None
    for path in library_paths:
        try:
            if platform.system() == "Darwin":
                lib_name = "libsparse.dylib"
            elif platform.system() == "Linux":
                lib_name = "libsparse.so"
            else:
                lib_name = "libsparse.dll"
            
            full_path = os.path.join(path, lib_name)
            if os.path.exists(full_path):
                lib = ctypes.CDLL(full_path)
                print(f"Successfully loaded library from: {full_path}")
                break
        except OSError:
            continue

    if lib is None:
        print("Could not find the manually compiled library. Please ensure it's in one of the searched locations.")
    else:
        # Define the function prototypes
        lib.sparse_nosym.argtypes = [ctypes.POINTER(SparseMat), ctypes.POINTER(SparseMat),
                                     ctypes.POINTER(SparseMat), ctypes.c_int]
        lib.sparse_nosym.restype = None

        lib.sparse_sym.argtypes = [ctypes.POINTER(SparseMat), ctypes.POINTER(SparseMat),
                                   ctypes.POINTER(SparseMat), ctypes.c_int]
        lib.sparse_sym.restype = None

        lib.dense_nosym.argtypes = [ctypes.POINTER(SparseMat), ctypes.POINTER(SparseMat), ctypes.POINTER(DArray)]
        lib.dense_nosym.restype = None

        # Convert numpy arrays to SparseMat structures
        spmat_A = csr_to_sparsemat(csr_matrix(A_matrix))
        spmat_B = csr_to_sparsemat(csr_matrix(B_matrix))
        spmat_C = csr_to_sparsemat(csr_matrix(C_matrix))

        # Sparse non-symmetric multiplication
        result_ptr_sparse = lib.create_sparsemat(A_matrix.shape[0], B_matrix.shape[1], A_matrix.shape[0] * B_matrix.shape[1])
        lib.sparse_nosym(ctypes.byref(spmat_A), ctypes.byref(spmat_B), result_ptr_sparse, ctypes.c_int(10))
        result_sparse = sparsemat_to_csr(result_ptr_sparse).toarray()

        print("Result of sparse non-symmetric A * B using manually compiled library:")
        print(result_sparse)

        # Dense non-symmetric multiplication
        result_ptr_dense = lib.create_darray(A_matrix.shape[0], B_matrix.shape[1])
        lib.dense_nosym(ctypes.byref(spmat_A), ctypes.byref(spmat_B), result_ptr_dense)
        result_dense = darray_to_numpy(result_ptr_dense)

        print("\nResult of dense non-symmetric A * B using manually compiled library:")
        print(result_dense)

        # Sparse symmetric multiplication
        result_ptr_sym_sparse = lib.create_sparsemat(C_matrix.shape[0], C_matrix.shape[0], C_matrix.shape[0] * C_matrix.shape[0])
        lib.sparse_sym(ctypes.byref(spmat_C), ctypes.byref(spmat_C), result_ptr_sym_sparse, ctypes.c_int(10))
        result_sym_sparse = sparsemat_to_csr(result_ptr_sym_sparse).toarray()

        print("\nResult of sparse symmetric C * C^T using manually compiled library (upper triangular):")
        print(np.triu(result_sym_sparse))

        # Clean up
        lib.destroy_sparsemat(result_ptr_sparse)
        lib.destroy_sparsemat(result_ptr_sym_sparse)
        lib.destroy_darray(result_ptr_dense)

    print("\nNote: If you're having trouble with the installed package, you can:")
    print("1. Compile the library manually using the provided Makefile:")
    print("   - For Mac: make -f Makefile.mac")
    print("   - For Linux: make -f Makefile.linux")
    print("2. Ensure the compiled library is in one of the searched locations.")
    print("3. Run this script to use the manually compiled library.")

