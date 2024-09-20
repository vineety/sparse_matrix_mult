Parallel Sparse Matrix Multiplication Routines
Overview
This repository provides efficient sparse-sparse (SS) matrix multiplication algorithms designed for use in linear atmospheric inverse problems. The package improves computational efficiency in calculating covariance matrices and posterior uncertainties.

The package implements parallelized matrix multiplication routines using OpenMP to improve performance on multi-core processors.

Key Features
Hybrid Parallel Sparse-Sparse Multiplication: Efficiently multiplies two sparse matrices in compressed sparse row (CSR) format using parallelization.
Multiple Output Formats: Supports both sparse and dense matrix outputs, with options for symmetric or non-symmetric results.
Triple Product Calculation: Efficient computation of triple products 
𝐴
×
𝐵
×
𝐴
𝑇
A×B×A 
T
 , often used in covariance and uncertainty calculations.
Requirements
C/C++ compiler with OpenMP support (e.g., gcc, clang, or MinGW on Windows)
Python 3.6+ with numpy and scipy installed
Routines
The package includes the following routines for Sparse x Sparse matrix multiplication:

Sparse x Sparse → Sparse (CSR):

python
Copy code
result_sparse = sparse_matrix_multiply(matrix_a, matrix_b, output_format='sparse', symmetric=False)
Sparse x Sparse → Dense:

python
Copy code
result_dense = sparse_matrix_multiply(matrix_a, matrix_b, output_format='dense', symmetric=False)
Sparse x Sparse → Symmetric Sparse (CSR):

python
Copy code
result_sym_sparse = sparse_matrix_multiply(matrix_a, matrix_b, output_format='sparse', symmetric=True)
Sparse x Sparse → Symmetric Dense:

python
Copy code
result_sym_dense = sparse_matrix_multiply(matrix_a, matrix_b, output_format='dense', symmetric=True)
Triple Product (A x B x A') → Symmetric Dense:

python
Copy code
result_triple_sym_dense = sparse_matrix_multiply(matrix_a, matrix_b, output_format='dense', use_triple_product=True)
Installation
Using pip (Recommended)
bash
Copy code
pip install .
For development:

bash
Copy code
pip install -e . -v
Using Makefiles
If pip install fails, compile manually using system-specific Makefiles:

macOS: make -f Makefile.mac
Linux: make -f Makefile.linux
Windows (MinGW): See Windows Installation Guide below
Windows Installation Guide
Install MinGW-w64:

Download from MinGW-w64 website
Extract to a folder (e.g., C:\mingw-w64)
Add C:\mingw-w64\bin to system PATH
Install Anaconda (Optional but Recommended):

Download from Anaconda website
Ensure "Add Anaconda to my PATH environment variable" is checked during installation
Install Dependencies:

bash
Copy code
pip install numpy scipy
Compile the Package:

bash
Copy code
cd path\to\sparse_matrix_mult
mingw32-make -f Makefile.windows_mingw
Install the Package:

bash
Copy code
pip install .
Verify Installation:

bash
Copy code
python matrix_ops_script.py
pytest
Quick Example
Here’s a test example that demonstrates the package in action:

python
Copy code
import numpy as np
from scipy import sparse
from sparse_matrix_mult import sparse_matrix_multiply
import time

def test_sparse_multiply_performance():
    # Create two sparse matrices
    size = 10000
    density = 0.01
    A = sparse.random(size, size, density=density, format='csr')
    B = sparse.random(size, size, density=density, format='csr')

    # Multiply using our package
    start_time = time.time()
    result_our = sparse_matrix_multiply(A, B, output_format='sparse', symmetric=False)
    our_time = time.time() - start_time

    # Multiply using SciPy
    start_time = time.time()
    result_scipy = A.dot(B)
    scipy_time = time.time() - start_time

    # Verify results
    assert np.allclose(result_our.toarray(), result_scipy.toarray(), rtol=1e-5, atol=1e-8)
    
    # Report times
    print(f"Our package time: {our_time:.4f} seconds")
    print(f"SciPy time: {scipy_time:.4f} seconds")
    print(f"Speedup: {scipy_time / our_time:.2f}x")

# Run the test
test_sparse_multiply_performance()
This example:

Creates two large sparse matrices using SciPy’s random sparse matrix generator.
Multiplies these matrices using both our package and SciPy's method, measuring the time taken for each.
Verifies that the results from both methods match within a small tolerance.
Prints the execution times and speedup.
Running Tests
Run the test suite to verify performance:

bash
Copy code
pytest
Troubleshooting
Missing Build Tools:

macOS: xcode-select --install
Linux: sudo apt-get install build-essential
Windows: Install MinGW-w64
Missing Python Dependencies:

bash
Copy code
pip install numpy scipy
OpenMP on macOS:

bash
Copy code
brew install libomp
Clean Build:

bash
Copy code
make clean
make
Citation
If you use this package in your research, please cite:

yaml
Copy code
Yadav, V. and Michalak, A. M.: Technical Note: Improving the computational efficiency of sparse matrix multiplication in linear atmospheric inverse problems, Geosci. Model Dev. Discuss., doi:10.5194/gmd-2016-204, 2016.
Acknowledgments
This work was supported by the National Science Foundation under Grant No. 1342076. The research was conducted at the Jet Propulsion Laboratory, California Institute of Technology, under a contract between the Carnegie Institution of Washington and NASA.

License
This package is released under the MIT License.

Contact
For questions or support, please contact Vineet Yadav at yadavvineet@gmail.com.




