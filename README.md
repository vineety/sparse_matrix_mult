Parallel Sparse Matrix Multiplication Routines
Overview
This repository provides efficient sparse-sparse (SS) matrix multiplication algorithms designed for use in linear atmospheric inverse problems. The package improves computational efficiency when calculating covariance matrices for observations and a posteriori uncertainties.

The package implements parallelized matrix multiplication routines using OpenMP for enhanced performance on multi-core processors.

Key Features
Hybrid Parallel Sparse-Sparse Multiplication: This package uses a hybrid-parallel approach to multiply two sparse matrices in compressed sparse row (CSR) format.
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
Multiplies two sparse matrices 
𝐴
×
𝐵
A×B, resulting in a sparse matrix 
𝐶
C in CSR format. You can control whether the output is symmetric or non-symmetric.

python
Copy code
result_sparse = sparse_matrix_multiply(matrix_a, matrix_b, output_format='sparse', symmetric=False)
Sparse x Sparse → Dense:
Multiplies two sparse matrices, 
𝐴
×
𝐵
A×B, and outputs a dense matrix 
𝐶
C. This is useful when the result is expected to be dense.

python
Copy code
result_dense = sparse_matrix_multiply(matrix_a, matrix_b, output_format='dense', symmetric=False)
Sparse x Sparse → Symmetric Sparse (CSR):
Multiplies two sparse matrices, 
𝐴
×
𝐵
A×B, and outputs a symmetric sparse matrix. Only the upper triangular matrix is computed, assuming symmetry.

python
Copy code
result_sym_sparse = sparse_matrix_multiply(matrix_a, matrix_b, output_format='sparse', symmetric=True)
Sparse x Sparse → Symmetric Dense:
Similar to the symmetric sparse operation, but the output is a dense matrix. This is useful for operations where the result is expected to be dense.

python
Copy code
result_sym_dense = sparse_matrix_multiply(matrix_a, matrix_b, output_format='dense', symmetric=True)
Triple Product (A x B x A') → Symmetric Dense:
Efficiently computes the triple product 
𝐴
×
𝐵
×
𝐴
𝑇
A×B×A 
T
 , often used in covariance calculations. The output is a symmetric dense matrix.

python
Copy code
result_triple_sym_dense = sparse_matrix_multiply(matrix_a, matrix_b, output_format='dense', use_triple_product=True)
Installation
Using pip (Recommended)
Ensure you have Python 3.6+ and pip installed.
Install the package:
bash
Copy code
pip install .
Alternatively, if you download the package from GitHub, navigate to the root directory and install with:
bash
Copy code
pip install -e . -v
Using Makefiles
If pip install fails due to missing dependencies or OpenMP support, you can compile the library manually using the system-specific Makefiles:

macOS
bash
Copy code
make -f Makefile.mac
Linux
bash
Copy code
make -f Makefile.linux
Windows (MinGW)
Windows Installation Guide
1. Install MinGW-w64
MinGW-w64 is required to compile the C++ code and link with OpenMP. Follow these steps:

Download MinGW-w64:

Visit the MinGW-w64 website and download the version that supports your system.
Example: Download the file x86_64-14.2.0-release-mcf-seh-ucrt-rt_v12-rev0.7z.
Extract MinGW:

Extract the downloaded file into a folder. For example, extract it to C:\mingw-w64.
Add MinGW to System Path:

Open the Start Menu, search for Environment Variables, and select Edit the system environment variables.
In the System Properties window, click Environment Variables.
Under System Variables, find the Path variable and click Edit.
Click New and add the path to the bin directory of your MinGW folder, e.g., C:\mingw-w64\bin.
This will allow you to use the g++ and make commands from the command line.
2. Install Python and Anaconda (Optional but Recommended)
Anaconda makes managing Python environments and dependencies easier. Follow these steps to install Anaconda:

Download Anaconda:

Visit the Anaconda website and download the Python 3.x version for Windows.
Install Anaconda:

During installation, ensure the option Add Anaconda to my PATH environment variable is checked.
This will allow you to run Python from the command line.
Verify Installation:

Open a command prompt and run python --version to check that Python is properly installed.
3. Install Dependencies
Before compiling, install the necessary Python dependencies:

bash
Copy code
pip install numpy scipy
4. Compiling the Package with MinGW
Once MinGW is set up and Python dependencies are installed, you can compile the package using the provided Makefile for Windows.

Navigate to the Package Directory: Open a command prompt and navigate to the root folder of the downloaded package:

bash
Copy code
cd path\to\sparse_matrix_mult
Compile the Package: Run the Makefile.windows_mingw file to compile the C++ code and create the necessary DLL files:

bash
Copy code
mingw32-make -f Makefile.windows_mingw
5. Verify Installation
After compilation, verify that the .dll files were created in the sparse_matrix_mult/lib directory. These DLL files are the compiled shared libraries that the Python interface will use.

6. Installing the Package
Once the compilation is successful, install the Python package by running the following command in the root directory:

bash
Copy code
pip install .
Alternatively, use the following for an editable install, which allows for testing and development:

bash
Copy code
pip install -e . -v
7. Running Tests and Example Scripts
To verify the installation, you can run the provided test scripts or use pytest to run all unit tests.

Run Example Script: In the sparse_matrix_mult directory, run the example script to test sparse matrix multiplication:

bash
Copy code
python matrix_ops_script.py
Run Unit Tests: Use pytest to run all the tests in the package:

bash
Copy code
pytest
Additional Information for OpenMP on Windows
OpenMP should be supported automatically when using MinGW, but make sure that g++ supports OpenMP:

To check if g++ supports OpenMP, run the following command:
bash
Copy code
g++ -fopenmp -o omp_test omp_test.cpp
If you encounter errors related to OpenMP, you may need to ensure that the correct version of MinGW with OpenMP support is installed. Additionally, if you are using a version of Python compiled without OpenMP support, you may need to recompile it.

Troubleshooting
Common Issues
Missing Build Tools:

macOS: Install Xcode Command Line Tools (xcode-select --install).
Linux: Install GCC and Make (sudo apt-get install build-essential on Ubuntu).
Windows: Install MinGW-w64.
Missing Python Dependencies:

bash
Copy code
pip install numpy scipy
OpenMP on macOS: If you encounter issues with OpenMP, you may need to install libomp:

bash
Copy code
brew install libomp
Cleaning Build Artifacts: If you're encountering issues with old builds, try cleaning the build directory:

bash
Copy code
make clean
make
Citation
If you use this package in your research, please cite the following paper:

yaml
Copy code
Yadav, V. and Michalak, A. M.: Technical Note: Improving the computational efficiency of sparse matrix multiplication in linear atmospheric inverse problems, Geosci. Model Dev. Discuss., doi:10.5194/gmd-2016-204, 2016.
Acknowledgments
This work was supported by the National Science Foundation under Grant No. 1342076. The research was conducted at the Jet Propulsion Laboratory, California Institute of Technology, under a contract between the Carnegie Institution of Washington and NASA.

License
This package is released under the MIT License.

Contact
For questions or support, please contact Vineet Yadav at yadavvineet@gmail.com.



