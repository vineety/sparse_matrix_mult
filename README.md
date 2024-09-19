# Efficient Sparse Matrix Multiplication for Atmospheric Inverse Problems

## Overview
This repository contains implementations of efficient sparse-sparse (SS) matrix multiplication algorithms specifically designed for linear atmospheric inverse problems. The algorithms improve computational efficiency in calculating covariance matrices of observations and a posteriori uncertainties.

## Key Features
1. A hybrid-parallel sparse-sparse matrix multiplication approach that is about 33% more efficient in execution time and requires about 38% fewer floating-point operations compared to standard algorithms.
2. Modifications for operations typical in atmospheric inverse problems:
   - Yielding only upper triangular matrices
   - Producing dense matrices from sparse matrix multiplication

## Algorithms
The package includes three main types of SS matrix multiplication routines:
1. Multiplication of a sparse matrix H with a sparse diagonal, sparse block-diagonal, or full-dense covariance matrix Q
2. Multiplication of a sparse HQ with a sparse H^T resulting in a sparse symmetric matrix
3. Multiplication of a sparse HQ with a sparse H^T resulting in a dense symmetric matrix

## Installation
To compile the source code, use the provided Makefile:

```bash
make
```

## Usage
The package provides C++ routines that can be integrated into atmospheric inverse modeling software. See the main function in the source code for usage examples with small test matrices.

For large matrices, use the file-based main function to read sparse matrices from files and write the output to a file.

## Performance
On test cases, the proposed algorithm shows:
- ~33% faster execution time
- ~38% fewer floating-point operations
compared to Intel's MKL csrmultcsr routine.

## Requirements
- C++ compiler with OpenMP support
- Intel Math Kernel Library (for comparison purposes)

## Citation
If you use this code in your research, please cite:
```
Yadav, V. and Michalak, A. M.: Technical Note: Improving the computational efficiency of sparse matrix multiplication in linear atmospheric inverse problems, Geosci. Model Dev. Discuss., doi:10.5194/gmd-2016-204, 2016.
```

## Acknowledgments
This work was supported by funds from National Science Foundation under Grant No. 1342076. The research was carried out at the Jet Propulsion Laboratory, California Institute of Technology, under a contract NNN15R040T between Carnegie Institution of Washington and National Aeronautics and Space Administration.

## License
[Insert appropriate license information here]

## Contact
For questions or support, please contact:
Vineet Yadav (yadavvineet@gmail.com)

## Installation Instructions

## Installation

### Option 1: Using setup.py (Recommended)

1. Ensure you have Python 3.6+ and pip installed.
2. Run:
   ```
   pip install .
   ```
   or
   ```
   python setup.py install
   ```

### Option 2: Using Makefiles

#### macOS
```
make -f Makefile.mac
```

#### Linux
```
make -f Makefile.linux
```

#### Windows
Instructions for Using MinGW and Compiling the Code
To compile the package using MinGW, follow these steps:

Download MinGW:
You can download MinGW from this link.
The version I downloaded was x86_64-14.2.0-release-mcf-seh-ucrt-rt_v12-rev0.7z.
These are pre-built binaries, so after downloading, extract the files into a folder (e.g., name it mingw).

Set up the Environment:
To use the MinGW g++ compiler and make directly from the command line, follow these steps:

Open Start, search for Environment Variables, and select Edit the system environment variables.
In the System Properties window, click Environment Variables.
Under System Variables, find the Path variable and select Edit.
Click New and add the path to the bin directory of the MinGW folder (e.g., C:\path\to\mingw\bin).
Install Anaconda (Optional but Recommended):
It is recommended to install Anaconda and add its path to the system variables as well. This will allow you to access Python (by simply typing python in the command line) and use packages like numpy.

During Anaconda installation, make sure to check the option to Add Anaconda to my PATH environment variable.

Compiling the Package:
Once you have access to g++, navigate to the root folder of the package and run the following command to compile the code:

mingw32-make -f Makefile.windows_mingw

```

### Troubleshooting

- **Build Tools**: 
  - macOS: Install Xcode Command Line Tools (`xcode-select --install`)
  - Linux: Install GCC and Make (`sudo apt-get install build-essential` on Ubuntu)
  - Windows: Install MinGW-w64

- **Python Dependencies**:
  ```
  pip install numpy scipy
  ```

- **OpenMP on macOS**:
  ```
  brew install libomp
  ```

- **Clean Build**:
  ```
  make clean
  make
  ```

### Verifying Installation and Example Use Cases

```python
An example script  "matrix_ops_script.py" is also included to test to compare results against numpy and to demonstrate how to use the code. For ideal performance compile with openmp

For further assistance, please open an issue on our GitHub repository.
If you successfully compile the library on macOS or Windows and would like to contribute build scripts, please open a pull request!
```

