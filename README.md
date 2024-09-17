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
Vineet Yadav (vineet.yadav@jpl.nasa.gov)

## Compilation Instructions

This package includes pre-compiled libraries for x64 and ARM64 architectures on Linux, macOS, and Windows. However, if you need to compile the libraries yourself, we provide Makefiles for Linux systems:

### Linux Compilation

We provide two Makefiles for different Linux compilation scenarios:

1. **Building for Current Architecture**

   If you want to build the library for your current Linux system architecture:

   1. Navigate to the `src/` directory.
   2. Run the following command:

      ```
      make -f Makefile.linux
      ```

   This will generate the appropriate library for your system's architecture.

2. **Cross-Compilation (x64 and ARM64)**

   If you need to build libraries for both x64 and ARM64 architectures on Linux:

   1. Ensure you have the necessary cross-compilation toolchain installed (aarch64-linux-gnu-g++).
   2. Navigate to the `src/` directory.
   3. Run the following command:

      ```
      make -f Makefile.linux_cross_compile
      ```

   This will generate libraries for both x64 and ARM64 architectures.

   Note: Cross-compilation requires appropriate tools and may not work on all systems.

For both Makefiles, you can use `make -f <Makefile_name> clean` to remove compiled libraries, and `make -f <Makefile_name> help` for more information.

### macOS and Windows Compilation

We do not provide Makefiles for macOS or Windows due to the variability in development environments on these platforms. If you need to compile the library on these systems:

- **macOS**: You may need to use Homebrew to install dependencies and manually compile the source files. Please refer to the source files and compilation flags in the Linux Makefiles for guidance.

- **Windows**: Consider using Visual Studio or MinGW to compile the source files. The compilation flags and source file list in the Linux Makefiles can serve as a reference.

If you successfully compile the library on macOS or Windows and would like to contribute build scripts, please open a pull request!
