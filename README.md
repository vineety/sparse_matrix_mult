# Parallel Sparse Matrix Multiplication Routines

## Overview

This repository provides efficient sparse-sparse (SS) matrix multiplication algorithms designed for use in linear atmospheric inverse problems. The package improves computational efficiency when calculating covariance matrices for observations and a posteriori uncertainties.

The package implements parallelized matrix multiplication routines using OpenMP for enhanced performance on multi-core processors.

## Key Features

- **Hybrid Parallel Sparse-Sparse Multiplication**: This package uses a hybrid-parallel approach to multiply two sparse matrices in compressed sparse row (CSR) format.
- **Multiple Output Formats**: Supports both sparse and dense matrix outputs, with options for symmetric or non-symmetric results.
- **Triple Product Calculation**: Efficient computation of triple products A×B×A^T, often used in covariance and uncertainty calculations.

## Requirements

- C/C++ compiler with OpenMP support (e.g., gcc, clang, or MinGW on Windows)
- Python 3.6+ with numpy and scipy installed

## Routines

The package includes the following routines for Sparse x Sparse matrix multiplication:

1. **Sparse x Sparse → Sparse (CSR)**:
   ```python
   result_sparse = sparse_matrix_multiply(matrix_a, matrix_b, output_format='sparse', symmetric=False)
   ```

2. **Sparse x Sparse → Dense**:
   ```python
   result_dense = sparse_matrix_multiply(matrix_a, matrix_b, output_format='dense', symmetric=False)
   ```

3. **Sparse x Sparse → Symmetric Sparse (CSR)**:
   ```python
   result_sym_sparse = sparse_matrix_multiply(matrix_a, matrix_b, output_format='sparse', symmetric=True)
   ```

4. **Sparse x Sparse → Symmetric Dense**:
   ```python
   result_sym_dense = sparse_matrix_multiply(matrix_a, matrix_b, output_format='dense', symmetric=True)
   ```

5. **Triple Product (A x B x A') → Symmetric Dense**:
   ```python
   result_triple_sym_dense = sparse_matrix_multiply(matrix_a, matrix_b, output_format='dense', use_triple_product=True)
   ```

## Installation

### Using pip (Recommended)

```bash
pip install .
```

For development:

```bash
pip install -e . -v
```

### Using Makefiles

If pip install fails, compile manually using system-specific Makefiles:

- **macOS**: `make -f Makefile.mac`
- **Linux**: `make -f Makefile.linux`
- **Windows (MinGW)**: See Windows Installation Guide below

## Windows Installation Guide

1. **Install MinGW-w64**:
   - Download from [MinGW-w64 website](https://sourceforge.net/projects/mingw-w64/)
   - Extract to a folder (e.g., `C:\mingw-w64`)
   - Add `C:\mingw-w64\bin` to system PATH

2. **Install Anaconda** (Optional but Recommended):
   - Download from [Anaconda website](https://www.anaconda.com/products/distribution)
   - Ensure "Add Anaconda to my PATH environment variable" is checked during installation

3. **Install Dependencies**:
   ```bash
   pip install numpy scipy
   ```

4. **Compile the Package**:
   ```bash
   cd path\to\sparse_matrix_mult
   mingw32-make -f Makefile.windows_mingw
   ```

5. **Install the Package**:
   ```bash
   pip install .
   ```

6. **Verify Installation**:
   ```bash
   python matrix_ops_script.py
   pytest
   ```

## Troubleshooting

- **Missing Build Tools**:
  - macOS: `xcode-select --install`
  - Linux: `sudo apt-get install build-essential`
  - Windows: Install MinGW-w64

- **Missing Python Dependencies**:
  ```bash
  pip install numpy scipy
  ```

- **OpenMP on macOS**:
  ```bash
  brew install libomp
  ```

- **Cleaning Build Artifacts**:
  ```bash
  make clean
  make
  ```

## Citation

If you use this package in your research, please cite:

```
Yadav, V. and Michalak, A. M.: Technical Note: Improving the computational efficiency of sparse matrix multiplication in linear atmospheric inverse problems, Geosci. Model Dev. Discuss., doi:10.5194/gmd-2016-204, 2016.
```

## Acknowledgments

This work was supported by the National Science Foundation under Grant No. 1342076. The research was conducted at the Jet Propulsion Laboratory, California Institute of Technology, under a contract between the Carnegie Institution of Washington and NASA.

## License

This package is released under the MIT License.

## Contact

For questions or support, please contact Vineet Yadav at yadavvineet@gmail.com.



