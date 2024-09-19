#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:48:30 2024
python setup.py build_ext
or
pip install -e . -v

@author: vineet
"""

import os
import platform
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.errors import CompileError, DistutilsError

# Global variable to track OpenMP support
openmp_supported = False

class BuildSharedLibrary(build_ext):
    
    def run(self):
        global openmp_supported
        system = platform.system()

        if system == "Windows":
            self.build_windows()
        else:
            architectures = self.get_macos_architectures() if system == "Darwin" else self.get_linux_architectures()
            self.build_unix(system, architectures)

        # Print OpenMP support status at the end of the build process
        if openmp_supported:
            print("\nInstallation completed with OpenMP support.")
            print("Your library will use parallel processing capabilities where available.")
        else:
            print("\nInstallation completed without OpenMP support.")
            print("Your library will run in single-threaded mode.")

    def build_unix(self, system, architectures):
        for arch in architectures:
            try:
                self.try_compile_shared_lib(arch)
            except CompileError as e:
                print(f"Error: Compilation failed on {system} for architecture {arch}.")
                print(f"Details: {str(e)}")
                print("Installation failed due to missing dependencies or compilation issues.")
                print("Please consider using the platform-specific Makefile for manual installation:")
                print(f"- For {system}, use the Makefile in the root directory.")
                
                if len(architectures) > 1:
                    print("Continuing with next architecture...")
                else:
                    raise DistutilsError("Failed to build the shared library.")
            
    def build_windows(self):
        global openmp_supported
        print("Building for Windows...")
        
        # Compiler settings for MinGW (assuming it's installed via Anaconda)
        cxx = "g++"
        cxxflags = ["-Wall", "-O3", "-shared", "-std=c++11"]
        includes = ["-I./include"]
        
        # Directories
        src_dir = "src"
        lib_dir = os.path.join("sparse_matrix_mult", "lib")
        
        # Output library (use .pyd for Windows Python extension)
        library = os.path.join(lib_dir, f"libsparse_{platform.machine()}.pyd")
        
        # Source files
        sources = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith('.cpp')]
        
        # Create lib directory
        os.makedirs(lib_dir, exist_ok=True)
        
        # OpenMP check
        if self.check_openmp_support(cxx):
            print("OpenMP is available. Compiling with OpenMP support.")
            cxxflags.append("-fopenmp")
            openmp_supported = True
        else:
            print("OpenMP not available. Compiling without OpenMP support.")
        
        # Compile and link in one step
        cmd = [cxx] + cxxflags + includes + sources + ["-o", library]
        self.execute_command(cmd, "compiling and linking library")

    def get_macos_architectures(self):
        architectures = []
        if subprocess.call(["sysctl", "-n", "hw.optional.arm64"]) == 0:  # Check if arm64 is supported
            architectures.append('arm64')
        architectures.append('x86_64')  # Always include x86_64 as fallback
        return architectures

    def get_linux_architectures(self):
        machine = platform.machine()
        if machine == 'x86_64':
            return ['x86_64']
        elif machine.startswith('arm') or machine.startswith('aarch64'):
            return ['arm64']
        else:
            return [machine]

    def try_compile_shared_lib(self, arch):
        global openmp_supported
        print(f"Attempting to compile the shared library for {platform.system()} with architecture {arch}...")

        # Compiler settings
        cxx = "g++"
        cxxflags = ["-Wall", "-O3", "-fPIC", "-std=c++11"]
        ldflags = ["-shared"]
        includes = ["-I./include"]
        
        # Directories
        src_dir = "src"
        obj_dir = "obj"
        lib_dir = "sparse_matrix_mult/lib"
        
        # Output library
        library = os.path.join(lib_dir, f"libsparse_{arch}.so")
        
        # Source files
        sources = [f for f in os.listdir(src_dir) if f.endswith('.cpp')]
        objects = [os.path.join(obj_dir, f.replace('.cpp', '.o')) for f in sources]
        
        # Create obj directory
        os.makedirs(obj_dir, exist_ok=True)
        
        # OpenMP check
        if self.check_openmp_support(cxx):
            print("OpenMP is available. Compiling with OpenMP support.")
            cxxflags.append("-fopenmp")
            ldflags.append("-fopenmp")
            openmp_supported = True
        else:
            print("OpenMP not available. Compiling without OpenMP support.")
        
        # Compile object files
        for source, obj in zip(sources, objects):
            cmd = [cxx] + cxxflags + includes + ["-c", os.path.join(src_dir, source), "-o", obj]
            self.execute_command(cmd, f"compiling {source}")
        
        # Link shared library
        os.makedirs(lib_dir, exist_ok=True)  # Ensure lib directory exists
        cmd = [cxx] + ldflags + objects + ["-o", library]
        self.execute_command(cmd, "linking shared library")

    def check_openmp_support(self, compiler):
        """Check if OpenMP is supported by compiling and running a small test program."""
        omp_test_code = r"""
        #include <omp.h>
        #include <stdio.h>
    
        int main() {
            int nthreads = 0;
            #pragma omp parallel
            {
                #pragma omp atomic
                nthreads++;
            }
            printf("OpenMP test: %d threads\n", nthreads);
            return 0;
        }
        """
        test_c_file = "test_openmp.c"
        test_exe_file = "test_openmp_exec"
        
        with open(test_c_file, 'w') as f:
            f.write(omp_test_code)
    
        # Compile the OpenMP test program
        compile_cmd = [compiler, test_c_file, "-o", test_exe_file, "-fopenmp"]
        try:
            subprocess.check_call(compile_cmd)
            
            # Run the compiled test program to verify OpenMP
            result = subprocess.check_output(f"./{test_exe_file}" if platform.system() != "Windows" else test_exe_file, shell=True)
            
            # Clean up the test files
            os.remove(test_c_file)
            os.remove(test_exe_file)
            
            if b"OpenMP test" in result:
                print(f"OpenMP is supported. {result.decode('utf-8').strip()}")
                return True
            else:
                print("OpenMP not supported or test program failed to run.")
                return False
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"OpenMP check failed: {e}")
            os.remove(test_c_file)
            if os.path.exists(test_exe_file):
                os.remove(test_exe_file)
            return False
    

    def execute_command(self, cmd, description):
        print(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            raise CompileError(f"Error {description}: {e}")

# Main setup function
setup(
    name="sparse_matrix_mult",
    version="0.1",
    packages=find_packages(include=['sparse_matrix_mult', 'sparse_matrix_mult.*']),
    include_package_data=True,
    package_data={
    'sparse_matrix_mult': ['lib/*.*'],  # This will include .so, .dylib, and .pyd files
    '': ['include/*.h'],
    },
    install_requires=[
        'numpy',
        'scipy',
    ],
    extras_require={
        'test': ['pytest'],
    },
    ext_modules=[Extension("sparse_matrix_mult", sources=[])],
    cmdclass={'build_ext': BuildSharedLibrary},
)
