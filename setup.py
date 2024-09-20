# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 20:43:27 2024

@author: yadav
"""
import os
import platform
import subprocess
import tempfile
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
            # Check if OpenMP is supported on Windows
            openmp_supported = self.check_openmp_support_windows("g++")
            self.build_windows()
        else:
            architectures = self.get_macos_architectures() if system == "Darwin" else self.get_linux_architectures()
            openmp_supported = self.check_openmp_support_unix("g++")
            self.build_unix(system, architectures)

        # Print OpenMP support status at the end of the build process based on actual support
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
    		
    		# Compiler settings for MinGW
    		cxx = "g++"
    		cxxflags = ["-Wall", "-O3", "-std=c++11", "-fopenmp", "-DSPARSE_LIB_EXPORTS"]
    		ldflags = ["-shared"]
    		includes = ["-I./include"]
    		libs = ["-lgomp"]
    		
    		# Directories
    		src_dir = "src"
    		lib_dir = os.path.join("sparse_matrix_mult", "lib")
    		
    		# Output library (use .dll for Windows shared library)
    		library = os.path.join(lib_dir, "libsparse_x64.dll")
    		
    		# Source files
    		sources = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith('.cpp')]
    		
    		# Create lib directory
    		os.makedirs(lib_dir, exist_ok=True)
    		
    		# Compile and link in one step
    		cmd = [cxx] + cxxflags + includes + ldflags + ["-o", library] + sources + libs
    		self.execute_command(cmd, "compiling and linking library")
    		
    		if os.path.exists(library):
    			print(f"Compilation succeeded with OpenMP support. Output: {library}")
    		else:
    			print("Compilation failed.")


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
        if self.check_openmp_support_unix(cxx):
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

    def check_openmp_support_unix(self, compiler):
        try:
            with open(os.devnull, 'w') as devnull:
                subprocess.check_call([compiler, "-fopenmp", "-E", "-", "-o", "/dev/null"], 
                                      stdin=subprocess.DEVNULL, stdout=devnull, stderr=devnull)
            return True
        except subprocess.CalledProcessError:
            return False

    def check_openmp_support_windows(self, compiler):
        omp_test_code = """
        #include <omp.h>
        #include <stdio.h>
        int main() {
            #pragma omp parallel
            { printf("Thread %d\\n", omp_get_thread_num()); }
            return 0;
        }
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as tmp_file:
            tmp_file.write(omp_test_code)
            tmp_file.flush()
            file_name = tmp_file.name

        try:
            # Compile
            subprocess.check_call([compiler, "-fopenmp", file_name, "-o", "omp_test.exe"], 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Run
            output = subprocess.check_output(["omp_test.exe"], universal_newlines=True)
            return "Thread" in output
        except subprocess.CalledProcessError:
            return False
        finally:
            os.remove(file_name)
            if os.path.exists("omp_test.exe"):
                os.remove("omp_test.exe")

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
        'sparse_matrix_mult': ['lib/*.*'],
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
