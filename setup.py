import os
import platform
import subprocess
import sys
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.errors import CompileError, DistutilsError

def get_lib_extension():
    if platform.system() == "Darwin":
        return "dylib"
    elif platform.system() == "Windows":
        return "dll"
    else:
        return "so"

class BuildSharedLibrary(build_ext):
    def run(self):
        if platform.system() == "Darwin":
            architectures = self.get_macos_architectures()
        else:
            architectures = [platform.machine()]
        
        for arch in architectures:
            try:
                self.try_compile_shared_lib(arch)
            except CompileError as e:
                print(f"Error: Compilation failed on {platform.system()} for architecture {arch}.")
                print(f"Details: {str(e)}")
                if len(architectures) > 1:
                    print(f"Continuing with next architecture...")
                else:
                    raise DistutilsError("Failed to build the shared library.")

    def get_macos_architectures(self):
        architectures = []
        # Check if arm64 is supported
        if subprocess.call(["sysctl", "-n", "hw.optional.arm64"]) == 0:
            architectures.append('arm64')
        # x86_64 is always added as a fallback
        architectures.append('x86_64')
        return architectures

    def try_compile_shared_lib(self, arch):
        print(f"Attempting to compile the shared library for {platform.system()} with architecture {arch}...")

        lib_extension = get_lib_extension()
        system = platform.system()

        compiler = self.get_compiler_command()

        source_dir = os.path.join(os.path.dirname(__file__), "src")
        include_dir = os.path.join(os.path.dirname(__file__), "include")
        output_lib = f"libsparse_{arch}.{lib_extension}"
        output_path = os.path.join("sparse_matrix_mult", "lib", output_lib)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Check if OpenMP is available before adding OpenMP flags
        openmp_available = self.check_openmp_support(compiler)

        compile_cmd = self.get_compile_command(compiler, include_dir, output_path, source_dir, arch, openmp_available)

        try:
            print("Compilation command:", ' '.join(compile_cmd))
            if openmp_available:
                print("Compiling with OpenMP support...")
            else:
                print("Compiling without OpenMP support...")
            result = subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
            print(f"Successfully compiled shared library: {output_path}")
            print("Compiler output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Failed to compile shared library. Error code: {e.returncode}")
            print("Compiler stderr:", e.stderr)
            print("Compiler stdout:", e.stdout)
            raise CompileError(f"Failed to compile shared library: {e}")

    def get_compiler_command(self):
        if platform.system() == "Darwin":
            return "clang++"
        elif platform.system() == "Windows":
            return "cl"
        else:
            return "g++"

    def get_compile_command(self, compiler, include_dir, output_path, source_dir, arch, openmp_available):
        # Use corrected filenames
        source_files = [
            os.path.join(source_dir, "memfunctions.cpp"),
            os.path.join(source_dir, "sparse_sparse_dense.cpp"),  # Renamed
            os.path.join(source_dir, "sparse_sparse_sparse.cpp"),  # Renamed
            os.path.join(source_dir, "sparsework.cpp"),
            os.path.join(source_dir, "workdivision.cpp"),
        ]

        omp_include, omp_lib = self.get_openmp_paths(arch)

        # If OpenMP is available, include its flags; otherwise, compile without OpenMP
        common_flags = [
            f"-I{include_dir}",
            "-Wall", "-O3", "-fPIC",
            "-shared",
            "-o", output_path,
        ]

        if openmp_available:
            common_flags.extend([
                f"-I{omp_include}",
                "-Xpreprocessor", "-fopenmp",
                f"-L{omp_lib}",
                "-lomp",
            ])

        if platform.system() == "Darwin":
            compile_cmd = [
                compiler,
                *source_files,
                f"-arch", arch,
                *common_flags,
            ]
            if arch == 'arm64':
                compile_cmd.extend(["-target", "arm64-apple-macos11"])
            elif arch == 'x86_64':
                compile_cmd.extend(["-target", "x86_64-apple-macos10.12"])
        else:
            compile_cmd = [
                compiler,
                *source_files,
                *common_flags,
            ]

        return compile_cmd

    def get_openmp_paths(self, arch):
        system = platform.system()
        if system == "Darwin":
            if arch == 'arm64':
                try:
                    brew_prefix = subprocess.getoutput("brew --prefix libomp")
                    return f"{brew_prefix}/include", f"{brew_prefix}/lib"
                except:
                    pass
            return "/usr/local/opt/libomp/include", "/usr/local/opt/libomp/lib"
        elif system == "Windows":
            return "", ""  # OpenMP is typically included with MSVC
        else:
            return "/usr/include", "/usr/lib"

    def check_openmp_support(self, compiler):
        """Check if the compiler supports OpenMP."""
        try:
            with open(os.devnull, 'w') as devnull:
                result = subprocess.run([compiler, "-Xpreprocessor", "-fopenmp", "-o", "/dev/null", "-x", "c++", "-"],
                                        input="", stdout=devnull, stderr=devnull, text=True)
            if result.returncode == 0:
                print("OpenMP is supported.")
                return True
            else:
                print("OpenMP is not supported.")
                return False
        except subprocess.CalledProcessError:
            print("Error checking OpenMP support. Assuming it's not available.")
            return False
        except Exception as e:
            print(f"Unexpected error checking OpenMP support: {e}")
            return False

setup(
    name="sparse_matrix_mult",
    version="0.1",
    packages=find_packages(include=['sparse_matrix_mult', 'sparse_matrix_mult.*']),
    include_package_data=True,
    package_data={
        'sparse_matrix_mult': [f'lib/*.{get_lib_extension()}'],
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
