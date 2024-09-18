import os
import platform
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.errors import CompileError, DistutilsError

# Function to get the shared library extension based on the operating system (e.g., .so, .dylib, .dll)
def get_lib_extension():
    if platform.system() == "Darwin":  # macOS uses .dylib for shared libraries
        return "dylib"
    elif platform.system() == "Windows":  # Windows uses .dll
        return "dll"
    else:  # Default to .so for Linux and other Unix-like systems
        return "so"

# Check if MinGW is available on Windows
def is_mingw_available():
    try:
        # Test if 'mingw32-make' is accessible from the system's PATH
        subprocess.check_output(['mingw32-make', '--version'], stderr=subprocess.STDOUT)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# Provide instructions for MinGW installation on Windows
def mingw_install_instructions():
    """Prints detailed instructions to install MinGW and configure environment variables for Windows users."""
    print("\nMinGW (Minimalist GNU for Windows) is required for compilation on Windows.")
    print("Please download and install MinGW from the following URL:")
    print("https://www.mingw-w64.org/downloads/")
    print("\nAfter installation, add the MinGW `bin` directory to your PATH environment variable.")
    print("Example path (may vary depending on installation directory):")
    print("C:\\MinGW\\bin")
    print("\nTo add MinGW to your PATH environment variable:")
    print("1. Right-click on 'This PC' or 'My Computer' and select 'Properties'.")
    print("2. Click 'Advanced system settings' and then click 'Environment Variables'.")
    print("3. Under 'System variables', find the 'Path' variable, select it, and click 'Edit'.")
    print("4. Add the path to the MinGW `bin` directory (e.g., C:\\MinGW\\bin).")
    print("5. Click 'OK' to save and close all windows.")
    print("\nAfter setting the PATH, restart your terminal or command prompt and try the installation again.\n")

# Custom build class to compile shared libraries
class BuildSharedLibrary(build_ext):
    def run(self):
        system = platform.system()
        if system == "Darwin":  # For macOS, get supported architectures
            architectures = self.get_macos_architectures()
        elif system == "Windows":  # For Windows, get the appropriate architecture (x86 or x86_64)
            architectures = self.get_windows_architectures()
        else:  # For Linux, detect the architecture (e.g., x86_64, ARM)
            architectures = self.get_linux_architectures()
        
        # Attempt to compile for each architecture
        for arch in architectures:
            try:
                self.try_compile_shared_lib(arch)
            except CompileError as e:
                # If compilation fails, provide instructions for using the platform-specific Makefile
                print(f"Error: Compilation failed on {system} for architecture {arch}.")
                print(f"Details: {str(e)}")
                print("Installation failed due to missing dependencies or compilation issues.")
                print("Please consider using the platform-specific Makefile for manual installation:")
                if system == "Windows":
                    print("- For Windows, use the Makefile in the root directory with MinGW called as Makefile.windows_mingw.")
                elif system == "Darwin":
                    print("- For macOS, use the Makefile in the root directory called Makefile.mac.")
                else:
                    print("- For Linux, use the Makefile in the root directory called Makefile.linux.")
                # If multiple architectures are supported, continue with the next one
                if len(architectures) > 1:
                    print(f"Continuing with next architecture...")
                else:
                    raise DistutilsError("Failed to build the shared library.")

    # Helper function to get supported architectures for macOS (e.g., arm64, x86_64)
    def get_macos_architectures(self):
        architectures = []
        if subprocess.call(["sysctl", "-n", "hw.optional.arm64"]) == 0:  # Check if arm64 is supported
            architectures.append('arm64')
        architectures.append('x86_64')  # Always include x86_64 as fallback
        return architectures

    # Helper function to determine the architecture for Windows (either x86 or x86_64)
    def get_windows_architectures(self):
        if platform.machine().endswith('64'):
            return ['x86_64']
        else:
            return ['x86']

    # Helper function to determine architecture for Linux (e.g., x86_64, arm64)
    def get_linux_architectures(self):
        machine = platform.machine()
        if machine == 'x86_64':
            return ['x86_64']
        elif machine.startswith('arm') or machine.startswith('aarch64'):
            return ['arm64']
        else:
            return [machine]

    # Function to compile shared library for the specified architecture
    def try_compile_shared_lib(self, arch):
        print(f"Attempting to compile the shared library for {platform.system()} with architecture {arch}...")

        lib_extension = get_lib_extension()  # Get the correct file extension for shared libraries
        system = platform.system()

        # On Windows, check for MinGW before proceeding with compilation
        if system == "Windows" and not is_mingw_available():
            mingw_install_instructions()  # Provide MinGW installation instructions
            raise CompileError("MinGW is not available. Please install MinGW for Windows.")

        compiler = self.get_compiler_command()  # Determine the appropriate compiler
        source_dir = os.path.join(os.path.dirname(__file__), "src")
        include_dir = os.path.join(os.path.dirname(__file__), "include")
        output_lib = f"libsparse_{arch}.{lib_extension}"  # Set output library file name
        output_path = os.path.join("sparse_matrix_mult", "lib", output_lib)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output directories exist

        openmp_available = self.check_openmp_support(compiler)  # Check if OpenMP is available

        compile_cmd = self.get_compile_command(compiler, include_dir, output_path, source_dir, arch, openmp_available)

        # Execute the compilation command and handle errors
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

    # Get the appropriate compiler based on the platform
    def get_compiler_command(self):
        if platform.system() == "Darwin":
            return "clang++"  # Use clang++ for macOS
        elif platform.system() == "Windows":
            return "g++"  # Use g++ (MinGW) for Windows
        else:
            return "g++"  # Use g++ for Linux

        # Construct the full compilation command
    def get_compile_command(self, compiler, include_dir, output_path, source_dir, arch, openmp_available):
        # List of source files for compilation
        source_files = [
            os.path.join(source_dir, "memfunctions.cpp"),
            os.path.join(source_dir, "sparse_sparse_dense.cpp"),
            os.path.join(source_dir, "sparse_sparse_sparse.cpp"),
            os.path.join(source_dir, "sparsework.cpp"),
            os.path.join(source_dir, "workdivision.cpp"),
        ]

        omp_include, omp_lib = self.get_openmp_paths(arch)  # Get OpenMP include/lib paths if available

        # Common compilation flags
        common_flags = [
            f"-I{include_dir}",
            "-Wall", "-O3", "-fPIC",
            "-shared",
            "-o", output_path,
        ]

        # Add OpenMP flags if available
        if openmp_available:
            common_flags.extend([
                f"-I{omp_include}",
                "-Xpreprocessor", "-fopenmp",
                f"-L{omp_lib}",
                "-lomp",
            ])

        # Handle macOS-specific flags
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


     # Get the appropriate OpenMP paths for the architecture
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
            return "", ""  # OpenMP is typically included with MinGW
        else:
            return "/usr/include", "/usr/lib"


    # Check if the current compiler supports OpenMP
    def check_openmp_support(self, compiler):
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

# Main setup function that defines the package configuration
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
