# Copyright 2013-2023 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os

from spack.package import *

def cmake_cache_filepath(name, value, comment=""):
    """Generate a string for a cmake cache variable"""
    return 'set({0} "{1}" CACHE FILEPATH "{2}")\n'.format(name, value, comment)

# This limits the versions of lots of things pretty severely.
#
#   - Only v1.5.2 and newer are buildable.
#   - CMake must be v3.22 or newer.
#   - CUDA must be v11.0.0 or newer.

class Hydrogen(CachedCMakePackage, CudaPackage, ROCmPackage):
    """Hydrogen: Distributed-memory dense and sparse-direct linear algebra
    and optimization library. Based on the Elemental library."""

    homepage = "https://libelemental.org"
    url = "https://github.com/LLNL/Elemental/archive/v1.5.1.tar.gz"
    git = "https://github.com/LLNL/Elemental.git"
    tags = ["ecp", "radiuss"]

    maintainers("bvanessen")

    version("develop", branch="hydrogen")
    version("1.5.2", sha256="a902cad3962471216cfa278ba0561c18751d415cd4d6b2417c02a43b0ab2ea33")
    version("1.5.1", sha256="447da564278f98366906d561d9c8bc4d31678c56d761679c2ff3e59ee7a2895c")
    # Older versions are no longer supported.

    variant(
        "shared",
        default=True,
        description="Enables the build of shared libraries.")
    variant(
        "build_type",
        default="Release",
        description="The build type to build",
        values=("Debug", "Release"))
    variant(
        "int64",
        default=False,
        description="Use 64-bit integers")
    variant(
        "al",
        default=False,
        description="Use Aluminum communication library")
    variant(
        "cub",
        default=True,
        when="+cuda",
        description="Use CUB/hipCUB for GPU memory management")
    variant(
        "cub",
        default=True,
        when="+rocm",
        description="Use CUB/hipCUB for GPU memory management")
    variant(
        "half",
        default=False,
        description="Support for FP16 precision data types")

    # FIXME: Add netlib-lapack. For GPU-enabled builds, typical
    # workflows don't touch host BLAS/LAPACK all that often, and even
    # less frequently in performance-critical regions.
    variant(
        "blas",
        default="any",
        values=("any", "openblas", "mkl", "accelerate", "essl", "libsci"),
        description="Specify a host BLAS library preference")
    variant(
        "int64_blas",
        default=False,
        description="Use 64-bit integers for (host) BLAS.")

    variant(
        "openmp",
        default=True,
        description="Make use of OpenMP within CPU kernels")
    variant(
        "omp_taskloops",
        when="+openmp",
        default=False,
        description="Use OpenMP taskloops instead of parallel for loops")

    # Users should spec this on their own on the command line, no?
    # This doesn't affect Hydrogen itself at all. Not one bit.
    # variant(
    #     "openmp_blas",
    #     default=False,
    #     description="Use OpenMP for threading in the BLAS library")

    variant(
        "test",
        default=False,
        description="Builds test suite")

    conflicts(
        "+cuda",
        when="+rocm",
        msg="CUDA and ROCm support are mutually exclusive")
    conflicts(
        "+half",
        when="+rocm",
        msg="FP16 support not implemented for ROCm.")

    depends_on("cmake@3.22.0:", type="build", when="@1.5.2:")

    depends_on("mpi")
    depends_on("blas")
    depends_on("lapack")

    # Note that #1712 forces us to enumerate the different blas variants
    # Note that this forces us to use OpenBLAS until #1712 is fixed
    depends_on("openblas", when="blas=openblas")
    depends_on("openblas +ilp64", when="blas=openblas +int64_blas")

    depends_on("intel-mkl", when="blas=mkl")
    depends_on("intel-mkl +ilp64", when="blas=mkl +int64_blas")

    # I don't think this is true...
    depends_on("veclibfort", when="blas=accelerate")

    depends_on("essl", when="blas=essl")
    depends_on("essl +ilp64", when="blas=essl +int64_blas")

    depends_on("netlib-lapack +external-blas", when="blas=essl")

    depends_on("cray-libsci", when="blas=libsci")

    # Specify the correct version of Aluminum
    depends_on("aluminum@0.7.0:", when="@1.5.2: +al")

    # Add Aluminum variants
    depends_on("aluminum +cuda +nccl +ht", when="+al +cuda")
    depends_on("aluminum +rocm +nccl +ht", when="+al +rocm")

    for arch in CudaPackage.cuda_arch_values:
        depends_on("aluminum cuda_arch=%s" % arch,
                   when="+al +cuda cuda_arch=%s" % arch)

    # variants +rocm and amdgpu_targets are not automatically passed to
    # dependencies, so do it manually.
    for val in ROCmPackage.amdgpu_targets:
        depends_on("aluminum amdgpu_target=%s" % val,
                   when="+al +rocm amdgpu_target=%s" % val)

    depends_on("cuda@11.0.0:", when="+cuda")
    depends_on("hipcub +rocm", when="+rocm +cub")
    depends_on("half", when="+half")

    depends_on("llvm-openmp", when="%apple-clang +openmp")

    @property
    def libs(self):
        shared = True if "+shared" in self.spec else False
        return find_libraries("libHydrogen", root=self.prefix, shared=shared, recursive=True)

    def cmake_args(self):
        spec = self.spec
        args = []
        return args

    def get_cuda_flags(self):
        spec = self.spec
        args = []
        if spec.satisfies("^cuda+allow-unsupported-compilers"):
            args.append("-allow-unsupported-compiler")

        if spec.satisfies("%clang"):
            for flag in spec.compiler_flags["cxxflags"]:
                if "gcc-toolchain" in flag:
                    args.append("-Xcompiler={0}".format(flag))
        return args

    def std_initconfig_entries(self):
        spec = self.spec
        entries = super(Hydrogen, self).std_initconfig_entries()

        # CMAKE_PREFIX_PATH, in CMake types, is a "STRING", not a "PATH". :/
        entries = [ x for x in entries if "CMAKE_PREFIX_PATH" not in x ]
        cmake_prefix_path = os.environ["CMAKE_PREFIX_PATH"].replace(':',';')
        entries.append(cmake_cache_string("CMAKE_PREFIX_PATH", cmake_prefix_path))
        # IDK why this is here, but it was in the original recipe. So, yeah.
        entries.append(cmake_cache_string("CMAKE_INSTALL_MESSAGE", "LAZY"))
        return entries

    def initconfig_compiler_entries(self):
        spec = self.spec
        entries = super(Hydrogen, self).initconfig_compiler_entries()

        # We don't need this generator, we don't want this generator.
        # We don't specify a generator for Hydrogen BECAUSE IT DOESN'T
        # (shouldn't) MATTER in the sense that it doesn't (shouldn't)
        # impact the correctness of a build, and it should not be
        # hard-coded into the CMake cache file (especially since
        # "-G<something else>" doesn't override it). Moreover, to
        # choose to encode it in the cache file is to make assumptions
        # about how the cache file will be consumed, which creates a
        # headache for consumers outside those assumptions (an old
        # adage about what happens when one assumes comes to mind).
        entries = [ x for x in entries if "CMAKE_GENERATOR" not in x and "CMAKE_MAKE_PROGRAM" not in x ]

        # FIXME: Enforce this better in the actual CMake.
        entries.append(cmake_cache_string("CMAKE_CXX_STANDARD", "17"))
        entries.append(cmake_cache_option("BUILD_SHARED_LIBS", "+shared" in spec))
        entries.append(cmake_cache_option("CMAKE_EXPORT_COMPILE_COMMANDS", True))

        if "+rocm" in spec:
            entries.append(
                cmake_cache_filepath(
                    "CMAKE_HIP_COMPILER",
                    os.path.join(spec["llvm-amdgpu"].prefix.bin, "clang++")))

        entries.append(cmake_cache_option("MPI_ASSUME_NO_BUILTIN_MPI", True))

        if spec.satisfies("%clang +openmp platform=darwin") or spec.satisfies("%clang +omp_taskloops platform=darwin"):
            clang = self.compiler.cc
            clang_bin = os.path.dirname(clang)
            clang_root = os.path.dirname(clang_bin)
            entries.append(cmake_cache_string("OpenMP_CXX_FLAGS", "-fopenmp=libomp"))
            entries.append(cmake_cache_string("OpenMP_CXX_LIB_NAMES", "libomp"))
            entries.append(cmake_cache_string("OpenMP_libomp_LIBRARY",
                                              "{0}/lib/libomp.dylib".format(clang_root)))

        return entries

    def initconfig_hardware_entries(self):
        spec = self.spec
        entries = super(Hydrogen, self).initconfig_hardware_entries()

        entries.append(cmake_cache_option("Hydrogen_ENABLE_CUDA", "+cuda" in spec))
        if spec.satisfies("+cuda"):
            entries.append(cmake_cache_string("CMAKE_CUDA_STANDARD", "17"))
            if not spec.satisfies("cuda_arch=none"):
                archs = spec.variants["cuda_arch"].value
                arch_str = ";".join(archs)
                entries.append(cmake_cache_string("CMAKE_CUDA_ARCHITECTURES", arch_str))

            # FIXME: Should this use the "cuda_flags" function of the
            # CudaPackage class or something? There might be other
            # flags in play, and we need to be sure to get them all.
            cuda_flags = self.get_cuda_flags()
            if len(cuda_flags) > 0:
                entries.append(cmake_cache_string("CMAKE_CUDA_FLAGS",
                                                  " ".join(cuda_flags)))

        entries.append(cmake_cache_option("Hydrogen_ENABLE_ROCM", "+rocm" in spec))
        if spec.satisfies("+rocm"):
            entries.append(cmake_cache_string("CMAKE_HIP_STANDARD", "17"))
            if not spec.satisfies("amdgpu_target=none"):
                archs = self.spec.variants["amdgpu_target"].value
                arch_str = ";".join(archs)
                entries.append(cmake_cache_string("CMAKE_HIP_ARCHITECTURES", arch_str))
                entries.append(cmake_cache_string("AMDGPU_TARGETS", arch_str))
                entries.append(cmake_cache_string("GPU_TARGETS", arch_str))
            entries.append(cmake_cache_path("HIP_ROOT_DIR", spec["hip"].prefix))

        return entries

    def initconfig_package_entries(self):
        spec = self.spec
        entries = super(Hydrogen, self).initconfig_package_entries()

        # Basic Hydrogen options
        entries.append(cmake_cache_option("Hydrogen_ENABLE_TESTING", "+test" in spec))
        entries.append(cmake_cache_option("Hydrogen_GENERAL_LAPACK_FALLBACK", True))
        entries.append(cmake_cache_option("Hydrogen_USE_64BIT_INTS", "+int64" in spec))
        entries.append(cmake_cache_option("Hydrogen_USE_64BIT_BLAS_INTS", "+int64_blas" in spec))

        # Advanced dependency options
        entries.append(cmake_cache_option("Hydrogen_ENABLE_ALUMINUM", "+al" in spec))
        entries.append(cmake_cache_option("Hydrogen_ENABLE_CUB", "+cub" in spec))
        entries.append(cmake_cache_option("Hydrogen_ENABLE_GPU_FP16", "+cuda +half" in spec))
        entries.append(cmake_cache_option("Hydrogen_ENABLE_HALF", "+half" in spec))
        entries.append(cmake_cache_option("Hydrogen_ENABLE_OPENMP", "+openmp" in spec))
        entries.append(cmake_cache_option("Hydrogen_ENABLE_OMP_TASKLOOP", "+omp_taskloops" in spec))

        # Note that CUDA/ROCm are handled above.

        if "blas=openblas" in spec:
            entries.append(cmake_cache_option("Hydrogen_USE_OpenBLAS", "blas=openblas" in spec))
            # CMAKE_PREFIX_PATH should handle this
            entries.append(cmake_cache_string("OpenBLAS_DIR", spec["openblas"].prefix))
        elif "blas=mkl" in spec:
            entries.append(cmake_cache_option("Hydrogen_USE_MKL", True))
        elif "blas=accelerate" in spec:
            entries.append(cmake_cache_option("Hydrogen_USE_ACCELERATE", True))
        elif "blas=essl" in spec:
            # IF IBM ESSL is used it needs help finding the proper LAPACK libraries
            entries.append(cmake_cache_string("LAPACK_LIBRARIES", "%s;-llapack;-lblas" % ";".join("-l{0}".format(lib) for lib in self.spec["essl"].libs.names)))
            entries.append(cmake_cache_string("BLAS_LIBRARIES", "%s;-lblas" % ";".join("-l{0}".format(lib) for lib in self.spec["essl"].libs.names)))

        return entries

    def setup_build_environment(self, env):
        if self.spec.satisfies("%apple-clang +openmp"):
            env.append_flags("CPPFLAGS", self.compiler.openmp_flag)
            env.append_flags("CFLAGS", self.spec["llvm-openmp"].headers.include_flags)
            env.append_flags("CXXFLAGS", self.spec["llvm-openmp"].headers.include_flags)
            env.append_flags("LDFLAGS", self.spec["llvm-openmp"].libs.ld_flags)
