# Copyright 2013-2023 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os

import spack.build_environment
from spack.package import *

def cmake_cache_filepath(name, value, comment=""):
    """Generate a string for a cmake cache variable"""
    return 'set({0} "{1}" CACHE FILEPATH "{2}")\n'.format(name, value, comment)

class Dihydrogen(CachedCMakePackage, CudaPackage, ROCmPackage):
    """DiHydrogen is the second version of the Hydrogen fork of the
    well-known distributed linear algebra library,
    Elemental. DiHydrogen aims to be a basic distributed
    multilinear algebra interface with a particular emphasis on the
    needs of the distributed machine learning effort, LBANN."""

    homepage = "https://github.com/LLNL/DiHydrogen.git"
    url = "https://github.com/LLNL/DiHydrogen/archive/v0.1.tar.gz"
    git = "https://github.com/LLNL/DiHydrogen.git"
    tags = ["ecp", "radiuss"]

    maintainers("bvanessen")

    version("develop", branch="develop")
    version("master", branch="master")

    version("0.2.1", sha256="11e2c0f8a94ffa22e816deff0357dde6f82cc8eac21b587c800a346afb5c49ac")
    version("0.2.0", sha256="e1f597e80f93cf49a0cb2dbc079a1f348641178c49558b28438963bd4a0bdaa4")
    version("0.1", sha256="171d4b8adda1e501c38177ec966e6f11f8980bf71345e5f6d87d0a988fef4c4e")

    # Primary features

    variant(
        "distconv",
        default=False,
        description="Enable (legacy) Distributed Convolution support.")

    variant(
        "nvshmem",
        default=False,
        description="Enable support for NVSHMEM-based halo exchanges.",
        when="+distconv")

    variant(
        "shared",
        default=True,
        description="Enables the build of shared libraries")

    # Some features of developer interest

    variant(
        "developer",
        default=False,
        description="Enable extra warnings and force tests to be enabled.")

    variant(
        "ci",
        default=False,
        description="Use default options for CI builds")

    variant(
        "coverage",
        default=False,
        description="Decorate build with code coverage instrumentation options",
        when="%gcc")
    variant(
        "coverage",
        default=False,
        description="Decorate build with code coverage instrumentation options",
        when="%clang")
    variant(
        "coverage",
        default=False,
        description="Decorate build with code coverage instrumentation options",
        when="%rocmcc")

    # Package conflicts and requirements

    conflicts(
        "+nvshmem",
        when="~cuda",
        msg="NVSHMEM requires CUDA support.")

    conflicts(
        "+cuda",
        when="+rocm",
        msg="CUDA and ROCm are mutually exclusive.")

    requires(
        "+cuda", "+rocm",
        when="+distconv",
        policy="any_of",
        msg="DistConv support requires CUDA or ROCm.")

    # Dependencies

    depends_on("catch2@3.0.1:", type=("build","test"), when="+developer")
    depends_on("cmake@3.21.0:", type="build")
    depends_on("cuda@11.0:", when="+cuda")
    depends_on("spdlog", when="@:0.1,0.2:")

    with when("+distconv"):
        depends_on("mpi")

        # All this nonsense for one silly little package.
        depends_on("aluminum@0.4.0:0.4", when="@0.1")
        depends_on("aluminum@0.5.0:0.5", when="@0.2.0")
        depends_on("aluminum@0.7.0:0.7", when="@0.2.1")
        depends_on("aluminum@0.7.0:", when="@:0.0,0.2.1:")

        # Add Aluminum variants
        depends_on("aluminum +cuda +nccl +cuda_rma", when="+cuda")
        depends_on("aluminum +rocm +rccl", when="+rocm")
        depends_on("aluminum +nccl", when="+distconv +cuda")
        depends_on("aluminum +rccl", when="+distconv +rocm")

        # TODO: Debug linker errors when NVSHMEM is built with UCX
        depends_on("nvshmem +nccl~ucx", when="+nvshmem")

        # OMP support is only used in DistConv, and only Apple needs
        # hand-holding with it.
        depends_on("llvm-openmp", when="%apple-clang")
        # FIXME: when="platform=darwin"??

        # CUDA/ROCm arch forwarding

        for arch in CudaPackage.cuda_arch_values:
            depends_on(
                "aluminum cuda_arch={0}".format(arch),
                when="+cuda cuda_arch={0}".format(arch))

            # NVSHMEM also needs arch forwarding
            depends_on(
                "nvshmem cuda_arch={0}".format(arch),
                when="+nvshmem +cuda cuda_arch={0}".format(arch))

        # Idenfity versions of cuda_arch that are too old from
        # lib/spack/spack/build_systems/cuda.py. We require >=60.
        illegal_cuda_arch_values = ["10", "11", "12", "13",
                                    "20", "21", "30", "32", "35", "37",
                                    "50", "52", "53"]
        for value in illegal_cuda_arch_values:
            conflicts("cuda_arch=" + value)

        for val in ROCmPackage.amdgpu_targets:
            depends_on(
                "aluminum amdgpu_target={0}".format(val),
                when="+rocm amdgpu_target={0}".format(val))

        # CUDA-specific distconv dependencies
        depends_on("cudnn", when="+cuda")

        # ROCm-specific distconv dependencies
        depends_on("hipcub", when="+rocm")
        depends_on("miopen-hip", when="+rocm")
        depends_on("roctracer-dev", when="+rocm")

    with when("+ci+coverage"):
        depends_on("lcov", type=("build", "run"))
        depends_on("py-gcovr", type=("build", "run"))
        # Technically it's not used in the build, but CMake sets up a
        # target, so it needs to be found.

    @property
    def libs(self):
        shared = True if "+shared" in self.spec else False
        return find_libraries("libH2Core", root=self.prefix, shared=shared, recursive=True)

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
        entries = super(Dihydrogen, self).std_initconfig_entries()

        # It's a "STRING", not a "PATH". :/
        entries = [ x for x in entries if "CMAKE_PREFIX_PATH" not in x ]

        cmake_prefix_path = os.environ["CMAKE_PREFIX_PATH"].replace(':',';')
        entries.append(cmake_cache_string("CMAKE_PREFIX_PATH", cmake_prefix_path))
        if "PYTHONPATH" in os.environ:
            entries.append("# PYTHONPATH={0}".format(os.environ["PYTHONPATH"]))

        return entries

    def initconfig_compiler_entries(self):
        spec = self.spec
        entries = super(Dihydrogen, self).initconfig_compiler_entries()

        # Ugh. We don't need their generator. We don't specify a
        # generator for DiHydrogen BECAUSE IT DOESN'T (shouldn't)
        # MATTER in the sense that it should not impact the
        # correctness of a build. Moreover, encoding it in the cache
        # file is a terrible life choice because it makes assumptions
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

        if "platform=cray" in spec:
            entries.append(cmake_cache_option("MPI_ASSUME_NO_BUILTIN_MPI", True))

        if spec.satisfies("%clang +distconv platform=darwin"):
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
        entries = super(Dihydrogen, self).initconfig_hardware_entries()

        entries.append(cmake_cache_option("H2_ENABLE_CUDA", "+cuda" in spec))
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

        enable_rocm_var="H2_ENABLE_ROCM" if spec.version < Version("0.3") else "H2_ENABLE_HIP_ROCM"
        entries.append(cmake_cache_option(enable_rocm_var, "+rocm" in spec))
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
        entries = super(Dihydrogen, self).initconfig_package_entries()

        entries.append(cmake_cache_option("H2_DEVELOPER_BUILD", "+developer" in spec))
        entries.append(cmake_cache_option("H2_ENABLE_TESTS", "+developer" in spec))
        if "+developer" in spec:
            entries.append(cmake_cache_path("Catch2_ROOT", spec["catch2"].prefix))

        entries.append(cmake_cache_option("H2_ENABLE_CODE_COVERAGE", "+coverage" in spec))
        entries.append(cmake_cache_option("H2_CI_BUILD", "+ci" in spec))

        entries.append(cmake_cache_path("spdlog_ROOT", spec["spdlog"].prefix))

        if "+coverage" in spec:
            entries.append(cmake_cache_path("lcov_ROOT", spec["lcov"].prefix))
            entries.append(cmake_cache_path("genhtml_ROOT", spec["lcov"].prefix))
            if "+ci" in spec:
                entries.append(cmake_cache_path("gcovr_ROOT", spec["py-gcovr"].prefix))

        # DistConv options
        entries.append(cmake_cache_option("H2_ENABLE_ALUMINUM", "+distconv" in spec))
        entries.append(cmake_cache_option("H2_ENABLE_DISTCONV_LEGACY", "+distconv" in spec))
        entries.append(cmake_cache_option("H2_ENABLE_OPENMP", "+distconv" in spec))
        if "+distconv" in spec:
            entries.append(cmake_cache_path("Aluminum_ROOT", spec["aluminum"].prefix))
            if "+cuda" in spec:
                entries.append(cmake_cache_path("cuDNN_ROOT", spec["cudnn"].prefix))

        return entries

    def setup_build_environment(self, env):
        if self.spec.satisfies("%apple-clang +openmp"):
            env.append_flags("CPPFLAGS", self.compiler.openmp_flag)
            env.append_flags("CFLAGS", self.spec["llvm-openmp"].headers.include_flags)
            env.append_flags("CXXFLAGS", self.spec["llvm-openmp"].headers.include_flags)
            env.append_flags("LDFLAGS", self.spec["llvm-openmp"].libs.ld_flags)
