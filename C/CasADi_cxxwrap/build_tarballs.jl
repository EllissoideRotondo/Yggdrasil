using BinaryBuilder, Pkg

name = "CasADi_cxxwrap"

version = v"0.1.0"

include("../../L/libjulia/common.jl")

sources = [
    DirectorySource("./bundled"),
]

script = raw"""
cd $WORKSPACE/srcdir
install_license LICENSE

cmake -B build -S . \
    -DCMAKE_INSTALL_PREFIX=${prefix} \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TARGET_TOOLCHAIN} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=${prefix} \
    -DJlCxx_DIR=${prefix}/lib/cmake/JlCxx \
    -DJulia_PREFIX=${prefix} \
    -DCASADI_ROOT=${prefix}

cmake --build build --parallel ${nproc}
cmake --install build
"""

products = [
    LibraryProduct("libcasadicxxwrap", :libcasadicxxwrap),
]

platforms = vcat(libjulia_platforms.(julia_versions)...)
platforms = expand_cxxstring_abis(platforms)
# riscv64: libcxxwrap_julia_jll has no riscv64 support
# FreeBSD: CasADi_jll excludes FreeBSD (Bonmin_jll incompatibility)
filter!(
    p -> arch(p) != "riscv64" && !Sys.isfreebsd(p),
    platforms,
)

dependencies = [
    BuildDependency("libjulia_jll"),
    Dependency("CasADi_jll"; compat="~3.7.3"),
    Dependency("libcxxwrap_julia_jll"; compat="~0.14.5"),
    Dependency("CompilerSupportLibraries_jll"),
]

build_tarballs(
    ARGS,
    name,
    version,
    sources,
    script,
    platforms,
    products,
    dependencies;
    preferred_gcc_version = v"9",
    julia_compat = libjulia_julia_compat(julia_versions),
)
