using BinaryBuilder, Pkg

name = "CasADi_cxxwrap"

version = v"3.7.2"

include("../../L/libjulia/common.jl")

sources = [
    DirectorySource("./bundled"),
]

script = raw"""
cd $WORKSPACE/srcdir/bundled
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
filter!(
    p -> arch(p) != "riscv64" &&
        !(arch(p) == "aarch64" && Sys.isfreebsd(p)),
    platforms,
)

dependencies = [
    BuildDependency(PackageSpec(; name="libjulia_jll", version="1.11.0")),
    Dependency("CasADi_jll"; compat="=3.7.2"),
    Dependency("libcxxwrap_julia_jll"; compat="~0.14.5"),
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
    preferred_gcc_version = v"8",
    julia_compat = libjulia_julia_compat(julia_versions),
)
