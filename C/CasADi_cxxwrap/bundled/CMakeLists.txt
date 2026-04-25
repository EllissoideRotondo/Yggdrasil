cmake_minimum_required(VERSION 3.16)

project(CasADiCxxWrap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(JlCxx REQUIRED)

set(Julia_PREFIX "" CACHE PATH "Julia installation prefix")
set(CASADI_ROOT "" CACHE PATH "CasADi installation prefix")
if(NOT Julia_PREFIX)
  set(Julia_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

find_package(casadi CONFIG QUIET)
find_package(CasADi CONFIG QUIET)

set(CASADI_TARGET "")
if(TARGET casadi::casadi)
  set(CASADI_TARGET casadi::casadi)
elseif(TARGET CasADi::casadi)
  set(CASADI_TARGET CasADi::casadi)
elseif(TARGET CasADi::CasADi)
  set(CASADI_TARGET CasADi::CasADi)
elseif(TARGET casadi)
  set(CASADI_TARGET casadi)
endif()

if(NOT CASADI_TARGET)
  find_path(
    CASADI_INCLUDE_DIR
    NAMES casadi/casadi.hpp
    HINTS "${CASADI_ROOT}" ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES include
  )

  find_library(
    CASADI_LIBRARY
    NAMES casadi libcasadi
    HINTS "${CASADI_ROOT}" ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib bin
  )

  if(NOT CASADI_INCLUDE_DIR OR NOT CASADI_LIBRARY)
    message(FATAL_ERROR "Could not find CasADi. Set CASADI_ROOT or CMAKE_PREFIX_PATH to the CasADi prefix.")
  endif()
endif()

set(CASADI_CXXWRAP_SOURCES
  callback.cpp
  casadi_cxxwrap.cpp
  codegen.cpp
  common.cpp
  factories.cpp
  functions.cpp
  matrix.cpp
  options.cpp
  sparsity.cpp
)

add_library(casadicxxwrap SHARED ${CASADI_CXXWRAP_SOURCES})

target_compile_features(casadicxxwrap PRIVATE cxx_std_17)
target_include_directories(casadicxxwrap PRIVATE "${Julia_PREFIX}/include/julia")

if(CASADI_TARGET)
  target_link_libraries(casadicxxwrap PRIVATE JlCxx::cxxwrap_julia JlCxx::cxxwrap_julia_stl "${CASADI_TARGET}")
else()
  target_include_directories(casadicxxwrap PRIVATE "${CASADI_INCLUDE_DIR}")
  target_link_libraries(casadicxxwrap PRIVATE JlCxx::cxxwrap_julia JlCxx::cxxwrap_julia_stl "${CASADI_LIBRARY}")
endif()

if(MSVC)
  target_compile_options(casadicxxwrap PRIVATE /permissive- /EHsc)
  set_target_properties(casadicxxwrap PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  target_compile_options(casadicxxwrap PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS casadicxxwrap
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
)
