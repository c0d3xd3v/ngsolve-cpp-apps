cmake_minimum_required(VERSION 3.1)

project(ngs_petsc_poission)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(NGSolve CONFIG REQUIRED)
find_package(MPI REQUIRED)

# PkgConfig
find_package(PkgConfig)

# PETSc
if (PKG_CONFIG_FOUND)
    pkg_check_modules(PETSC PETSc)
endif()

include_directories(${PETSC_INCLUDE_DIRS})
include_directories(${NGSPETSC_DIR}/include/)

add_executable(ngs_petsc_poission ngs_petsc_poission.cpp petsc_solve.cpp)
target_link_libraries(ngs_petsc_poission
    ngcore
    ngcomp
    ngbla
    ngla
    ngfem
    ngstd
    nglib
    nggui
    ngsolve
    -L${NGSPETSC_DIR}
    -lpetscinterface
    ${PETSC_LDFLAGS}
    --enable-new-dtags
    /usr/lib/libmpi.so
)
message(STATUS ${NGSPETSC_DIR})
