cmake_minimum_required(VERSION 3.1)

project(external)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(ExternalProject)

set(patch_command git apply --unsafe-paths --quiet --reject --allow-empty --directory=${CMAKE_CURRENT_BINARY_DIR}/ngs-petsc ${CMAKE_CURRENT_LIST_DIR}/ngs-petsc.patch && echo "patch applied" || echo "patch not applied")
#set(patch_command patch --directory=${CMAKE_CURRENT_BINARY_DIR}/ngs-petsc < ${CMAKE_CURRENT_LIST_DIR}/ngs-petsc.patch)

ExternalProject_Add(ngs-petsc
    GIT_REPOSITORY https://github.com/NGSolve/ngs-petsc.git
    SOURCE_DIR ngs-petsc
    INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/ngs-petsc
    PATCH_COMMAND ${patch_command}
    UPDATE_DISCONNECTED 1
    CMAKE_ARGS
        -DNGSolve_DIR=/home/kai/Development/libs/ngsolve-install/lib/cmake/ngsolve/
        -DPETSC_DIR=/home/kai/Development/libs/petsc-install/
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/ngs-petsc-install/
    BUILD_ALWAYS false
)
