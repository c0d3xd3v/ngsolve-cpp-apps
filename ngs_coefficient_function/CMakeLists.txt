cmake_minimum_required(VERSION 3.1)

project(ngs_coefficientf)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(NGSolve CONFIG REQUIRED)
find_package(MPI REQUIRED)

add_executable(ngs_coefficientf ngs_coefficientf.cpp)
target_link_libraries(ngs_coefficientf
    ngcore
    ngcomp
    ngbla
    ngla
    ngfem
    ngstd
    nglib
    nggui
    ngsolve
    --enable-new-dtags
    /usr/lib/libmpi.so
)
