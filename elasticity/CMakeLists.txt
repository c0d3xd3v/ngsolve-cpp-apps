cmake_minimum_required(VERSION 3.1)

project(ngsolve-apps)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(NGSolve CONFIG REQUIRED)
find_package(MPI REQUIRED)

add_executable(ngs_arnoldi ngs_arnoldi.cpp)
target_link_libraries(ngs_arnoldi
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
