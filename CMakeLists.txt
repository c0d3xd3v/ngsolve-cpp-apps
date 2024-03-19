cmake_minimum_required(VERSION 3.1)

project(ngs_apps)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_subdirectory(elasticity)
add_subdirectory(poission)
add_subdirectory(ngs_coefficient_function)

option(BUILD_WITH_NGSPETSC "build ngspetsc and example" OFF)
if(${BUILD_WITH_NGSPETSC})
    add_subdirectory(external)
    message(STATUS ${CMAKE_CURRENT_BINARY_DIR})
    set(NGSPETSC_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/ngs-petsc-install/ngs_petsc_complex)
    add_subdirectory(ngs_petsc.poission)
endif()
