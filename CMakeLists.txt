cmake_minimum_required(VERSION 3.18)
project(APD LANGUAGES CXX CUDA)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(CUDAToolkit REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} .)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-std=c++11 -pthread -Wall -Wextra -Wno-unused-function -Wno-switch)
endif()

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++11 -Xcompiler=-Wall -gencode arch=compute_120,code=sm_120")

add_executable(APD
    main.h
    APD.h
    APD.cpp
    APD.cu
    main.cpp
    Checkpoint.h
    Checkpoint.cpp
)

target_link_libraries(APD
    PRIVATE
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        CUDA::cudart
)
