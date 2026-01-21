cmake_minimum_required(VERSION 3.18)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake_config")

include(NVSHMEMEnv)
include(CMakePackageConfigHelpers)
# Allow users to set the CUDA toolkit through the env.
if(NOT CUDAToolkit_Root AND NOT CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA_HOME: ${CUDA_HOME}")
  set(CUDAToolkit_Root ${CUDA_HOME} CACHE PATH "Root of Cuda Toolkit." FORCE)
  set(CMAKE_CUDA_COMPILER "${CUDA_HOME}/bin/nvcc" CACHE PATH "Root of Cuda Toolkit." FORCE)
endif()

# Save value of CMAKE_CUDA_ARCHITECTURES before calling project()
# This is to solve a chicken-egg problem. It seems FIND_CUDA doesn't
# work without project(), but project() overwrites CMAKE_CUDA_ARCHITECTURES.
# We need to know the version of the CUDA Toolkit before we can set the project
# defaults. So this lets us know beforehand if the user has specified a preference.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES_UNDEFINED 1)
endif()

if (NOT DEFINED CUDA_ARCHITECTURES)
  set(CUDA_ARCHITECTURES_UNDEFINED 1)
endif()

project(P2P_Examples LANGUAGES CXX CUDA)


# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)
find_library(CUDA_LIBRARY NAMES cuda REQUIRED)

# C++/CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# GPU architecture (default: sm_90, override with -DCUDA_ARCH=sm_XX)
# if(NOT DEFINED CUDA_ARCH)
#     set(CUDA_ARCH "sm_90" CACHE STRING "CUDA GPU architecture")
# endif()
# string(REGEX REPLACE "sm_([0-9]+)" "\\1" CUDA_ARCH_NUM "${CUDA_ARCH}")
# set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH_NUM}" CACHE STRING "CUDA architectures" FORCE)

if(DEFINED CMAKE_CUDA_ARCHITECTURES_UNDEFINED)
  if(NOT DEFINED CUDA_ARCHITECTURES_UNDEFINED)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES} CACHE STRING "CUDA ARCHITECTURES" FORCE)
  else()
    if(CUDAToolkit_VERSION_MAJOR LESS 11)
      set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "CUDA ARCHITECTURES" FORCE)
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL 11 AND CUDAToolkit_VERSION_MINOR LESS 8)
      set(CMAKE_CUDA_ARCHITECTURES "70-real;80" CACHE STRING "CUDA ARCHITECTURES" FORCE)
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL 11 OR (CUDAToolkit_VERSION_MAJOR EQUAL 12 AND CUDAToolkit_VERSION_MINOR LESS 8))
      set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;90" CACHE STRING "CUDA ARCHITECTURES" FORCE)
    else()
      set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;90-real;100" CACHE STRING "CUDA ARCHITECTURES" FORCE)
    endif()
  endif()
endif()

message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# Include and library directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../build/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../build/src/include/host
    ${CMAKE_CURRENT_SOURCE_DIR}/../build/src/include/host/env
    ${CUDAToolkit_INCLUDE_DIRS}
)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build/src/lib)

# Build p2p_multi executable
add_executable(p2p_multi p2p_multi.cu)
target_link_libraries(p2p_multi PRIVATE nvshmem cuda)
set_target_properties(p2p_multi PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_definitions(p2p_multi PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:__STDC_LIMIT_MACROS;__STDC_CONSTANT_MACROS>
)
target_compile_options(p2p_multi PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-g -O0 -rdc=true>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra>
)

# Build p2p_one_to_one executable
add_executable(p2p_one_to_one p2p_one_to_one.cu)
target_link_libraries(p2p_one_to_one PRIVATE nvshmem cuda)
set_target_properties(p2p_one_to_one PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_definitions(p2p_one_to_one PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:__STDC_LIMIT_MACROS;__STDC_CONSTANT_MACROS>
)
target_compile_options(p2p_one_to_one PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-g -O0 -rdc=true>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra>
)

# Print configuration summary
message(STATUS "P2P Examples Configuration:")
message(STATUS "  CUDA Architecture: ${CUDA_ARCH} (${CMAKE_CUDA_ARCHITECTURES})")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "To change GPU architecture: cmake -DCUDA_ARCH=sm_XX ..")