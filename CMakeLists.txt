cmake_minimum_required(VERSION 3.19)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_config")

include(NVSHMEMEnv)
include(CMakePackageConfigHelpers)

enable_testing()

if (NVSHMEM_DEBUG)
	message(STATUS "Setting build type to Debug as requested by the environment.")
  set(CMAKE_BUILD_TYPE "debug" CACHE STRING "Choose the type of build." FORCE)
endif()

set(default_build_type "release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
endif()

# Set the possible values of build type
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "debug" "release"
             "MinSizeRel" "RelWithDebInfo")

## CUDA Selection Start

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

project(
  NVSHMEM
  LANGUAGES CUDA CXX C
  VERSION 3.2.5.1
)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set_property(CACHE CMAKE_INSTALL_PREFIX PROPERTY VALUE "${NVSHMEM_PREFIX}")
endif()

find_package(CUDAToolkit)

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

## CUDA Selection End

## Find Packages Start

if(NVSHMEM_BUILD_BITCODE_LIBRARY)
  if(NVSHMEM_CLANG_DIR)
    find_package(Clang CONFIG PATHS ${NVSHMEM_CLANG_DIR} NO_DEFAULT_PATH REQUIRED)
  else()
    find_package(Clang CONFIG REQUIRED)
  endif()
endif()

## Find Packages End

# set these flags to add a RUNPATH entry to the libraries that points at the lib dir in a portable way.
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$\{ORIGIN\}")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--enable-new-dtags")

cmake_policy(SET CMP0105 NEW)
set(CMAKE_CUDA_DEVICE_COMPILER_WRAPPER_FLAG "")

set(INCLUDE_INSTALL_DIR include/)
set(LIB_INSTALL_DIR lib/)
set(BIN_INSTALL_DIR bin/)

configure_package_config_file(
  ${CMAKE_SOURCE_DIR}/cmake_config/NVSHMEMConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/NVSHMEMConfig.cmake
  INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/nvshmem
  PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR BIN_INSTALL_DIR)

write_basic_package_version_file(
  NVSHMEMVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

add_subdirectory(src)
