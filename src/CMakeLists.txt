# this requires cmake 3.19
include(CheckCompilerFlag)
include(CheckSourceCompiles)
include(ExternalProject)

check_compiler_flag(CUDA -t4 NVCC_THREADS)

if(NOT CMAKE_CUDA_ARCHITECTURES_UNDEFINED)
    list(APPEND my_cmake_cache_args
        -DCMAKE_CUDA_ARCHITECTURES:STRING=${CMAKE_CUDA_ARCHITECTURES}
    )
endif()

# Start nvshmem lib definitions
add_library(
  nvshmem STATIC
)

add_library(
  nvshmem_host SHARED
)

## Start generic variable configuration
set_target_properties(nvshmem nvshmem_host
                      PROPERTIES POSITION_INDEPENDENT_CODE ON
                      CXX_STANDARD_REQUIRED ON
                      CUDA_STANDARD_REQUIRED ON
                      CXX_STANDARD 11
                      CUDA_STANDARD 11
                      CUDA_SEPARABLE_COMPILATION ON
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
                      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
		      VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
		      SOVERSION ${PROJECT_VERSION_MAJOR}
)

macro(nvshmem_library_set_base_config LIBNAME)
  target_compile_definitions(${LIBNAME}
    PRIVATE $<$<CONFIG:Debug>:_NVSHMEM_DEBUG;NVSHMEM_IBGDA_DEBUG>
    $<IF:$<STREQUAL:"${CMAKE_HOST_SYSTEM_PROCESSOR}","x86_64">,NVSHMEM_X86_64,>
    $<IF:$<STREQUAL:"${CMAKE_HOST_SYSTEM_PROCESSOR}","ppc64le">,__STDC_LIMIT_MACROS;__STDC_CONSTANT_MACROS;NVSHMEM_PPC64LE,>
    $<IF:$<STREQUAL:"${CMAKE_HOST_SYSTEM_PROCESSOR}","aarch64">,NVSHMEM_AARCH64,>
    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:__STDC_LIMIT_MACROS;__STDC_CONSTANT_MACROS>
  )

  target_compile_options(${LIBNAME}
    INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVSHMEM_VERBOSE}>>:-Xptxas -v>
    PRIVATE $<IF:$<CONFIG:Debug>,-O0;-g;,-O3>
    $<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
    $<IF:$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>,-O0;-g;-G;,-O3>
    $<IF:$<STREQUAL:${CMAKE_HOST_SYSTEM_PROCESSOR},"x86_64">,-msse,>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
  )

  if(NVSHMEM_DEVEL)
    target_compile_options(
      ${LIBNAME}
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Werror
              all-warnings>
              $<$<COMPILE_LANGUAGE:CXX>:-Werror
              -Wall
              -Wextra
              -Wno-unused-function
              -Wno-unused-parameter
              -Wno-missing-field-initializers>
    )
  endif()
endmacro()

set(NVSHMEM_HOST_SOURCES
    host/transport/transport.cpp
    host/init/cudawrap.cpp
    p2pcomm.cpp
    host/util/env_vars.cpp
    host/util/debug.cpp
    host/init/init.cu
    host/util/util.cpp
)

target_include_directories(
  nvshmem
  PRIVATE include
          include/host/env
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:include>
)

target_include_directories(
  nvshmem_host
  PRIVATE include
          include/host/env
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:include>
)

target_link_libraries(nvshmem PRIVATE CUDA::cudart_static)
target_link_libraries(nvshmem_host PRIVATE CUDA::cudart_static)
nvshmem_library_set_base_config(nvshmem)
nvshmem_library_set_base_config(nvshmem_host)
## End generic variable configuration

## Start transports
set(TRANSPORT_VERSION_MAJOR 3)
set(TRANSPORT_VERSION_MINOR 0)
set(TRANSPORT_VERSION_PATCH 0)

check_source_compiles(C
  "#include <infiniband/verbs.h>
  int main(void) { int x = IBV_ACCESS_RELAXED_ORDERING; return 1; }"
  HAVE_IBV_ACCESS_RELAXED_ORDERING
)

set(ACTIVE_TRANSPORTS)

if(NVSHMEM_IBGDA_SUPPORT)
  find_library(MLX5_lib NAMES mlx5)
endif()

if(NVSHMEM_IBGDA_SUPPORT)
  check_source_compiles(C
    "#include <infiniband/mlx5dv.h>
    int main(void) { int x = MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED; return 1; }"
    HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED
  )
  check_source_compiles(C
    "#include <infiniband/mlx5dv.h>
    int main(void) { int x = MLX5DV_UMEM_MASK_DMABUF; return 1; }"
    HAVE_MLX5DV_UMEM_MASK_DMABUF
  )
  add_library(
    nvshmem_transport_ibgda SHARED
  )
  nvshmem_library_set_base_config(nvshmem_transport_ibgda)
  target_sources(nvshmem_transport_ibgda PRIVATE
                 modules/transport/ibgda/ibgda.cpp
                 modules/transport/common/transport_common.cpp
                 modules/transport/common/transport_ib_common.cpp
                 modules/transport/common/transport_mlx5_common.cpp
  )
  target_link_options(nvshmem_transport_ibgda PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/nvshmem_transport.sym")
  target_include_directories(nvshmem_transport_ibgda
                             PRIVATE include
                             modules/transport/common

  )
  target_link_libraries(nvshmem_transport_ibgda PRIVATE ${MLX5_lib} CUDA::cudart_static)
  if(HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED)
    target_compile_definitions(nvshmem_transport_ibgda PRIVATE HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED=${HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED})
  endif()
  if(HAVE_MLX5DV_UMEM_MASK_DMABUF)
      target_compile_definitions(nvshmem_transport_ibgda PRIVATE HAVE_MLX5DV_UMEM_MASK_DMABUF=${HAVE_MLX5DV_UMEM_MASK_DMABUF})
  endif()
  if(HAVE_IBV_ACCESS_RELAXED_ORDERING)
    target_compile_definitions(nvshmem_transport_ibgda PRIVATE HAVE_IBV_ACCESS_RELAXED_ORDERING=${HAVE_IBV_ACCESS_RELAXED_ORDERING})
  endif()
  set(ACTIVE_TRANSPORTS ${ACTIVE_TRANSPORTS} nvshmem_transport_ibgda)
endif()

set_target_properties(${ACTIVE_TRANSPORTS}
                      PROPERTIES PREFIX ""
                      CXX_STANDARD_REQUIRED ON
                      CXX_STANDARD 11
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
                      VERSION ${TRANSPORT_VERSION_MAJOR}.${TRANSPORT_VERSION_MINOR}.${TRANSPORT_VERSION_PATCH}
                      SOVERSION ${TRANSPORT_VERSION_MAJOR})
## End transports

## Start final lib prep
## zhangyan35
target_link_libraries(nvshmem_host)
##target_link_libraries(nvshmem_host PRIVATE nvshmem_device)
target_link_options(nvshmem_host PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/nvshmem_host.sym")
set_source_files_properties(${NVSHMEM_HOST_SOURCES} ${NVSHMEM_DEVICE_SOURCES} PROPERTIES COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=32>)
target_sources(nvshmem PRIVATE ${NVSHMEM_HOST_SOURCES} ${NVSHMEM_HOST_SOURCES_NOMAXREGCOUNT} ${NVSHMEM_DEVICE_SOURCES} ${NVSHMEM_DEVICE_SOURCES_NOMAXREGCOUNT})
target_sources(nvshmem_host PRIVATE ${NVSHMEM_HOST_SOURCES} ${NVSHMEM_HOST_SOURCES_NOMAXREGCOUNT})

add_library(NVSHMEM::nvshmem ALIAS nvshmem)

## End final lib prep
# End nvshmem lib definitions

# Start header configuration and export
configure_file(include/non_abi/nvshmem_version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/non_abi/nvshmem_version.h @ONLY)
configure_file(include/non_abi/nvshmem_build_options.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/non_abi/nvshmem_build_options.h)

set(INC_EXPORTS
    ${INC_EXPORTS}
    non_abi/device/pt-to-pt/configs.cuh
    non_abi/device/pt-to-pt/exception.cuh
    non_abi/device/pt-to-pt/utils.cuh
    non_abi/device/pt-to-pt/ibgda_device.cuh
    non_abi/device/pt-to-pt/utils_device.cuh
    device/nvshmem_device_macros.h
    device_host_transport/nvshmem_common_ibgda.h
    non_abi/nvshmemx_error.h
    non_abi/nvshmem_build_options.h
    non_abi/nvshmem_version.h
    internal/host/util.h
    internal/host/error_codes_internal.h
    internal/host/nvshmem_internal.h
    internal/host/debug.h
    internal/host_transport/transport.h
    internal/host_transport/cudawrap.h
    internal/host_transport/nvshmemi_transport_defines.h
    host/p2pcomm_api.h
    host/env/env_defs.h
    bootstrap_host_transport/env_defs_internal.h
)

foreach(INC ${INC_EXPORTS})
  configure_file(include/${INC} ${CMAKE_CURRENT_BINARY_DIR}/include/${INC} COPYONLY)
endforeach(INC ${INC_EXPORTS})
# End header configuration and export

# start share configuration

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nvshmem_transport.sym
               ${CMAKE_CURRENT_BINARY_DIR}/share/src/transport-plugins/nvshmem_transport.sym COPYONLY)

set(TRANSPORT_INC_EXPORTS
    bootstrap_host_transport/env_defs_internal.h
    device_host_transport/nvshmem_common_ibgda.h
    internal/host_transport/cudawrap.h
    internal/host_transport/nvshmemi_transport_defines.h
    internal/host_transport/transport.h
    non_abi/nvshmem_version.h
    non_abi/nvshmemx_error.h
    non_abi/nvshmem_build_options.h
    host/p2pcomm_api.h
)

foreach(SHARE ${TRANSPORT_INC_EXPORTS})
  configure_file(include/${SHARE} ${CMAKE_CURRENT_BINARY_DIR}/share/src/transport-plugins/include/${SHARE} COPYONLY)
endforeach(SHARE ${TRANSPORT_INC_EXPORTS})

set(TRANSPORT_EXPORTS
    common/env_defs.h
    common/mlx5_ifc.h
    common/mlx5_prm.h
    common/transport_common.cpp
    common/transport_common.h
    common/transport_ib_common.cpp
    common/transport_ib_common.h
    common/transport_mlx5_common.cpp
    common/transport_mlx5_common.h
    ibgda/ibgda.cpp
)

foreach(SHARE ${TRANSPORT_EXPORTS})
  configure_file(modules/transport/${SHARE} ${CMAKE_CURRENT_BINARY_DIR}/share/src/transport-plugins/${SHARE} COPYONLY)
endforeach(SHARE ${TRANSPORT_EXPORTS})

set(TRANSPORT_CMAKE_EXPORTS
    CMakeLists.txt
    common/CMakeLists.txt
    ibgda/CMakeLists.txt
)

foreach(SHARE ${TRANSPORT_CMAKE_EXPORTS})
  configure_file(modules/transport/${SHARE}.in ${CMAKE_CURRENT_BINARY_DIR}/share/src/transport-plugins/${SHARE} @ONLY)
endforeach(SHARE ${TRANSPORT_CMAKE_EXPORTS})
# end share configuration

# start custom builds for nvshmem packages
add_custom_target(git_commit ALL COMMAND test -f git_commit.txt || git rev-parse HEAD > git_commit.txt || echo "not built from a git repo" > git_commit.txt
                  COMMAND grep -q NVSHMEM_MAJOR git_commit.txt || echo "NVSHMEM_MAJOR := ${PROJECT_VERSION_MAJOR}" >> git_commit.txt
                  COMMAND grep -q NVSHMEM_MINOR git_commit.txt || echo "NVSHMEM_MINOR := ${PROJECT_VERSION_MINOR}" >> git_commit.txt
                  COMMAND grep -q NVSHMEM_PATCH git_commit.txt || echo "NVSHMEM_PATCH := ${PROJECT_VERSION_PATCH}" >> git_commit.txt
                  COMMAND grep -q NVSHMEM_PACKAGE git_commit.txt || echo "NVSHMEM_PACKAGE := ${PROJECT_VERSION_TWEAK}" >> git_commit.txt
                  COMMAND cp git_commit.txt version.txt
                  BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/../git_commit.txt
                             ${CMAKE_CURRENT_SOURCE_DIR}/../version.txt
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(NVSHMEM_BUILD_PACKAGES AND NVSHMEM_BUILD_HYDRA_LAUNCHER)
  add_custom_target(hydra
                    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/install_hydra.sh ${CMAKE_CURRENT_SOURCE_DIR}/hydra_build ${CMAKE_CURRENT_BINARY_DIR}
                    COMMAND rm -rf ${CMAKE_CURRENT_SOURCE_DIR}/hydra_build
                    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/bin/hydra_nameserver
                               ${CMAKE_CURRENT_BINARY_DIR}/bin/hydra_persist
                               ${CMAKE_CURRENT_BINARY_DIR}/bin/hydra_pmi_proxy
                               ${CMAKE_CURRENT_BINARY_DIR}/bin/nvshmrun
                               ${CMAKE_CURRENT_BINARY_DIR}/bin/nvshmrun.hydra
                  )
  add_dependencies(nvshmem git_commit hydra)
endif()

