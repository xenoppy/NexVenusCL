# This file includes boiler plate code for transforming
# environment variables to CMake variables.
# RULES:
# 1. Options only need an entry here if they are on by default
# 2. Paths only need an entry here if they need a non-empty default

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

macro(nvshmem_add_default_on_option OPTION_NAME DESCRIPTION)
  if (DEFINED ENV{${OPTION_NAME}})
    set(${OPTION_NAME}_DEFAULT $ENV{${OPTION_NAME}})
  else()
    set(${OPTION_NAME}_DEFAULT ON)
  endif()
  option(${OPTION_NAME} ${DESCRIPTION} ${${OPTION_NAME}_DEFAULT})
  message( "${OPTION_NAME}: ${${OPTION_NAME}}")
endmacro()

macro (nvshmem_add_default_off_option OPTION_NAME DESCRIPTION)
  option(${OPTION_NAME} ${DESCRIPTION} $ENV{${OPTION_NAME}})
  message( "${OPTION_NAME}: ${${OPTION_NAME}}")
endmacro()



macro (nvshmem_add_default_environment_path PATH_NAME DEFAULT_VALUE DESCRIPTION)
if (DEFINED ENV{${PATH_NAME}})
    set(${PATH_NAME}_DEFAULT $ENV{${PATH_NAME}})
  else()
    set(${PATH_NAME}_DEFAULT ${DEFAULT_VALUE})
  endif()
  set(${PATH_NAME} ${${PATH_NAME}_DEFAULT} CACHE PATH ${DESCRIPTION})
  message( "${PATH_NAME}: ${${PATH_NAME}}")
endmacro()

message( "                                   **NVSHMEM OPTIONS OVERVIEW**                                   \n")
message( "Options set to ON by Default")
message( "___________________________________________________________________________________________________")
message( "\n__TRANSPORT__\n")
nvshmem_add_default_on_option(NVSHMEM_IBGDA_SUPPORT "Enable compilation of the IBGDA remote transport")

message( "\n__BUILD__\n")
nvshmem_add_default_on_option(NVSHMEM_BUILD_TESTS "Build tests")
nvshmem_add_default_on_option(NVSHMEM_BUILD_EXAMPLES "Build examples")
message( "___________________________________________________________________________________________________\n\n")

message( "Options set to OFF by Default")
message( "___________________________________________________________________________________________________")
message( "\n__TRANSPORT__\n")
nvshmem_add_default_off_option(NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY "Force the IBGDA remote transport to only use GPUMEM.")

message( "\n__FUNCTIONALITY__\n")
nvshmem_add_default_off_option(NVSHMEM_ENABLE_ALL_DEVICE_INLINING "Inline all device library code")
nvshmem_add_default_off_option(NVSHMEM_TIMEOUT_DEVICE_POLLING "Enable timeouts for NVSHMEM device-side polling functions (e.g. wait_until)")

message( "\n__BUILD__\n")
nvshmem_add_default_off_option(NVSHMEM_DEBUG "Toggles NVSHMEM debug compilation settings")
nvshmem_add_default_off_option(NVSHMEM_DEVEL "Toggles NVSHMEM devel compilation settings")
nvshmem_add_default_off_option(NVSHMEM_TRACE "Enable NVSHMEM trace print events")
nvshmem_add_default_off_option(NVSHMEM_VERBOSE "Enable the ptxas verbose compilation option")
nvshmem_add_default_off_option(NVSHMEM_ENV_ALL "Display all runtime environment variables, regardless of build settings")
message( "___________________________________________________________________________________________________\n\n")


message( "Paths of dependencies")
message( "___________________________________________________________________________________________________")
nvshmem_add_default_environment_path(CUDA_HOME "/usr/local/cuda" "path to CUDA installation")
nvshmem_add_default_environment_path(NVSHMEM_CLANG_DIR "" "path to force cmake to look for clang when compiling the bitcode library.")
nvshmem_add_default_environment_path(NVSHMEM_PREFIX "/usr/local/nvshmem" "path to NVSHMEM install directory.")
message( "___________________________________________________________________________________________________\n\n")


message( "Packaging specific options set to OFF by default")
message( "___________________________________________________________________________________________________")
nvshmem_add_default_off_option(NVSHMEM_BUILD_PACKAGES "Build package dependencies - gates all other packaging variables")
nvshmem_add_default_off_option(NVSHMEM_BUILD_RPM_PACKAGE "Build RPM package")
nvshmem_add_default_off_option(NVSHMEM_BUILD_DEB_PACKAGE "Build DEB package")
nvshmem_add_default_off_option(NVSHMEM_BUILD_TGZ_PACKAGE "Build TGZ package")
if (NOT NVSHMEM_BUILD_PACKAGES)
  nvshmem_add_default_off_option(NVSHMEM_BUILD_BITCODE_LIBRARY "Build the nvshmem_device bitcode library")
endif()
message( "___________________________________________________________________________________________________\n\n")

message( "Packaging specific options set to ON by default")
message( "___________________________________________________________________________________________________")
nvshmem_add_default_on_option(NVSHMEM_BUILD_HYDRA_LAUNCHER "Enables Building the hydra launcher package")
nvshmem_add_default_on_option(NVSHMEM_BUILD_TXZ_PACKAGE "Build TXZ package")
if (NVSHMEM_BUILD_PACKAGES)
  nvshmem_add_default_on_option(NVSHMEM_BUILD_BITCODE_LIBRARY "Build the nvshmem_device bitcode library")
endif()
message( "___________________________________________________________________________________________________\n\n")


### HIDDEN ###
if (DEFINED ENV{NVSHMEM_DEVICELIB_CUDA_HOME})
  set(NVSHMEM_DEVICELIB_CUDA_HOME_DEFAULT $ENV{NVSHMEM_DEVICELIB_CUDA_HOME})
elseif(DEFINED ENV{CUDA_HOME})
  set(NVSHMEM_DEVICELIB_CUDA_HOME_DEFAULT $ENV{CUDA_HOME})
else()
  set(NVSHMEM_DEVICELIB_CUDA_HOME_DEFAULT "/usr/local/cuda")
endif()
set(NVSHMEM_DEVICELIB_CUDA_HOME ${NVSHMEM_DEVICELIB_CUDA_HOME_DEFAULT} CACHE PATH "path to CUDA installation")

option(NVSHMEM_INSTALL_FUNCTIONAL_TESTS "Install functional tests" $ENV{NVSHMEM_INSTALL_FUNCTIONAL_TESTS})
option(NVSHMEM_TEST_STATIC_LIB "Force tests to link only against the combined nvshmem.a binary" $ENV{NVSHMEM_TEST_STATIC_LIB})

# Set a variable for the nvshmem build information in NVSHMEM
# TODO actually fill this variable with useful information.
set(INFO_BUILD_VARS
  "\"NVSHMEM_DEBUG=${NVSHMEM_DEBUG} \
NVSHMEM_DEVEL=${NVSHMEM_DEVEL} \
NVSHMEM_ENABLE_ALL_DEVICE_INLINING=${NVSHMEM_ENABLE_ALL_DEVICE_INLINING} \
NVSHMEM_IBGDA_SUPPORT=${NVSHMEM_IBGDA_SUPPORT} \
NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY=${NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY} \
NVSHMEM_TEST_STATIC_LIB=${NVSHMEM_TEST_STATIC_LIB} \
NVSHMEM_TIMEOUT_DEVICE_POLLING=${NVSHMEM_TIMEOUT_DEVICE_POLLING} \
NVSHMEM_TRACE=${NVSHMEM_TRACE} \
NVSHMEM_VERBOSE=${NVSHMEM_VERBOSE} \
CUDA_HOME=${CUDA_HOME} \
NVSHMEM_PREFIX=${NVSHMEM_PREFIX} \"")
