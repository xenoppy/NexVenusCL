/*
 * Copyright (c) Nex-AGI. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "internal/host_transport/transport.h"          // for nvshmem_trans...
#include "bootstrap_host_transport/env_defs_internal.h" // for nvshmemi_opti...
#include "internal/host/debug.h"                        // for INFO, NVSHMEM...
#include "internal/host/error_codes_internal.h"         // for NVSHMEMI_INTE...
#include "internal/host/nvshmem_internal.h"             // for nvshmemi_loca...
#include "internal/host/util.h"                         // for nvshmemi_options
#include "non_abi/nvshmem_build_options.h"              // for NVSHMEM_IBGDA...
#include "non_abi/nvshmem_version.h"                    // for NVSHMEM_TRANS...
#include "non_abi/nvshmemx_error.h"                     // for NVSHMEMI_ERRO...
#include <assert.h>                                     // for assert
#include <dlfcn.h>                                      // for dlclose, dlerror
#include <stdint.h>                                     // for SIZE_MAX
#include <stdio.h>                                      // for snprintf, NULL
#include <stdlib.h>                                     // for calloc
#include <strings.h>                                    // for strncasecmp

#define TRANSPORT_STRING_MAX_LENGTH 8
#define NVSHMEM_TRANSPORT_COUNT 6
#define NUM_OF_PES 2

#ifdef NVSHMEM_IBGDA_SUPPORT
static void *transport_lib_IBGDA = NULL;
#endif

int nvshmemi_transport_init(nvshmem_transport_t *transport, void *device_state,
                            int pe) {
  int status = 0;
  int index = 0;
  nvshmemi_transport_init_fn init_fn;
  const int transport_object_file_len = 100;
  char transport_object_file[transport_object_file_len];

  if (nvshmemi_options.IB_ENABLE_IBGDA) {
    status = snprintf(transport_object_file, transport_object_file_len,
                      "nvshmem_transport_ibgda.so.%d",
                      NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION);
    if (status < 0 || status > transport_object_file_len) {
      WARN("Unable to open the %s transport. %s\n", transport_object_file,
           dlerror());
      goto out;
    }
    transport_lib_IBGDA = dlopen(transport_object_file, RTLD_NOW);
    if (transport_lib_IBGDA == NULL) {
      WARN("Unable to open the %s transport. %s\n", transport_object_file,
           dlerror());
      goto out;
    }

    init_fn =
        (nvshmemi_transport_init_fn)dlsym(transport_lib_IBGDA, "nvshmemt_init");
    if (!init_fn) {
      dlclose(transport_lib_IBGDA);
      transport_lib_IBGDA = NULL;
      WARN("Unable to get info from %s transport.\n", transport_object_file);
      goto out;
    }

    status = init_fn(transport, nvshmemi_cuda_syms,
                     NVSHMEM_TRANSPORT_INTERFACE_VERSION);
    if (!status) {
      assert(NVSHMEM_TRANSPORT_MAJOR_MINOR_VERSION((*transport)->api_version) <=
             NVSHMEM_TRANSPORT_MAJOR_MINOR_VERSION(
                 NVSHMEM_TRANSPORT_INTERFACE_VERSION));
      // IBGDA plugin filled host_ops; set base metadata that core code relies
      // on (heap base, device state pointer, cap table size, etc.).
      (*transport)->heap_base = nullptr;
      (*transport)->log2_cumem_granularity =
          20; // make it 1MB by default for IBGDA
      (*transport)->cap = (int *)calloc(NUM_OF_PES, sizeof(int));
      (*transport)->index = index;
      (*transport)->my_pe = pe;
      (*transport)->n_pes = NUM_OF_PES;
      (*transport)->cache_handle = nullptr;
      (*transport)->type_specific_shared_state = device_state;
      if ((*transport)->max_op_len == 0) {
        (*transport)->max_op_len = SIZE_MAX;
      }
      // nvshmemi_device_state.ibgda_is_initialized = true;
      index++;
    } else {
      NVSHMEMI_ERROR_PRINT("init failed for transport: IBGDA");
      dlclose(transport_lib_IBGDA);
      transport_lib_IBGDA = NULL;
      status = 0;
    }
  } else {
    INFO(NVSHMEM_INIT, "IBGDA Disabled by the environment.");
  }

  if (index == 0) {
    NVSHMEMI_ERROR_PRINT(
        "Unable to initialize any transports. returning error.");
    status = NVSHMEMX_ERROR_INTERNAL;
  }
out:
  return status;
}

int nvshmemi_transport_finalize(nvshmem_transport_t transport) {
  INFO(NVSHMEM_INIT, "In nvshmemi_transport_finalize");
  int status = 0;
  if (transport == NULL)
    return 0;

  status = transport->host_ops.finalize(transport);
  NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                        "transport finalize failed \n");

out:
#ifdef NVSHMEM_IBGDA_SUPPORT
  if (transport_lib_IBGDA) {
    dlclose(transport_lib_IBGDA);
    transport_lib_IBGDA = NULL;
  }
#endif
  return status;
}
