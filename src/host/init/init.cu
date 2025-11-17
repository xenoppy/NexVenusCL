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
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cuda.h>
#include <dirent.h>
#include <fcntl.h>
#include <list>
#include <sched.h>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "internal/host/nvshmem_internal.h"
#include "non_abi/nvshmem_build_options.h"

#ifdef NVSHMEM_IBGDA_SUPPORT
#include "device_host_transport/nvshmem_common_ibgda.h"
#endif

#include "internal/host/debug.h"
#include "internal/host/util.h"
#include "unistd.h"
#include <stdlib.h>
#include <string.h>

struct nvshmemi_cuda_fn_table *nvshmemi_cuda_syms;

int nvshmem_debug_level;
uint64_t nvshmem_debug_mask =
    NVSHMEM_INIT | NVSHMEM_MEM; // Default debug sub-system mask is INIT&MEM
pthread_mutex_t nvshmem_debug_output_lock;
FILE *nvshmem_debug_file = stdout;

// Used by CUDA_XXX macros
const char *p_err_str;

#ifdef NVSHMEM_TRACE
std::chrono::high_resolution_clock::time_point nvshmem_epoch;
#endif

void nvshmemi_init_debug() {
  const char *nvshmem_debug = nvshmemi_options.DEBUG;
  if (!nvshmemi_options.DEBUG_provided &&
      !nvshmemi_options.DEBUG_SUBSYS_provided) {
    nvshmem_debug_level = NVSHMEM_LOG_NONE;
  } else if (strcmp_case_insensitive(nvshmem_debug, "VERSION") == 0) {
    nvshmem_debug_level = NVSHMEM_LOG_VERSION;
  } else if (strcmp_case_insensitive(nvshmem_debug, "WARN") == 0) {
    nvshmem_debug_level = NVSHMEM_LOG_WARN;
  } else if (strcmp_case_insensitive(nvshmem_debug, "INFO") == 0) {
    nvshmem_debug_level = NVSHMEM_LOG_INFO;
  } else if (strcmp_case_insensitive(nvshmem_debug, "ABORT") == 0) {
    nvshmem_debug_level = NVSHMEM_LOG_ABORT;
  } else if (strcmp_case_insensitive(nvshmem_debug, "TRACE") == 0) {
    nvshmem_debug_level = NVSHMEM_LOG_TRACE;
  } else {
    /* OpenSHMEM spec treats SHMEM_DEBUG as a boolean, enable INFO logging
     * when user-supplied value does match one of the above. */
    nvshmem_debug_level = NVSHMEM_LOG_INFO;
  }

  /* Parse the NVSHMEM_DEBUG_SUBSYS env var
   * This can be a comma separated list such as INIT,P2P
   * or ^INIT,P2P etc
   */
  /* Note: strtok will modify the string, operate on a copy */
  char *nvshmem_debug_subsys = strdup(nvshmemi_options.DEBUG_SUBSYS);
  if (nvshmem_debug_subsys != NULL) {
    char *subsys = strtok(nvshmem_debug_subsys, ",");
    while (subsys != NULL) {
      int invert = 0;
      uint64_t mask = 0;
      if (subsys[0] == '^') {
        invert = 1;
        subsys++;
      }
      if (strcmp_case_insensitive(subsys, "INIT") == 0) {
        mask = NVSHMEM_INIT;
      } else if (strcmp_case_insensitive(subsys, "P2P") == 0) {
        mask = NVSHMEM_P2P;
      } else if (strcmp_case_insensitive(subsys, "TRANSPORT") == 0) {
        mask = NVSHMEM_TRANSPORT;
      } else if (strcmp_case_insensitive(subsys, "MEM") == 0) {
        mask = NVSHMEM_MEM;
      } else if (strcmp_case_insensitive(subsys, "UTIL") == 0) {
        mask = NVSHMEM_UTIL;
      } else if (strcmp_case_insensitive(subsys, "ALL") == 0) {
        mask = NVSHMEM_ALL;
      } else {
        mask = 0;
        WARN("Unrecognized value in DEBUG_SUBSYS: %s%s", invert ? "^" : "",
             subsys);
      }
      if (mask) {
        if (invert)
          nvshmem_debug_mask &= ~mask;
        else
          nvshmem_debug_mask |= mask;
      }
      subsys = strtok(NULL, ",");
    }

    free(nvshmem_debug_subsys);
  }

  /* Parse and expand the NVSHMEM_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * NVSHMEM_DEBUG level is > VERSION
   */
  const char *nvshmem_debug_filename = nvshmemi_options.DEBUG_FILE;
  if (nvshmem_debug_level > NVSHMEM_LOG_VERSION &&
      nvshmemi_options.DEBUG_FILE_provided) {
    int c = 0;
    char debugFn[PATH_MAX + 1] = "";
    char *dfn = debugFn;
    while (nvshmem_debug_filename[c] != '\0' && c < PATH_MAX) {
      if (nvshmem_debug_filename[c++] != '%') {
        *dfn++ = nvshmem_debug_filename[c - 1];
        continue;
      }
      switch (nvshmem_debug_filename[c++]) {
        case '%': // Double %
          *dfn++ = '%';
          break;
        case 'h': // %h = hostname
          char hostname[1024];
          nvshmemu_gethostname(hostname, 1024);
          dfn += snprintf(dfn, PATH_MAX, "%s", hostname);
          break;
        case 'p': // %p = pid
          dfn += snprintf(dfn, PATH_MAX, "%d", getpid());
          break;
        default: // Echo everything we don't understand
          *dfn++ = '%';
          *dfn++ = nvshmem_debug_filename[c - 1];
          break;
      }
    }
    *dfn = '\0';
    if (debugFn[0] != '\0') {
      FILE *file = fopen(debugFn, "w");
      if (file != NULL) {
        INFO(NVSHMEM_ALL, "DEBUG file is '%s'", debugFn);
        nvshmem_debug_file = file;
      }
    }
  }
  pthread_mutex_init(&nvshmem_debug_output_lock, NULL);

#ifdef NVSHMEM_TRACE
  nvshmem_epoch = std::chrono::high_resolution_clock::now();
#endif
}

int nvshmemi_get_cucontext() {
  CUdevice cudevice;
  // int leastPriority, greatestPriority;
  int status = NVSHMEMX_SUCCESS;

  CUCHECK(nvshmemi_cuda_syms, cuInit(0));

  status = CUPFN(nvshmemi_cuda_syms, cuCtxGetDevice(&cudevice));
  if (status) {
    TRACE(NVSHMEM_INIT, "GPU not selected, cuCtxGetDevice failed, err: %d",
          status);
    status = NVSHMEMX_ERROR_GPU_NOT_SELECTED;
    goto out;
  } else {
    CUresult cres = CUPFN(nvshmemi_cuda_syms, cuCtxSynchronize());
    status = NVSHMEMX_SUCCESS;
  }
out:
  return status;
}