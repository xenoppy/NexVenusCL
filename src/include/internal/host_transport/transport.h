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

#ifndef __TRANSPORT_H
#define __TRANSPORT_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

/* This header, along with the six below, comprise
 * the ABI for transport modules.
 */
#include "bootstrap_host_transport/env_defs_internal.h"
#include "internal/host_transport/nvshmemi_transport_defines.h"
#include "non_abi/nvshmem_build_options.h"
#include "non_abi/nvshmem_version.h"
#include "non_abi/nvshmemx_error.h"

/* patch_version + minor_version * 100 + major_version * 10000 */
#define NVSHMEM_TRANSPORT_INTERFACE_VERSION                                    \
  (NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION * 10000 +                            \
   NVSHMEM_TRANSPORT_PLUGIN_MINOR_VERSION * 100 +                              \
   NVSHMEM_TRANSPORT_PLUGIN_PATCH_VERSION)

#define NVSHMEM_TRANSPORT_MAJOR_VERSION(ver) (ver / 10000)
#define NVSHMEM_TRANSPORT_MAJOR_MINOR_VERSION(ver) (ver / 100)

enum {
  NVSHMEM_TRANSPORT_CAP_MAP = 1,
  NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST = 1 << 1,
  NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD = 1 << 2,
  NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS = 1 << 3,
  NVSHMEM_TRANSPORT_CAP_CPU_WRITE = 1 << 4,
  NVSHMEM_TRANSPORT_CAP_CPU_READ = 1 << 5,
  NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS = 1 << 6,
  NVSHMEM_TRANSPORT_CAP_GPU_WRITE = 1 << 7,
  NVSHMEM_TRANSPORT_CAP_GPU_READ = 1 << 8,
  NVSHMEM_TRANSPORT_CAP_GPU_ATOMICS = 1 << 9,
  NVSHMEM_TRANSPORT_CAP_MAX = INT_MAX
};

enum {
  TRANSPORT_OPTIONS_STYLE_INFO = 0,
  TRANSPORT_OPTIONS_STYLE_RST = 1,
  TRANSPORT_OPTIONS_STYLE_MAX = INT_MAX
};

enum {
  NVSHMEM_TRANSPORT_ATTR_NO_ENDPOINTS = 1,
  NVSHMEM_TRANSPORT_ATTR_CONNECTED = 1 << 1,
  NVSHMEM_TRANSPORT_ATTR_MAX = INT_MAX,
};

typedef enum {
  NVSHMEM_TRANSPORT_LIB_CODE_NONE = 0,
  NVSHMEM_TRANSPORT_LIB_CODE_IBGDA = 1,
  NVSHMEM_TRANSPORT_LIB_CODE_MAX = INT_MAX,
} nvshmem_transport_inline_lib_code_type_t;

typedef struct nvshmem_transport_pe_info {
  pcie_id_t pcie_id;
  int pe;
  uint64_t hostHash;
  cudaUUID_t gpu_uuid;
} nvshmem_transport_pe_info_t;

typedef int (*fence_handle)(struct nvshmem_transport *tcurr, int pe,
                            int is_proxy);
typedef int (*quiet_handle)(struct nvshmem_transport *tcurr, int pe,
                            int is_proxy);

struct nvshmem_transport_host_ops {
  int (*can_reach_peer)(int *access, nvshmem_transport_pe_info_t *peer_info,
                        struct nvshmem_transport *transport);
  int (*create_qps)(struct nvshmem_transport *tcurr, int *selected_dev_ids,
                    int num_selected_devs, void **out_rc_handles,
                    int *out_handles_len, int *out_num_dci_eps);
  int (*connect_endpoints)(struct nvshmem_transport *tcurr,
                           int *selected_dev_ids, int num_selected_devs,
                           void *rc_handles);
  int (*get_mem_handle)(nvshmem_mem_handle_t *mem_handle,
                        nvshmem_mem_handle_t *mem_handle_in, void *buf,
                        size_t size, struct nvshmem_transport *transport,
                        bool local_only);
  int (*release_mem_handle)(nvshmem_mem_handle_t *mem_handle,
                            struct nvshmem_transport *transport);
  int (*finalize)(struct nvshmem_transport *transport);
  int (*show_info)(struct nvshmem_transport *transport, int style);
  int (*progress)(struct nvshmem_transport *transport);

  fence_handle fence;
  quiet_handle quiet;
  int (*enforce_cst)(struct nvshmem_transport *transport);
  int (*enforce_cst_at_target)(struct nvshmem_transport *transport);
  int (*add_device_remote_mem_handles)(struct nvshmem_transport *transport,
                                       int transport_stride,
                                       nvshmem_mem_handle_t *mem_handles,
                                       uint64_t heap_offset, size_t size);
};

typedef struct nvshmem_transport {
  /* lib identifiers */
  int api_version;
  nvshmem_transport_inline_lib_code_type_t type;
  int *cap;
  /* APIs */
  struct nvshmem_transport_host_ops host_ops;
  void *state;
  void *type_specific_shared_state;
  void *cache_handle;
  /* transport shares to lib */
  char **device_pci_paths;
  int attr;
  int n_devices;
  bool atomics_complete_on_quiet;
  bool is_successfully_initialized;
  bool no_proxy;
  /* lib shares to transport */
  void *heap_base;
  size_t log2_cumem_granularity;
  uint64_t max_op_len;
  uint32_t atomic_host_endian_min_size;
  int index;
  int my_pe;
  int n_pes;
} nvshmem_transport_v1;

typedef nvshmem_transport_v1 *nvshmem_transport_t;

/**
 * @brief Initialize the P2P transport module.
 * @param transport Pointer to the transport object to be initialized.
 * @return 0 on success, non-zero on failure.
 *
 * This function load the transport module via dynamic library and initializes
 * the transport object.
 */
int nvshmemt_p2p_init(nvshmem_transport_t *transport);

/**
 * @brief Finalize the P2P transport module.
 * @param transport The transport object to be finalized.
 * @return 0 on success, non-zero on failure.
 *
 * This function releases transport resources via the underlying
 * host_ops.finalize function.
 */
int nvshmemi_transport_finalize(nvshmem_transport_t transport);

typedef int (*nvshmemi_transport_init_fn)(nvshmem_transport_t *transport,
                                          struct nvshmemi_cuda_fn_table *table,
                                          int api_version);

#endif
