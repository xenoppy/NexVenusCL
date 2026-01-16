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
 * @file p2pcomm.cpp
 * @brief P2P Communication entry API implementation
 *
 */
#include "device_host_transport/nvshmem_common_ibgda.h"
#include "host/p2pcomm_api.h"
#include "internal/host/debug.h"
#include "internal/host/nvshmem_internal.h"
#include "internal/host/util.h"
#include "internal/host_transport/cudawrap.h"
#include "internal/host_transport/transport.h"
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace p2p_comm {
std::once_flag flag;
// flag indicating whether global init is done
int global_init_status = 0;

// used by p2pcomm_malloc & p2pcomm_free
// map from aligned pointer to original pointer
std::unordered_map<void *, void *> p2pcomm_mem_map;

void *p2pcomm_malloc(size_t size, size_t alignment) {
  int status = 0;
  void *ptr = nullptr;
  void *aligned_ptr = nullptr;
  size_t bufsize = size;
  int attr_val = 1;

  if (alignment > 0) {
    bufsize = size + alignment - 1;
  }
  status = cudaMalloc(&ptr, bufsize);
  NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                        "cudaMalloc failed.\n");

  status = cuPointerSetAttribute(&attr_val, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                 (CUdeviceptr)ptr);
  NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                        "cuPointerSetAttribute failed.\n");

  status = cudaMemset(ptr, 0, bufsize);
  NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                        "cudaMemset failed.\n");

  if (alignment > 0) {
    aligned_ptr =
        (void *)((size_t)((char *)ptr + alignment - 1) & (~(alignment - 1)));
    // only record if aligned_ptr != ptr
    p2pcomm_mem_map[aligned_ptr] = ptr;
  } else {
    aligned_ptr = ptr;
  }
  INFO(NVSHMEM_MEM,
       "p2pcomm_malloc ptr: %p, aligned_ptr: %p, size: %lu, bufsize:%lu.", ptr,
       aligned_ptr, size, bufsize);

out:
  if (status) {
    if (ptr) {
      cudaError_t _status = cudaFree(ptr);
      CUDA_RUNTIME_ERROR_STRING(_status);
    }
    return nullptr;
  }
  return aligned_ptr;
}

int p2pcomm_free(void *dptr) {
  if (dptr == nullptr) {
    return 0;
  }
  void *orig_ptr = dptr;
  // dptr is aligned_ptr, so we need to get the original ptr
  // if dptr is not in the map, it means it is the original ptr
  if (p2pcomm_mem_map.find(dptr) != p2pcomm_mem_map.end()) {
    orig_ptr = p2pcomm_mem_map[dptr];
    p2pcomm_mem_map.erase(dptr);
  }

  int status = cudaFree(orig_ptr);
  NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                        "cudaFree failed.\n");
  INFO(NVSHMEM_MEM, "p2pcomm_free orig_ptr: %p, dptr: %p.", orig_ptr, dptr);
out:
  return status;
}

/**
 * @brief Static global initialization function.
 * @return 0 on success, non-zero on failure.
 *
 * This initialization includes:
 * - nvshmem options initialization
 * - CUDA library loading and function pointer resolution
 * - CUDA context retrieval
 */
static int global_init() {
  int status = 0;
  status = nvshmemi_options_init();
  NZ_DEBUG_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
               "nvshmem options init failed \n");

  nvshmemi_init_debug();

  if (!nvshmemi_cuda_syms) {
    nvshmemi_cuda_syms = (struct nvshmemi_cuda_fn_table *)calloc(
        1, sizeof(struct nvshmemi_cuda_fn_table));
    NVSHMEMI_NULL_ERROR_JMP(nvshmemi_cuda_syms, status,
                            NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate cuda function table.\n");
  }
  status = nvshmemi_cuda_library_init(nvshmemi_cuda_syms);
  NZ_DEBUG_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
               "nvshmem cuda library init failed \n");

  status = nvshmemi_get_cucontext();
  NZ_DEBUG_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
               "nvshmemi_get_cucontext failed \n");
out:
  global_init_status = status;
  return status;
}

int P2PComm::init() {
  if (initialized) {
    return 0;
  }
  int status = 0;
  nvshmemi_init_ibgda_device_state(nvshmemi_ibgda_device_state);
  int selected_devices[1] = {0};

  std::call_once(flag, global_init);
  status = global_init_status;
  NZ_DEBUG_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "global init failed \n");

  status = nvshmemi_transport_init(&trans, &nvshmemi_ibgda_device_state, pe);
  NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                        "nvshmem transport init failed \n");
  assert(trans != nullptr);

  // create all qps and rc_handle for rcs
  status = trans->host_ops.create_qps(trans, selected_devices, 1, &rc_handles,
                                      &rc_handles_len, &num_dci_eps);
  // rc_handles is a contiguous blob containing both local/remote connection
  // descriptors; each side slices its half using chunk_size below.
  NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                        "connect EPS failed \n");
  status =
      cudaMalloc(&p2pcomm_device_state_d, sizeof(p2pcomm_ibgda_device_state_t));
  NVSHMEMI_NULL_ERROR_JMP(p2pcomm_device_state_d, status,
                          NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                          "Unable to allocate ibgda_device_state_d.\n");
out:
  return status;
}

int P2PComm::register_memory(void *dptr, size_t size) {
  // check if get_mem_handle is supported
  if (!NVSHMEMI_TRANSPORT_OPS_IS_GET_MEM(trans)) {
    return 0;
  }
  // Cache the transport-exported mem handle (lkey/rkey) for this PE; later
  // copied into device state for verbs access.
  return trans->host_ops.get_mem_handle(
      (nvshmem_mem_handle_t *)(&(mem_handle[pe])), nullptr, dptr, size, trans,
      false);
}

int P2PComm::deregister_memory(void *dptr) {
  return trans->host_ops.release_mem_handle(
      (nvshmem_mem_handle_t *)(&(mem_handle[pe])), trans);
}

std::string P2PComm::get_local_mem_handle() const {
  return std::string(mem_handle[pe].reserved, NVSHMEM_MEM_HANDLE_SIZE);
}

int P2PComm::set_remote_mem_handle(const std::string &handle) {
  int status = 0;
  if (handle.size() != NVSHMEM_MEM_HANDLE_SIZE) {
    status = NVSHMEMX_ERROR_INVALID_VALUE;
  }
  NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                        "Invalid remote mem handle size %lu, expected %d \n",
                        handle.size(), NVSHMEM_MEM_HANDLE_SIZE);
  memcpy(mem_handle[1 ^ pe].reserved, handle.data(), NVSHMEM_MEM_HANDLE_SIZE);
out:
  return status;
}

std::string P2PComm::get_local_conn_handle() const {
  size_t chunk_size = rc_handles_len / 2;
  return std::string((char *)rc_handles + (1 ^ pe) * chunk_size, chunk_size);
}

int P2PComm::connect_endpoint(const std::string &conn_handle) {
  int status = 0;
  int selected_devices[1] = {0};

  size_t chunk_size = rc_handles_len / 2;
  assert(conn_handle.size() == chunk_size);
  // Copy remote half of the RC handle blob into our buffer; local half was
  // filled during create_qps.
  memcpy((char *)rc_handles + pe * chunk_size, conn_handle.data(), chunk_size);

  status =
      trans->host_ops.connect_endpoints(trans, selected_devices, 1, rc_handles);
  NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                        "connect EPS failed \n");
  if (status == 0) {
    initialized = true;
  }
out:
  return status;
}

int P2PComm::fini() {
  int status = 0;
  bool failed = false;
  // the trans itself is freed by ibgda transport finalize
  status = nvshmemi_transport_finalize(trans);
  if (status != 0) {
    failed = true;
  }
  // rc_handles is malloced inside trans->host_ops.create_qps
  free(rc_handles);
  status = cudaFree(p2pcomm_device_state_d);
  if (status != cudaSuccess) {
    NVSHMEMI_ERROR_PRINT("cudaFree failed \n");
    failed = true;
  }
  return failed ? -1 : 0;
}

/**
 * @brief Convert nvshmemi_ibgda_device_state_t to p2pcomm_ibgda_device_state_t
 */
static int convert_device_state(p2pcomm_ibgda_device_state_t &p2p_st,
                                const nvshmemi_ibgda_device_state_t &nvshmem_st,
                                int pe, int num_dci_eps,
                                p2pcomm_mem_handle_t *mem_handle) {
  p2p_st.version = nvshmem_st.version;
  p2p_st.num_rc_per_pe = nvshmem_st.num_rc_per_pe;
  p2p_st.rc_map_type = nvshmem_st.rc_map_type;
  p2p_st.num_devices_initialized = nvshmem_st.num_devices_initialized;
    // underlying mem_handle layout: struct nvshmemt_ib_common_mem_handle; we
    // only pick the lkey/rkey words the device-side verbs need.
  p2p_st.lkey.key = htobe32(*((int32_t *)(&mem_handle[pe]) + 5));
  p2p_st.rkey.key = htobe32(*((int32_t *)(&mem_handle[1 ^ pe]) + 6));
  INFO(NVSHMEM_INIT, "p2pcomm convert_device_state lkey: %u, rkey: %u.",
       p2p_st.lkey.key, p2p_st.rkey.key);
  p2p_st.cqs =
      nvshmem_st.globalmem.cqs + num_dci_eps; // top dic.num_eps of cqs for dci
  p2p_st.rcs = nvshmem_st.globalmem.rcs;
  return 0;
}

int P2PComm::update_device_state_d() {
  int status = 0;
  p2pcomm_ibgda_device_state_t tmp;
  convert_device_state(tmp, nvshmemi_ibgda_device_state, pe, num_dci_eps,
                       mem_handle);
  // Push the current host snapshot into the device-visible state struct.
  status =
      cudaMemcpy(p2pcomm_device_state_d, (void *)&tmp,
                 sizeof(p2pcomm_ibgda_device_state_t), cudaMemcpyHostToDevice);
  return status;
}

} // end of namespace p2p_comm
