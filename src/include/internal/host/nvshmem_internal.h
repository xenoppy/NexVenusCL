/*
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _INTERNAL_H
#define _INTERNAL_H

#include <atomic>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include "internal/host_transport/cudawrap.h"
#include "internal/host_transport/transport.h"
#include "non_abi/nvshmem_build_options.h"
#include "non_abi/nvshmemx_error.h"

/* This is a requirement imposed by DMA-BUF which only supports 32-bit
 * registrations */
#define NVSHMEMI_DMA_BUF_MAX_LENGTH 0x100000000ULL
#define NVSHMEMI_MAX_HANDLE_LENGTH 2147483648ULL

#define MAX_TRANSPORT_EP_COUNT 1

#define NVSHMEMI_TRANSPORT_IS_CAP(transport, cap_idx, flag)                    \
  ((transport)->cap[(cap_idx)] & flag)
#define NVSHMEMI_TRANSPORT_OPS_IS_GET_MEM(transport)                           \
  ((transport)->host_ops.get_mem_handle != NULL)
#define NVSHMEMI_TRANSPORT_OPS_IS_RELEASE_MEM(transport)                       \
  ((transport)->host_ops.release_mem_handle != NULL)
#define NVSHMEMI_TRANSPORT_OPS_IS_ADD_DEVICE_REMOTE_MEM(transport)             \
  ((transport)->host_ops.add_device_remote_mem_handles != NULL)

void nvshmemi_init_debug();
int nvshmemi_get_cucontext();

extern struct nvshmemi_cuda_fn_table *nvshmemi_cuda_syms;

int nvshmemi_cuda_library_init(struct nvshmemi_cuda_fn_table *table);

int nvshmemi_transport_init(nvshmem_transport_t *transport, void *device_state,
                            int pe);

#endif
