/****
 * Copyright (c) 2016-2020, NVIDIA Corporation.  All rights reserved.
 *
 * Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
 *
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
 * This software is available to you under the BSD license.
 *
 * Portions of this file are derived from Sandia OpenSHMEM.
 *
 * See COPYRIGHT for license information
 ****/

#ifndef _UTIL_H
#define _UTIL_H

#include "bootstrap_host_transport/env_defs_internal.h"
#include "internal/host/debug.h"
#include "internal/host/error_codes_internal.h"
#include "non_abi/nvshmem_build_options.h"
#include "non_abi/nvshmemx_error.h"
#include <cassert>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <inttypes.h>
#include <sstream>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <tuple>
#include <unistd.h>
#include <vector>

#ifndef likely
#define likely(x) (__builtin_expect(!!(x), 1))
#endif

#ifndef unlikely
#define unlikely(x) (__builtin_expect(!!(x), 0))
#endif

#define NZ_DEBUG_JMP(status, err, label, ...)                                  \
  do {                                                                         \
    if (unlikely(status != 0)) {                                               \
      if (nvshmem_debug_level >= NVSHMEM_LOG_TRACE) {                          \
        fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__,     \
                status);                                                       \
        fprintf(stderr, __VA_ARGS__);                                          \
      }                                                                        \
      status = err;                                                            \
      goto label;                                                              \
    }                                                                          \
  } while (0)

#define CUDA_DRIVER_CHECK(cmd)                                                 \
  do {                                                                         \
    CUresult r = cmd;                                                          \
    cuGetErrorString(r, &p_err_str);                                           \
    if (unlikely(CUDA_SUCCESS != r)) {                                         \
      WARN("Cuda failure '%s'", p_err_str);                                    \
      return NVSHMEMI_UNHANDLED_CUDA_ERROR;                                    \
    }                                                                          \
  } while (false)

#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    CUresult result = (stmt);                                                  \
    cuGetErrorString(result, &p_err_str);                                      \
    if (unlikely(CUDA_SUCCESS != result)) {                                    \
      fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__,    \
              p_err_str);                                                      \
      exit(-1);                                                                \
    }                                                                          \
    assert(CUDA_SUCCESS == result);                                            \
  } while (0)

#define CUDA_RUNTIME_CHECK(stmt)                                               \
  do {                                                                         \
    cudaError_t result = (stmt);                                               \
    if (unlikely(cudaSuccess != result)) {                                     \
      fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__,    \
              cudaGetErrorString(result));                                     \
      exit(-1);                                                                \
    }                                                                          \
    assert(cudaSuccess == result);                                             \
  } while (0)

#define CUDA_RUNTIME_CHECK_GOTO(stmt, res, label)                              \
  do {                                                                         \
    cudaError_t result = (stmt);                                               \
    if (unlikely(cudaSuccess != result)) {                                     \
      fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__,    \
              cudaGetErrorString(result));                                     \
      res = NVSHMEMI_UNHANDLED_CUDA_ERROR;                                     \
      goto label;                                                              \
    }                                                                          \
  } while (0)

#define CUDA_RUNTIME_ERROR_STRING(result)                                      \
  do {                                                                         \
    if (unlikely(cudaSuccess != result)) {                                     \
      fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__,    \
              cudaGetErrorString(result));                                     \
    }                                                                          \
  } while (0)

#define CUDA_DRIVER_ERROR_STRING(result)                                       \
  do {                                                                         \
    if (unlikely(CUDA_SUCCESS != result)) {                                    \
      cuGetErrorString(result, &p_err_str);                                    \
      fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__,    \
              p_err_str);                                                      \
    }                                                                          \
  } while (0)

nvshmemResult_t nvshmemu_gethostname(char *hostname, int maxlen);

#define NVSHMEMI_WRAPLEN 80
char *nvshmemu_wrap(const char *str, const size_t wraplen, const char *indent,
                    const int strip_backticks);

extern const char *p_err_str;

extern struct nvshmemi_options_s nvshmemi_options;

enum { NVSHMEMI_OPTIONS_STYLE_INFO = 0, NVSHMEMI_OPTIONS_STYLE_RST };

int nvshmemi_options_init(void);
void nvshmemi_options_print(int style);

#endif
