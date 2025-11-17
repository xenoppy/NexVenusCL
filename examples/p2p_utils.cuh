/**
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
 * @file p2p_utils.cuh
 * @brief Common utilities and CUDA kernels for P2P communication
 *
 * Shared utilities and CUDA kernels used across multiple test programs for P2P
 * communication using InfiniBand GPU Direct Access (IBGDA) transport.
 *
 * This file contains:
 * - Host-side utility functions (file I/O, string operations)
 * - Device-side CUDA kernels for data processing and verification
 */

#ifndef P2P_UTILS_CUH_
#define P2P_UTILS_CUH_

#include "non_abi/device/pt-to-pt/ibgda_device.cuh"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <p2pcomm_api.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

// =============================================================================
// Host-side Utility Functions
// =============================================================================

// Directory for storing communication handles and temporary files
constexpr const char *TMP_DIR = "tmp";

/**
 * @brief Ensure tmp directory exists, create if it doesn't
 */
inline void ensure_tmp_dir() {
  struct stat info;
  if (stat(TMP_DIR, &info) != 0) {
// Directory doesn't exist, create it
#ifdef _WIN32
    _mkdir(TMP_DIR);
#else
    mkdir(TMP_DIR, 0755);
#endif
  }
}

/**
 * @brief Get full path for a file in tmp directory
 *
 * @param filename Base filename
 * @return Full path with tmp/ prefix
 */
inline std::string tmp_path(const std::string &filename) {
  return std::string(TMP_DIR) + "/" + filename;
}

/**
 * @brief Save a pointer value to a file (for inter-process communication)
 *
 * @param ptr Pointer to save
 * @param filepath Full file path (use tmp_path() to generate)
 */
inline void save_pointer_to_file(void *ptr, const char *filepath) {
  ensure_tmp_dir();
  FILE *file = fopen(filepath, "wb");
  if (file == NULL) {
    perror("Failed to open file");
    return;
  }
  size_t ptr_value = reinterpret_cast<size_t>(ptr);
  fwrite(&ptr_value, sizeof(ptr_value), 1, file);
  fclose(file);
}

/**
 * @brief Load a pointer value from a file
 *
 * @param filepath Full file path (use tmp_path() to generate)
 * @return Pointer value loaded from file, or NULL on error
 */
inline void *read_pointer_from_file(const char *filepath) {
  FILE *file = fopen(filepath, "rb");
  if (file == NULL) {
    perror("Failed to open file");
    return NULL;
  }
  size_t ptr_value;
  fread(&ptr_value, sizeof(ptr_value), 1, file);
  fclose(file);
  return reinterpret_cast<void *>(ptr_value);
}

/**
 * @brief Save string data to a file
 *
 * @param filepath Full file path (use tmp_path() to generate)
 * @param data String data to save
 */
inline void save_string_to_file(const std::string &filepath,
                                const std::string &data) {
  ensure_tmp_dir();
  std::ofstream out(filepath, std::ios::binary);
  if (!out) {
    std::cerr << "Failed to open " << filepath << " for writing\n";
    return;
  }
  out.write(data.data(), data.size());
  out.close();
}

/**
 * @brief Load string data from a file
 *
 * @param filepath Full file path (use tmp_path() to generate)
 * @return String data loaded from file, or empty string on error
 */
inline std::string load_string_from_file(const std::string &filepath) {
  std::ifstream in(filepath, std::ios::binary | std::ios::ate);
  if (!in) {
    std::cerr << "Failed to open " << filepath << " for reading\n";
    return "";
  }

  std::streamsize size = in.tellg();
  in.seekg(0, std::ios::beg);

  std::string data(size, '\0');
  in.read(&data[0], size);
  in.close();
  return data;
}

/**
 * @brief Split a comma-separated string into a vector of strings
 *
 * @param input Comma-separated string to split
 * @return Vector of individual strings
 */
inline std::vector<std::string>
split_string_by_comma(const std::string &input) {
  std::vector<std::string> result;
  size_t start = 0;
  size_t end = input.find(',');

  while (end != std::string::npos) {
    result.push_back(input.substr(start, end - start));
    start = end + 1;
    end = input.find(',', start);
  }

  result.push_back(input.substr(start));
  return result;
}

/**
 * @brief Print first N values from device buffer
 *
 * @param prefix Prefix string for output
 * @param d_buf Device buffer pointer
 * @param n Number of values to print
 */
inline void print_first_n_values(const char *prefix, void *d_buf, int n) {
  float *h_data = new float[n];
  cudaMemcpy(h_data, d_buf, n * sizeof(float), cudaMemcpyDeviceToHost);
  printf("%s: ", prefix);
  for (int i = 0; i < n; i++) {
    printf("%.6f ", h_data[i]);
  }
  printf("\n");
  delete[] h_data;
}

/**
 * @brief Allocate and initialize buffer with zero
 *
 * @param size Buffer size in bytes
 * @param alignment Alignment requirement in bytes (default: same as size)
 * @return Pointer to allocated buffer, or nullptr on error
 */
inline void *allocate_buffer(size_t size, size_t alignment = 0) {
  if (alignment == 0) {
    alignment = size;
  }
  void *buf = p2p_comm::p2pcomm_malloc(size, alignment);
  if (!buf) {
    std::cerr << "malloc failed.\n";
    return nullptr;
  }
  cudaMemset(buf, 0, size);
  return buf;
}

/**
 * @brief Cleanup a single P2PComm connection (deregister memory but don't free
 * buffer)
 *
 * @param comm P2PComm reference to cleanup
 * @param buf Buffer pointer to deregister (can be nullptr)
 */
inline void cleanup_single_comm(p2p_comm::P2PComm &comm, void *buf) {
  if (buf) {
    comm.deregister_memory(buf);
    cudaDeviceSynchronize();
  }
  comm.fini();
}

/**
 * @brief Free a P2P buffer (call this once after all comms are cleaned up)
 *
 * @param buf Buffer pointer to free (can be nullptr)
 */
inline void free_p2p_buffer(void *buf) {
  if (buf) {
    p2p_comm::p2pcomm_free(buf);
    cudaDeviceSynchronize();
  }
}

/**
 * @brief Register memory and save handles for a single connection
 *
 * @param comm P2PComm reference
 * @param buf Buffer pointer
 * @param size Buffer size
 * @param local_tag Local tag for file naming
 * @param remote_tag Remote tag for file naming
 * @return 0 on success, -1 on error
 */
inline int register_memory_and_save_handles(p2p_comm::P2PComm &comm, void *buf,
                                            size_t size,
                                            const std::string &local_tag,
                                            const std::string &remote_tag) {
  if (comm.register_memory(buf, size) != 0) {
    std::cerr << "register_memory failed for [" << remote_tag << "]"
              << std::endl;
    return -1;
  }

  auto mem_str = comm.get_local_mem_handle();
  save_string_to_file(
      tmp_path(local_tag + "-" + remote_tag + "-mem_handle.txt"), mem_str);
  save_pointer_to_file(
      buf, tmp_path(local_tag + "-" + remote_tag + "-addr_handle.txt").c_str());

  return 0;
}

/**
 * @brief Load and set remote memory handle for a single connection
 *
 * @param comm P2PComm reference
 * @param local_tag Local tag for file naming
 * @param remote_tag Remote tag for file naming
 * @param remote_addr_map Map to store remote address pointer
 * @return 0 on success, -1 on error
 */
inline int
load_and_set_remote_handle(p2p_comm::P2PComm &comm,
                           const std::string &local_tag,
                           const std::string &remote_tag,
                           std::map<std::string, void *> &remote_addr_map) {
  std::string remote_mem = load_string_from_file(
      tmp_path(remote_tag + "-" + local_tag + "-mem_handle.txt"));
  comm.set_remote_mem_handle(remote_mem);

  void *remote_ptr = read_pointer_from_file(
      tmp_path(remote_tag + "-" + local_tag + "-addr_handle.txt").c_str());
  remote_addr_map[remote_tag] = remote_ptr;

  return 0;
}

// =============================================================================
// Device-side CUDA Kernels
// =============================================================================

/**
 * @brief Initialize data buffer with a known pattern for verification
 *
 * Initializes buffer with a pattern based on client_idx and position,
 * multiplied by 1.5 to create unique float values. This pattern is used to
 * verify data correctness.
 *
 * @param buf Buffer to initialize
 * @param size Data size in bytes
 * @param client_idx Client index (for unique pattern)
 * @param alloc_buf Allocated buffer size per endpoint (0 for single endpoint)
 */
__global__ void init_data_pattern_kernel(float *buf, size_t size,
                                         int client_idx, size_t alloc_buf) {
  int block_id = blockIdx.x;
  int idx = threadIdx.x;
  int total_threads = blockDim.x;
  int num_elements = size / sizeof(float);

  // Calculate base offset based on whether multi-endpoint or single endpoint
  size_t base_offset =
      (alloc_buf > 0) ? (block_id * (alloc_buf / sizeof(float))) : 0;

  for (int i = idx; i < num_elements; i += total_threads) {
    // Use a pattern based on client_idx and position, multiplied by 1.5
    // Pattern: (client_idx + 1 + i) * 1.5 to create a unique pattern with
    // decimals
    buf[base_offset + i] = (float)(client_idx + 1 + i) * 1.5f;
  }
}

/**
 * @brief Verify data correctness based on pattern (for multi-endpoint P2P
 * style)
 *
 * Compares received data against expected pattern (original data * 1.5).
 * Expected value is the same as original data since server doesn't process it.
 *
 * @param local_buf Received data buffer
 * @param size Data size in bytes
 * @param client_idx Client index (for pattern reconstruction)
 * @param alloc_buf Allocated buffer size per endpoint (0 for single endpoint)
 * @param errors Pointer to error counter (device memory)
 */
__global__ void verify_data_pattern_kernel(float *local_buf, size_t size,
                                           int client_idx, size_t alloc_buf,
                                           int *errors) {
  int block_id = blockIdx.x;
  int idx = threadIdx.x;
  int total_threads = blockDim.x;
  int num_elements = size / sizeof(float);

  // Calculate base offset based on whether multi-endpoint or single endpoint
  size_t base_offset =
      (alloc_buf > 0) ? (block_id * (alloc_buf / sizeof(float))) : 0;

  for (int i = idx; i < num_elements; i += total_threads) {
    // Original value sent (multiplied by 1.5)
    float original = (float)(client_idx + 1 + i) * 1.5f;
    // Expected value is the same as original (no server processing)
    float expected = original;
    float received = local_buf[base_offset + i];

    // Use small epsilon for floating point comparison
    float diff = fabsf(received - expected);
    if (diff > 0.0001f) {
      int error_count = atomicAdd(errors, 1);
      // Print first few mismatches for debugging
      if (error_count < 10 && i < 100) {
        printf("Mismatch at block[%d] idx[%d]: received=%.6f, expected=%.6f "
               "(original=%.6f)\n",
               block_id, i, received, expected, original);
      }
    }
  }
}

/**
 * @brief Verify data correctness using expected buffer (for one-to-one P2P
 * style)
 *
 * Verifies that received data matches expected values (original * multiplier).
 *
 * @param local_buf Received data buffer
 * @param expected_buf Original data buffer (before processing)
 * @param size Buffer size in bytes
 * @param multiplier Multiplier applied by server (e.g., 2.0 for one-to-one
 * test)
 * @param errors Pointer to error counter (device memory)
 */
__global__ void verify_data_buffer_kernel(float *local_buf, float *expected_buf,
                                          size_t size, float multiplier,
                                          int *errors) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_elements = size / sizeof(float);

  if (idx < num_elements) {
    float received = local_buf[idx];
    float expected = expected_buf[idx] * multiplier;

    float diff = fabsf(received - expected);
    if (diff > 0.0001f) {
      atomicAdd(errors, 1);
      if (idx < 10) {
        printf("Mismatch at idx[%d]: received=%.6f, expected=%.6f, diff=%.6f\n",
               idx, received, expected, diff);
      }
    }
  }
}

/**
 * @brief Multiply buffer values by a constant (server-side processing)
 *
 * @param buf Buffer containing float values
 * @param size Buffer size in bytes
 * @param multiplier Multiplier value (e.g., 2.0f)
 */
__global__ void multiply_buffer_kernel(float *buf, size_t size,
                                       float multiplier) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size / sizeof(float)) {
    buf[idx] *= multiplier;
  }
}

#endif // P2P_UTILS_CUH_
