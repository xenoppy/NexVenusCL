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
 * @file p2p_multi.cu
 * @brief Multi-endpoint P2P communication test
 *
 * Benchmarks point-to-point communication latency between multiple GPU
 * endpoints using InfiniBand GPU Direct Access (IBGDA) transport.
 */

#include "non_abi/device/pt-to-pt/ibgda_device.cuh"
#include "p2p_utils.cuh"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <p2pcomm_api.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace p2p_comm;

// =============================================================================
// Constants
// =============================================================================

constexpr size_t DEFAULT_BUFFER_SIZE = 35 * 10 << 20;
constexpr size_t BUFFER_ALIGNMENT = 1 << 20;
constexpr size_t ALLOC_BUF_PER_ENDPOINT = 10 << 20;
constexpr int WARMUP_ITERATIONS = 10;
constexpr int BENCHMARK_ITERATIONS = 200;

// =============================================================================
// CUDA Kernels
// =============================================================================

/**
 * @brief Client kernel to send data to multiple remote endpoints using PUT
 * operation
 *
 * This kernel sends data from local buffer to remote buffers across multiple
 * endpoints. Each block handles one endpoint. After PUT, it signals completion
 * via atomic operation and waits for server acknowledgment.
 *
 * @param remote_bufs Array of remote buffer pointers (one per endpoint)
 * @param local_buf Local buffer pointer
 * @param size Data size in bytes
 * @param dspe Device-side PE ID
 * @param alloc_buf Allocated buffer size per endpoint
 * @param states Array of IBGDA device states (one per endpoint)
 * @param iteration Current iteration number
 */
__global__ void p2p_client_put_kernel(void **remote_bufs, void *local_buf,
                                      size_t size, int dspe, size_t alloc_buf,
                                      p2pcomm_ibgda_device_state_t **states,
                                      int iteration) {
  const int block_id = blockIdx.x;
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  const int num_warps = blockDim.x / 32;
  size_t chunk_size = (size + num_warps - 1) / num_warps;
  size_t offset = warp_id * chunk_size;
  auto remote_buf = remote_bufs[block_id];
  p2pcomm_ibgda_device_state_t *state = states[block_id];
  size_t data_size;

  if (offset < size) {
    data_size = min(chunk_size, size - offset);
    ibgda_p2p::nvshmemi_ibgda_put_nbi_warp(
        (uint64_t)remote_buf + offset,
        (uint64_t)local_buf + block_id * alloc_buf + offset, data_size, dspe,
        warp_id, lane_id, 0, state);
  }

  __threadfence_system();

  if (lane_id == 0) {
    ibgda_p2p::nvshmemi_ibgda_amo_nonfetch_add((int *)remote_buf + warp_id + 1 +
                                                   size / sizeof(int),
                                               1, dspe, warp_id, state);
  }
  __threadfence_system();

  if (warp_id == 0 && lane_id == 0) {
    float *f_recv_buf = (float *)((char *)local_buf + block_id * alloc_buf);
    while (f_recv_buf[size / 4] != 1 + iteration) {
      ibgda_p2p::nvshmemi_ibgda_get_nbi_warp(
          (uint64_t)remote_buf + size,
          (uint64_t)local_buf + block_id * alloc_buf + size, 4, dspe, warp_id,
          lane_id, 0, state);
      __threadfence_system();
      auto qp = ibgda_p2p::ibgda_get_rc(dspe, 0, state);
      ibgda_p2p::ibgda_quiet(qp, state);
    }
  }
}

/**
 * @brief Client kernel to receive data from multiple remote endpoints using GET
 * operation
 *
 * This kernel retrieves data from remote buffers to local buffer across
 * multiple endpoints. Each block handles one endpoint. Data is fetched in
 * chunks per warp.
 *
 * @param remote_bufs Array of remote buffer pointers (one per endpoint)
 * @param local_buf Local buffer pointer
 * @param size Data size in bytes
 * @param dspe Device-side PE ID
 * @param alloc_buf Allocated buffer size per endpoint
 * @param states Array of IBGDA device states (one per endpoint)
 * @param iteration Current iteration number (unused, kept for consistency)
 */
__global__ void p2p_get_kernel(void **remote_bufs, void *local_buf, size_t size,
                               int dspe, size_t alloc_buf,
                               p2pcomm_ibgda_device_state_t **states,
                               int iteration) {
  const int block_id = blockIdx.x;
  const int lane_id = threadIdx.x % 32;
  const auto warp_id = threadIdx.x / 32;
  const auto num_warps = blockDim.x / 32;
  const auto chunk_size = (size + num_warps - 1) / num_warps;
  auto offset = warp_id * chunk_size;
  auto data_size = min(chunk_size, size - offset);
  auto remote_buf = remote_bufs[block_id];
  p2pcomm_ibgda_device_state_t *state = states[block_id];

  ibgda_p2p::nvshmemi_ibgda_get_nbi_warp(
      (uint64_t)remote_buf + offset,
      (uint64_t)local_buf + block_id * alloc_buf + offset, data_size, dspe,
      warp_id, lane_id, 0, state);
  __threadfence_system();

  if (lane_id == 0) {
    auto qp = ibgda_p2p::ibgda_get_rc(dspe, 0, state);
    ibgda_p2p::ibgda_quiet(qp, state);
  }
  __threadfence_system();
}

/**
 * @brief Server kernel to poll and wait for data arrival, then signal
 * completion
 *
 * This kernel polls synchronization flags to wait for data to arrive from
 * client (via PUT operation), ensures all warps have completed their transfers,
 * and then signals completion back to client.
 *
 * @param local_buf Local buffer pointer
 * @param alloc_buf Allocated buffer size per endpoint
 * @param size Data size in bytes
 * @param iteration Current iteration number
 */
__global__ void p2p_server_poll_kernel(void *local_buf, size_t alloc_buf,
                                       size_t size, int iteration) {
  const int block_id = blockIdx.x;
  const int lane_id = threadIdx.x % 32;
  const auto warp_id = threadIdx.x / 32;
  const int num_warps = blockDim.x / 32;

  // Wait for data to arrive (only lane 0 polls)
  if (lane_id == 0) {
    while (ibgda_p2p::ld_acquire_global(
               (int *)local_buf + (block_id * int(alloc_buf)) / 4 +
               size / sizeof(int) + warp_id + 1) != iteration + 1) {
      // Polling...
    }
  }

  // Ensure data is visible to all threads
  __threadfence_system();
  __syncthreads();

  // Wait for all warps to complete their PUT operations
  // Check all warp flags to ensure complete data transfer
  if (lane_id == 0) {
    for (int w = 0; w < num_warps; w++) {
      while (ibgda_p2p::ld_acquire_global(
                 (int *)local_buf + (block_id * int(alloc_buf)) / 4 +
                 size / sizeof(int) + w + 1) != iteration + 1) {
        // Wait for all warps...
      }
    }
  }

  // Ensure all data is visible
  __threadfence_system();
  __syncthreads();

  // Set flag after data is received (only lane 0)
  if (lane_id == 0) {
    ((float *)((char *)local_buf + block_id * alloc_buf))[size / 4] =
        iteration + 1;
  }
  __threadfence_system();
}

// Note: init_data_pattern_kernel and verify_data_kernel are now in
// p2p_utils.cuh

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Initialize P2P connections for multiple endpoints
 */
int setup_connections(const std::vector<std::string> &remote_tag_vec,
                      const std::string &local_tag, bool is_client,
                      std::map<std::string, P2PComm *> &comm_map) {
  for (const auto &remote_tag : remote_tag_vec) {
    P2PComm *pcomm = new P2PComm(is_client);
    if (pcomm->init() != 0) {
      std::cerr << "Failed to init P2PComm for [" << remote_tag << "]"
                << std::endl;
      return -1;
    }

    auto conn_str = pcomm->get_local_conn_handle();
    // Persist the connection blob per peer; remote side reads its counterpart
    // and calls connect_endpoint to finalize RC links.
    save_string_to_file(
        tmp_path(local_tag + "-" + remote_tag + "-conn_handle.txt"), conn_str);
    comm_map[remote_tag] = pcomm;
  }
  return 0;
}

/**
 * @brief Register memory and exchange handles with remote endpoints
 */
int register_and_exchange_handles(
    std::map<std::string, P2PComm *> &comm_map, const std::string &local_tag,
    void *buf, size_t size, std::map<std::string, void *> &remote_addr_map) {
  // Register memory
  for (const auto &[remote_tag, pcomm] : comm_map) {
    std::string remote_conn = load_string_from_file(
        tmp_path(remote_tag + "-" + local_tag + "-conn_handle.txt"));
    pcomm->connect_endpoint(remote_conn);

    if (register_memory_and_save_handles(*pcomm, buf, size, local_tag,
                                         remote_tag) != 0) {
      return -1;
    }
  }

  sleep(10);

  // Exchange handles
  for (const auto &[remote_tag, pcomm] : comm_map) {
    if (load_and_set_remote_handle(*pcomm, local_tag, remote_tag,
                                   remote_addr_map) != 0) {
      return -1;
    }
  }
  return 0;
}

/**
 * @brief Device parameters structure
 */
struct DeviceParams {
  void **d_remote_ptrs;
  p2pcomm_ibgda_device_state_t **d_states;
  size_t num;
};

/**
 * @brief Prepare device parameters for kernel execution
 */
DeviceParams
prepare_device_params(const std::map<std::string, P2PComm *> &comm_map,
                      const std::map<std::string, void *> &remote_addr_map,
                      int client_idx) {
  DeviceParams params;
  params.num = comm_map.size();
  std::vector<void *> h_remote_ptrs;
  std::vector<p2pcomm_ibgda_device_state_t *> h_states;

  for (const auto &[remote_tag, pcomm] : comm_map) {
    void *offset_ptr = (char *)remote_addr_map.at(remote_tag) +
                       (client_idx * ALLOC_BUF_PER_ENDPOINT);
    h_remote_ptrs.push_back(offset_ptr);
    h_states.push_back(pcomm->get_device_state_d());
  }

  cudaMalloc(&params.d_remote_ptrs, params.num * sizeof(void *));
  cudaMalloc(&params.d_states,
             params.num * sizeof(p2pcomm_ibgda_device_state_t *));
  cudaMemcpy(params.d_remote_ptrs, h_remote_ptrs.data(),
             params.num * sizeof(void *), cudaMemcpyHostToDevice);
  cudaMemcpy(params.d_states, h_states.data(),
             params.num * sizeof(p2pcomm_ibgda_device_state_t *),
             cudaMemcpyHostToDevice);

  return params;
}

/**
 * @brief Run benchmarks for all buffer sizes
 */
void run_benchmarks(void **d_remote_ptrs, void *buf,
                    p2pcomm_ibgda_device_state_t **d_states, size_t num,
                    bool is_client, int client_idx, int *d_errors) {
  // 设置总迭代次数（包含预热期）
  int iteration = BENCHMARK_ITERATIONS; 
  
  // 外层循环：遍历数据包大小，从 8 字节开始，每次翻倍，直到 8MB (8 << 20)
  for (int buf_size = 8; buf_size <= (8 << 20); buf_size *= 2) {
    float total_ms = 0.0; // 用于累计非预热迭代的总耗时
    
    // 根据数据大小动态计算线程数：小于 32 字节用 32 线程，大则取 buf_size 与 256 的最小值
    int threads = (buf_size <= 32) ? 32 : min(256, buf_size); 

    // 仅客户端：在测试该数据大小时，先初始化本地 buffer 的数据模式以供后续比对
    if (is_client) {
      init_data_pattern_kernel<<<num, 256>>>((float *)buf, buf_size, client_idx,
                                             ALLOC_BUF_PER_ENDPOINT);
      cudaDeviceSynchronize(); // 等待 GPU 初始化完成
    }

    // 执行多次迭代进行压测
    for (int i = 0; i < iteration; ++i) {
      if (is_client) { // 客户端逻辑：执行数据传输并计时
        cudaEvent_t start, stop;
        cudaEventCreate(&start); // 创建计时开始事件
        cudaEventCreate(&stop);  // 创建计时结束事件
        cudaEventRecord(start); // 记录开始时间

        // 调用 PUT Kernel：将本地数据推送到所有远程端点
        p2p_client_put_kernel<<<num, threads>>>(d_remote_ptrs, buf, buf_size, 0,
                                                ALLOC_BUF_PER_ENDPOINT,
                                                d_states, i);
        // 调用 GET Kernel：从所有远程端点拉回数据
        p2p_get_kernel<<<num, threads>>>(d_remote_ptrs, buf, buf_size, 0,
                                         ALLOC_BUF_PER_ENDPOINT, d_states, i);

        cudaEventRecord(stop); // 记录结束时间
        cudaDeviceSynchronize(); // 关键同步：确保 GPU 任务完成

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop); // 计算本次迭代的毫时

        // 排除掉前几轮的预热记录（WARMUP_ITERATIONS），只累加稳定期的时间
        if (i >= WARMUP_ITERATIONS) {
          total_ms += ms;
        }

        cudaEventDestroy(start); // 销毁计时资源
        cudaEventDestroy(stop);
      } else { 
        // 服务端逻辑：不计时，只负责轮询/响应客户端的事务
        p2p_server_poll_kernel<<<num, threads>>>(buf, ALLOC_BUF_PER_ENDPOINT,
                                                 buf_size, i);
      }
    }

    // 确保该数据大小下的所有 GPU 操作彻底完成
    cudaDeviceSynchronize();

    // 仅客户端：验证数据正确性
    if (is_client) {
      cudaMemset(d_errors, 0, sizeof(int)); // 清零 GPU 端错误计数器
      // 检查 GET 回来的数据是否与预期模式一致
      verify_data_pattern_kernel<<<num, 256>>>(
          (float *)buf, buf_size, client_idx, ALLOC_BUF_PER_ENDPOINT, d_errors);
      cudaDeviceSynchronize();

      int errors = 0;
      // 将 GPU 计算的错误数拷贝回主机内存
      cudaMemcpy(&errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost);

      // 计算有效迭代的平均耗时
      float avg_ms = total_ms / (iteration - WARMUP_ITERATIONS);
      
      // 输出该包大小的测试结果（耗时及是否通过）
      if (errors == 0) {
        std::cout << buf_size << "\t\t" << avg_ms << "\t\t✓ PASS" << std::endl;
      } else {
        std::cout << buf_size << "\t\t" << avg_ms << "\t\t✗ FAIL (" << errors
                  << " errors)" << std::endl;
      }
    }
  }
}

/**
 * @brief Cleanup resources
 */
void cleanup_resources(std::map<std::string, P2PComm *> &comm_map, void *buf,
                       void **d_remote_ptrs,
                       p2pcomm_ibgda_device_state_t **d_states, int *d_errors) {
  if (d_errors) {
    cudaFree(d_errors);
  }
  if (d_states) {
    cudaFree(d_states);
  }
  if (d_remote_ptrs) {
    cudaFree(d_remote_ptrs);
  }

  // Cleanup all comms (deregister memory but don't free buffer yet)
  for (const auto &[remote_tag, pcomm] : comm_map) {
    cleanup_single_comm(*pcomm, buf);
    delete pcomm;
  }

  // Free buffer once after all comms are cleaned up
  free_p2p_buffer(buf);
}

// =============================================================================
// Main
// =============================================================================

// Usage: ./p2p_multi role gpu_device_id local_tag client_index remote_tag_list
// Example: ./p2p_multi client 5 c2 1 s1,s2,s3,s4
int main(int argc, char **argv) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " [server|client] gpu_device_id local_tag client_index "
                 "remote_tag_list"
              << std::endl;
    return 1;
  }

  std::string role = argv[1];
  bool is_client = (role == "client");
  bool is_server = (role == "server");

  if (!is_client && !is_server) {
    std::cerr << "Unknown role: " << role << std::endl;
    return 1;
  }

  int gpu_device_id = std::atoi(argv[2]);
  std::string local_tag = argv[3];
  int client_idx = std::atoi(argv[4]);
  std::string remote_tag_list = argv[5];
  std::vector<std::string> remote_tag_vec =
      split_string_by_comma(remote_tag_list);

  std::cout << "Local device: [" << gpu_device_id << "], tag: [" << local_tag
            << "], index: [" << client_idx << "], remote tags: ["
            << remote_tag_list << "]" << std::endl;

  cudaError_t err = cudaSetDevice(gpu_device_id);
  if (err != cudaSuccess) {
    std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err)
              << std::endl;
    return -1;
  }

  // Initialize connections
  std::map<std::string, P2PComm *> comm_map;
  if (setup_connections(remote_tag_vec, local_tag, is_client, comm_map) != 0) {
    return -1;
  }

  sleep(5);

  // Allocate memory
  size_t size = DEFAULT_BUFFER_SIZE;
  void *buf = allocate_buffer(size, BUFFER_ALIGNMENT);
  if (!buf) {
    cleanup_resources(comm_map, buf, nullptr, nullptr, nullptr);
    return -1;
  }

  // Register memory and exchange handles
  std::map<std::string, void *> remote_addr_map;
  if (register_and_exchange_handles(comm_map, local_tag, buf, size,
                                    remote_addr_map) != 0) {
    cleanup_resources(comm_map, buf, nullptr, nullptr, nullptr);
    return -1;
  }

  // Prepare device parameters
  DeviceParams params =
      prepare_device_params(comm_map, remote_addr_map, client_idx);
  sleep(5);

  // Allocate device memory for error counting (only for client)
  int *d_errors = nullptr;
  if (is_client) {
    cudaMalloc(&d_errors, sizeof(int));
  }

  // Run benchmarks
  run_benchmarks(params.d_remote_ptrs, buf, params.d_states, params.num,
                 is_client, client_idx, d_errors);

  // Cleanup
  cleanup_resources(comm_map, buf, params.d_remote_ptrs, params.d_states,
                    d_errors);

  std::cout << "Done." << std::endl;
  return 0;
}
