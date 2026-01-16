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
 * @file p2p_one_to_one.cu
 * @brief One-to-one point-to-point P2P communication test
 *
 * Demonstrates basic point-to-point communication between a single
 * client-server pair using InfiniBand GPU Direct Access (IBGDA) transport. The
 * server processes received data by multiplying values by 2, and the client
 * verifies the correctness.
 */

#include "non_abi/device/pt-to-pt/ibgda_device.cuh"
#include "p2p_utils.cuh"
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <p2pcomm_api.h>
#include <stdio.h>
#include <unistd.h>

using namespace p2p_comm;

/**
 * @brief Client kernel to send data to remote buffer using PUT operation
 *
 * @param remote_buf Remote buffer pointer
 * @param local_buf Local buffer pointer
 * @param size Data size in bytes
 * @param dspe Device-side PE ID
 * @param state IBGDA device state
 * @param flag_offset Offset for synchronization flag
 */
__global__ void p2p_client_put_kernel(void *remote_buf, void *local_buf,
                                      size_t size, int dspe,
                                      p2pcomm_ibgda_device_state_t *state,
                                      size_t flag_offset) {
  const int lane_id = threadIdx.x % 32;
  const auto warp_id = threadIdx.x / 32;
  const auto num_warps = blockDim.x / 32;
  const auto chunk_size = (size + num_warps - 1) / num_warps;
  auto offset = warp_id * chunk_size;
  auto data_size = min(chunk_size, size - offset);

  ibgda_p2p::nvshmemi_ibgda_put_nbi_warp(
      (uint64_t)remote_buf + offset, (uint64_t)local_buf + offset, chunk_size,
      dspe, warp_id, lane_id, 0, state);
  __threadfence_system();
  if (lane_id == 0) {
    int *flag_ptr =
        (int *)((char *)remote_buf + flag_offset + warp_id * sizeof(int));
    ibgda_p2p::nvshmemi_ibgda_amo_nonfetch_add(flag_ptr, 1, dspe, warp_id,
                                               state);
  }
  __threadfence_system();
}
// Note: multiply_kernel is now multiply_buffer_kernel in p2p_utils.cuh
/**
 * @brief Client kernel to receive data from remote buffer using GET operation
 *
 * @param remote_buf Remote buffer pointer
 * @param local_buf Local buffer pointer
 * @param size Data size in bytes
 * @param dspe Device-side PE ID
 * @param state IBGDA device state
 */
__global__ void p2p_get_kernel(void *remote_buf, void *local_buf, size_t size,
                               int dspe, p2pcomm_ibgda_device_state_t *state) {
  const int lane_id = threadIdx.x % 32;
  const auto warp_id = threadIdx.x / 32;
  const auto num_warps = blockDim.x / 32;
  const auto chunk_size = (size + num_warps - 1) / num_warps;
  auto offset = warp_id * chunk_size;
  auto data_size = min(chunk_size, size - offset);
  ibgda_p2p::nvshmemi_ibgda_get_nbi_warp(
      (uint64_t)remote_buf + offset, (uint64_t)local_buf + offset, chunk_size,
      dspe, warp_id, lane_id, 0, state);
  __threadfence_system();
  if (lane_id == 0) {
    auto qp = ibgda_p2p::ibgda_get_rc(dspe, warp_id, state);
    ibgda_p2p::ibgda_quiet(qp, state);
  }
  __threadfence_system();
}
/**
 * @brief Server kernel to poll and wait for data arrival
 *
 * This kernel polls the synchronization flag to wait for data to arrive from
 * client.
 *
 * @param local_buf Local buffer pointer
 * @param flag_offset Offset for synchronization flag
 */
__global__ void p2p_server_poll_kernel(void *local_buf, size_t flag_offset) {
  const int lane_id = threadIdx.x % 32;
  const auto warp_id = threadIdx.x / 32;
  if (lane_id == 0) {
    int *flag_ptr =
        (int *)((char *)local_buf + flag_offset + warp_id * sizeof(int));
    while (ibgda_p2p::ld_acquire_global(flag_ptr) != 1) {
    }
    __threadfence_system();
  }

  __threadfence_system();
}

// Note: verify_data_kernel is now verify_data_buffer_kernel in p2p_utils.cuh

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Setup P2P connection and exchange handles
 */
int setup_p2p_connection(P2PComm &comm, const std::string &self_prefix,
                         const std::string &peer_prefix) {
  auto conn_str = comm.get_local_conn_handle();
  // Export local RC connection blob; peer will mirror-read before calling
  // connect_endpoint.
  save_string_to_file(tmp_path(self_prefix + "-conn_handle.txt"), conn_str);

  sleep(10);

  std::string remote_conn =
      load_string_from_file(tmp_path(peer_prefix + "-conn_handle.txt"));
  comm.connect_endpoint(remote_conn);

  return 0;
}

/**
 * @brief Setup buffers and memory registration
 */
struct BufferSetup {
  void *buf;
  void *remote_ptr;
  float *h_original_data;
  size_t alloc_size;
  size_t data_size;
  size_t flag_offset;
};

BufferSetup setup_buffers(P2PComm &comm, bool is_client,
                          const std::string &self_prefix,
                          const std::string &peer_prefix) {
  BufferSetup setup;
  setup.alloc_size = 1 << 20;
  setup.data_size = 2048;
  setup.flag_offset = setup.alloc_size - 256;
  setup.h_original_data = nullptr;

  setup.buf = allocate_buffer(setup.alloc_size, setup.alloc_size);
  if (!setup.buf) {
    return setup;
  }

  if (is_client) {
    float *h_data = new float[setup.alloc_size / sizeof(float)];
    srand(time(NULL));
    for (size_t i = 0; i < setup.alloc_size / sizeof(float); ++i) {
      h_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    cudaMemcpy(setup.buf, h_data, setup.alloc_size, cudaMemcpyHostToDevice);
    setup.h_original_data = h_data;
  } else {
    cudaMemset(setup.buf, 0, setup.alloc_size);
  }

  const char *self_file = is_client ? "clientbuffer.bin" : "serverbuffer.bin";
  const char *peer_file = is_client ? "serverbuffer.bin" : "clientbuffer.bin";

  save_pointer_to_file(setup.buf, tmp_path(self_file).c_str());

  // Register memory (one-to-one test uses simple prefix naming)
  if (comm.register_memory(setup.buf, setup.alloc_size) != 0) {
    std::cerr << "register_memory failed.\n";
    return setup;
  }

  auto mem_handle = comm.get_local_mem_handle();
  // Write mem handle + device pointer so the peer can issue RDMA against this
  // buffer.
  save_string_to_file(tmp_path(self_prefix + "-mem_handle.txt"), mem_handle);
  sleep(5);

  // Load remote handle (one-to-one test uses simple prefix naming)
  std::string remote_mem =
      load_string_from_file(tmp_path(peer_prefix + "-mem_handle.txt"));
  comm.set_remote_mem_handle(remote_mem);
  setup.remote_ptr = read_pointer_from_file(tmp_path(peer_file).c_str());

  return setup;
}

/**
 * @brief Client: PUT data to server
 */
void client_put_data(void *remote_ptr, void *buf, size_t data_size,
                     size_t flag_offset, p2pcomm_ibgda_device_state_t *state) {
  p2p_client_put_kernel<<<1, 256>>>(remote_ptr, buf, data_size, 0, state,
                                    flag_offset);
  cudaDeviceSynchronize();
  print_first_n_values("[CLIENT] First 10 values sent", buf, 10);
}

/**
 * @brief Server: receive data from client
 */
void server_receive_data(void *buf, size_t flag_offset) {
  p2p_server_poll_kernel<<<1, 256>>>(buf, flag_offset);
  cudaDeviceSynchronize();
  print_first_n_values("[SERVER] First 10 values received (before processing)",
                       buf, 10);
}

/**
 * @brief Client: GET processed data back from server
 */
void client_get_data(void *remote_ptr, void *buf, size_t data_size,
                     p2pcomm_ibgda_device_state_t *state) {
  sleep(5);
  p2p_get_kernel<<<1, 256>>>(remote_ptr, buf, data_size, 0, state);
  cudaDeviceSynchronize();
  print_first_n_values("[CLIENT] First 10 values received back", buf, 10);
}

/**
 * @brief Server: process received data
 */
void server_process_data(void *buf, size_t data_size) {
  multiply_buffer_kernel<<<8, 256>>>((float *)buf, data_size, 2.0f);
  cudaDeviceSynchronize();
  print_first_n_values("[SERVER] First 10 values after processing (x2)", buf,
                       10);
  sleep(10);
}

/**
 * @brief Client: verify received data correctness
 */
int verify_client_data(void *buf, float *h_original_data, size_t data_size) {
  int *d_errors = nullptr;
  float *d_original_data = nullptr;

  cudaMalloc(&d_errors, sizeof(int));
  cudaMemset(d_errors, 0, sizeof(int));
  cudaMalloc(&d_original_data, data_size);
  cudaMemcpy(d_original_data, h_original_data, data_size,
             cudaMemcpyHostToDevice);

  int num_elements = data_size / sizeof(float);
  int num_blocks = (num_elements + 255) / 256;

  verify_data_buffer_kernel<<<num_blocks, 256>>>((float *)buf, d_original_data,
                                                 data_size, 2.0f, d_errors);
  cudaDeviceSynchronize();

  int errors = 0;
  cudaMemcpy(&errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost);

  if (errors == 0) {
    printf("✓ Data verification passed! All %d elements match expected values "
           "(2x original).\n",
           num_elements);
  } else {
    printf(
        "✗ Data verification failed! Found %d mismatches out of %d elements.\n",
        errors, num_elements);
  }

  cudaFree(d_original_data);
  cudaFree(d_errors);

  return errors;
}

/**
 * @brief Cleanup resources
 */
void cleanup_resources(P2PComm &comm, void *buf, float *h_original_data) {
  cleanup_single_comm(comm, buf);
  free_p2p_buffer(buf);

  if (h_original_data != nullptr) {
    delete[] h_original_data;
  }
}

// =============================================================================
// Main Function
// =============================================================================

/**
 * @brief Main function for one-to-one P2P test
 *
 * Usage: ./p2p_one_to_one [server|client]
 *
 * This test demonstrates:
 * 1. Client sends random float data to server
 * 2. Server receives data and multiplies by 2
 * 3. Client GETs processed data back
 * 4. Client verifies data correctness (should be 2x original)
 */
int main(int argc, char **argv) {
  // Parse arguments
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [server|client]" << std::endl;
    return 1;
  }

  std::string role = argv[1];
  bool is_client = (role == "client");
  bool is_server = (role == "server");

  if (!is_client && !is_server) {
    std::cerr << "Unknown role: " << role << std::endl;
    return 1;
  }

  // Setup device and prefixes
  std::string self_prefix = is_client ? "client" : "server";
  std::string peer_prefix = is_client ? "server" : "client";
  cudaSetDevice(is_client ? 0 : 1);

  // Initialize P2PComm
  P2PComm comm(is_client);
  if (comm.init() != 0) {
    std::cerr << "Failed to init P2PComm.\n";
    return -1;
  }

  // Setup connection
  if (setup_p2p_connection(comm, self_prefix, peer_prefix) != 0) {
    return -1;
  }

  // Setup buffers
  BufferSetup setup = setup_buffers(comm, is_client, self_prefix, peer_prefix);
  if (!setup.buf) {
    return -1;
  }

  p2pcomm_ibgda_device_state_t *state = comm.get_device_state_d();

  // Communication phase
  if (is_client) {
    client_put_data(setup.remote_ptr, setup.buf, setup.data_size,
                    setup.flag_offset, state);
    client_get_data(setup.remote_ptr, setup.buf, setup.data_size, state);
  } else {
    server_receive_data(setup.buf, setup.flag_offset);
    server_process_data(setup.buf, setup.data_size);
  }

  // Verification (client only)
  if (is_client && setup.h_original_data != nullptr) {
    verify_client_data(setup.buf, setup.h_original_data, setup.data_size);
  }

  // Cleanup
  cleanup_resources(comm, setup.buf, setup.h_original_data);

  return 0;
}