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
 * @file p2pcomm_api.h
 * @brief P2P Communication entry API definitions
 *
 */
#ifndef _P2PCOMM_API_H_
#define _P2PCOMM_API_H_

#include "device_host_transport/nvshmem_common_ibgda.h"
#include "internal/host_transport/nvshmemi_transport_defines.h"
#include "internal/host_transport/transport.h"
#include <cstddef>
#include <string>

namespace p2p_comm {
/**
 * client/Server workflow
 *
 * Initialization:
 * 1. call init() to initialize the P2P communication module.
 *
 * Connection establishment:
 * 2. call get_local_conn_handle() to get local connection handle.
 * 3. exchange conn_handle with server by external means.
 * 4. call connect_endpoint(remote_conn_handle) to connect to the server.
 *
 * Memory registration:
 * 5. call p2pcomm_malloc() to alloc memory.
 * 6. call register_memory() to register the memory with the P2P communication
 * module.
 * 7. call get_local_mem_handle() to get local memory handle.
 * 8. exchange mem_handle with server by external means.
 * 9. call set_remote_mem_handle() to set remote memory handle.
 *
 * Communication:
 * 10. exchange registered memory addresses with server by external means.
 * 11. use get_device_state_d() to get device state pointer on device side.
 * 12. client and server now are ready to do P2P communications.
 *
 * Cleanup:
 * 13. call deregister_memory() to deregister memory.
 * 14. call fini() to finalize the P2P communication module.
 * 15. call p2pcomm_free() to free memory.
 */

/**
 * @brief Allocate memory with specified size and alignment.
 * @param size Size of the memory to allocate.
 * @param alignment Alignment of the memory to allocate.
 * @return Pointer to the allocated memory. nullptr on failure.
 *
 * The allocated memory is not bound to any specific communication context.
 * The caller is responsible for freeing the memory explicitly using the
 * p2pcomm_free() function.
 */
void *p2pcomm_malloc(size_t size, size_t alignment);

/**
 * @brief Free the memory allocated by p2pcomm_malloc.
 * @param dptr Pointer to the memory to free.
 * @return 0 on success, non-zero on failure.
 */
int p2pcomm_free(void *dptr);

/**
 * @brief A struct to represent a memory handle.
 * This struct is a raw binary string which is opaque to the user.
 * The underlying data is the serialized form of real mem_handle_t struct.
 */
typedef nvshmem_mem_handle_t p2pcomm_mem_handle_t;

/**
 * @brief P2P Communication class.
 * This class provides an interface for peer-to-peer communication using IBGDA
 * transport.
 * Note on the "final" specifier:
 *  - The class is declared "final", which prevents inheritance. It signals that
 *    the class is not intended to be a base class and prohibits deriving other
 *    classes from it (helps the compiler optimize and avoid virtual dispatch
 *    surprises).
 */

class P2PComm final {
public:
  P2PComm(bool isClient) {
    if (isClient)
      pe = 1;
  }
  ~P2PComm() = default;

  P2PComm(const P2PComm &) = delete;
  P2PComm &operator=(const P2PComm &) = delete;

  P2PComm(P2PComm &&) = delete;
  P2PComm &operator=(P2PComm &&) = delete;

  /**
   * @brief Initialize the P2P communication object.
   * @return 0 on success, non-zero on failure.
   *
   * It should be called before any other P2P communication functions are used.
   * This function initializes both local & global resources / settings.
   */
  int init();

  /**
   * @brief Register a memory region for P2P communication.
   * @param dptr Pointer to the memory region to register.
   * @param size Size of the memory region to register.
   * @return 0 on success, non-zero on failure.
   */
  int register_memory(void *dptr, size_t size);

  /**
   * @brief Deregister a memory region from P2P communication.
   * @param dptr Pointer to the memory region to deregister.
   * @return 0 on success, non-zero on failure.
   */
  int deregister_memory(void *dptr);

  /**
   * @brief Get the local memory handle for the registered memory.
   * @return The local memory handle as a string.
   */
  std::string get_local_mem_handle() const;

  /**
   * @brief Set the remote memory handle for the registered memory.
   * @param mem_handle The remote memory handle as a string.
   * @return 0 on success, non-zero on failure.
   *
   * the mem_handle string should be obtained from the remote peer untouched.
   */
  int set_remote_mem_handle(const std::string &mem_handle);

  /**
   * @brief Get the local connection handle for establishing connection.
   * @return The local connection handle as a string.
   */
  std::string get_local_conn_handle() const;

  /**
   * @brief Connect to the remote endpoint using the provided connection handle.
   * @param conn_handle The remote connection handle as a string.
   * @return 0 on success, non-zero on failure.
   *
   * the conn_handle string should be obtained from the remote peer untouched.
   */
  int connect_endpoint(const std::string &conn_handle);

  /**
   * @brief Get the device state pointer on the device.
   * @return Pointer to the device state on the device.
   *
   * the function should be called after memory registration and connection
   * establishment.
   */
  p2pcomm_ibgda_device_state_t *get_device_state_d() {
    update_device_state_d();
    return p2pcomm_device_state_d;
  }

  /**
   * @brief Finalize the P2P communication object.
   * @return 0 on success, non-zero on failure.
   *
   * It should be called to release all resources allocated by the P2P
   * communication object.
   */
  int fini();

private:
  /**
   * @brief Update the device state on the device.
   * @return 0 on success, non-zero on failure.
   *
   * A helper function to update the device state on the device.
   */
  int update_device_state_d();

private:
  bool initialized = false;
  // rc_handles* for connection handle data
  void *rc_handles = nullptr;
  int rc_handles_len = 0;
  // memory handle for registered memory
  p2pcomm_mem_handle_t mem_handle[2]; // idx = 0 pe = 0; idx = 1 pe = 1

  nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state;
  // Ptr to GPU buffer. CPU cannot dereference.
  p2pcomm_ibgda_device_state_t *p2pcomm_device_state_d = nullptr;
  nvshmem_transport_t trans = nullptr;
  int pe = 0;          // server 0; client 1
  int num_dci_eps = 0; // number of dci eps
};
} // end of namespace p2p_comm

#endif // end of _P2PCOMM_API_H_
