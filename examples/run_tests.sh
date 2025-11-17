#!/bin/bash
# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @file run_tests.sh
# @brief Launch script for P2P communication tests
#
# This script supports multiple test configurations:
# 1. Unit test: Single client-server pair (1v1)
# 2. Multi-endpoint test: 2 servers + 2 clients (2v2)
#
# Usage:
#   ./run_tests.sh one_to_one   # Run one-to-one P2P test (1v1)
#   ./run_tests.sh multi        # Run 2v2 multi-endpoint test
#   ./run_tests.sh              # Run both tests sequentially
#
# Prerequisites:
# - Compiled binaries (p2p_one_to_one, p2p_multi) in current directory
# - Proper InfiniBand HCA configuration
# - Required environment variables are set by this script
#

# Common environment variables for IBGDA transport
export NVSHMEM_IB_DISABLE_DMABUF=1
#export NVSHMEM_DEBUG=TRACE
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export LD_LIBRARY_PATH=../build/src/lib/:$LD_LIBRARY_PATH
# =============================================================================
# One-to-One P2P Test: Single client-server pair (1v1)
# =============================================================================
run_one_to_one_test() {
    echo "=========================================="
    echo "Starting One-to-One P2P Test: 1 server + 1 client"
    echo "=========================================="
      
    # Start server (using mlx5_0 HCA)
    export NVSHMEM_HCA_LIST=mlx5_0 && ./build/p2p_one_to_one server >server.log 2>&1 &
    
    # Start client (using mlx5_1 HCA)
    export NVSHMEM_HCA_LIST=mlx5_1 && ./build/p2p_one_to_one client >client.log 2>&1 &
    
    wait
    echo "One-to-one test completed. Check server.log and client.log for results."
    echo ""
}

# =============================================================================
# Multi-Endpoint Test: 2 servers + 2 clients (2v2)
# =============================================================================
run_multi_test() {
    echo "=========================================="
    echo "Starting Multi-Endpoint Test: 2 servers + 2 clients"
    echo "=========================================="
    
    # Start 2 servers (using different HCAs to avoid conflict)
    export NVSHMEM_HCA_LIST=mlx5_0 && ./build/p2p_multi server 0 s1 0 c1,c2 >s1.log 2>&1 &
    export NVSHMEM_HCA_LIST=mlx5_1 && ./build/p2p_multi server 1 s2 0 c1,c2 >s2.log 2>&1 &
    
    # Start 2 clients (using different HCAs to avoid conflict)
    export NVSHMEM_HCA_LIST=mlx5_2 && ./build/p2p_multi client 2 c1 0 s1,s2 >c1.log 2>&1 &
    export NVSHMEM_HCA_LIST=mlx5_3 && ./build/p2p_multi client 3 c2 1 s1,s2 >c2.log 2>&1 &
    
    wait
    echo "2v2 test completed. Check s1.log, s2.log, c1.log, and c2.log for results."
    echo ""
}

# =============================================================================
# Main
# =============================================================================
case "$1" in
    one_to_one|1v1|p2p_one_to_one)
        run_one_to_one_test
        ;;
    multi|2v2)
        run_multi_test
        ;;
    "")
        # Run both tests if no argument provided
        run_one_to_one_test
        run_multi_test
        echo "All tests completed."
        ;;
    *)
        echo "Usage: $0 [one_to_one|multi]"
        echo ""
        echo "Options:"
        echo "  one_to_one, 1v1, p2p_one_to_one - Run one-to-one P2P test (1v1)"
        echo "  multi, 2v2                      - Run multi-endpoint test (2v2)"
        echo "  (no argument)                    - Run both tests sequentially"
        exit 1
        ;;
esac

