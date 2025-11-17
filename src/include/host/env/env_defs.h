/****
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
 * Copyright (c) 2016-2023, NVIDIA Corporation.  All rights reserved.
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

/* NVSHMEMI_ENV_DEF( name, kind, default, category, short description )
 *
 * Kinds: long, size, bool, string
 * Categories: NVSHMEMI_ENV_CAT_OPENSHMEM, NVSHMEMI_ENV_CAT_OTHER,
 *             NVSHMEMI_ENV_CAT_TRANSPORT,
 *             NVSHMEMI_ENV_CAT_HIDDEN
 */

#include <stddef.h> // for size_t
#ifndef NVSHMEM_ENV_DEFS_INTERNAL
#include "bootstrap_host_transport/env_defs_internal.h" // IWYU pragma: keep
#endif
#include "non_abi/nvshmem_build_options.h" // for NVSHMEM_IBGDA_SUPPORT
#include "non_abi/nvshmem_version.h"

#define ENV_DEFS_STRINGIFY(x) #x
#define ENV_DEFS_TOSTRING(x) ENV_DEFS_STRINGIFY(x)

#ifdef NVSHMEMI_ENV_DEF

NVSHMEMI_ENV_DEF(VERSION, bool, false, NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Print library version at startup")
NVSHMEMI_ENV_DEF(INFO, bool, false, NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Print environment variable options at startup")
NVSHMEMI_ENV_DEF(INFO_HIDDEN, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Print hidden environment variable options at startup")

NVSHMEMI_ENV_DEF(DEBUG, string, "", NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Set to enable debugging messages.\n"
                 "Optional values: VERSION, WARN, INFO, ABORT, TRACE")

/** Library initialization **/
NVSHMEMI_ENV_DEF(CUDA_PATH, string, "", NVSHMEMI_ENV_CAT_OTHER,
                 "Path to directory containing libcuda.so (for use when not in "
                 "default location)")

/** Debugging **/
NVSHMEMI_ENV_DEF(DEBUG_SUBSYS, string, "", NVSHMEMI_ENV_CAT_HIDDEN,
                 "Comma separated list of debugging message sources. Prefix "
                 "with '^' to exclude.\n"
                 "Values: INIT, P2P, TRANSPORT, MEM, UTIL, ALL")
NVSHMEMI_ENV_DEF(
    DEBUG_FILE, string, "", NVSHMEMI_ENV_CAT_OTHER,
    "Debugging output filename, may contain %h for hostname and %p for pid")

/** Transport **/
NVSHMEMI_ENV_DEF(
    ENABLE_NIC_PE_MAPPING, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
    "When not set or set to 0, a PE is assigned the NIC on the node that is "
    "closest to it by distance. When set to 1, NVSHMEM either assigns NICs to "
    "PEs on a round-robin basis or uses ``NVSHMEM_HCA_PE_MAPPING`` or "
    "``NVSHMEM_HCA_LIST`` when they are specified.")
#if defined(NVSHMEM_IBGDA_SUPPORT) || defined(NVSHMEM_ENV_ALL)
/** GPU-initiated communication **/
NVSHMEMI_ENV_DEF(IB_ENABLE_IBGDA, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Set to enable GPU-initiated communication transport.")
#endif

#endif
