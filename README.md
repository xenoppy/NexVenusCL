# Introduction

NexVenusCL is a one-sided, GPU-initiated peer-to-peer communication library built on IBGDA (InfiniBand GPUDirect Async) technology. Unlike the InfiniBand Reliable Connection (IBRC) model, which relies on a host proxy thread, NexVenusCL enables CUDA kernels to directly post work requests to the RDMA work queue (WQ) and poll completions from the completion queue (CQ) using the RDMA doorbell mechanism.

By providing warp-level non-blocking PUT/GET/SIGNAL programming interfaces, NexVenusCL eliminates the need for a host proxy thread to manage RDMA operations. This allows application CUDA kernels to perform other computations during ongoing communications, rather than stalling for CPU-triggered RDMA operations. As a result, NexVenusCL improves efficiency and reduces latency for fine-grained communication across inter-node GPUs. Additionally, send and receive buffers can be dynamically allocated, destroyed, and registered with the RNIC, offering greater flexibility in memory management.

NexVenusCL has been successfully deployed in MoE (Mixture of Experts) inference application [[arXiv:2509.17863](https://arxiv.org/abs/2509.17863)]. In an Expert-as-a-Service (EaaS) setting, where attention and expert components in an MoE layer are disaggregated across different GPUs, NexVenusCL enables efficient token transfer between attention clients and expert servers through its high-performance communication interfaces.

If highly efficient and effective collective communication lib is required, please refer to [VCCL](https://github.com/sii-research/VCCL/). 

## Performance

We evaluated the peer-to-peer (P2P) communication latency under various configurations. The following results show the measured latencies (in milliseconds, ms) for different sender-receiver group sizes. Each row corresponds to a message size, doubling from 4 Bytes up to 8 MB.

| Message Size | 8 vs 8 | 16 vs 16 | 32 vs 32 | 8 vs 16 | 16 vs 8 | 16 vs 32 | 32 vs 16 | 8 vs 32 | 32 vs 8 |
|--------------|--------|----------|----------|---------|---------|----------|----------|---------|---------|
| 4B           | 0.070  | 0.071    | 0.077    | 0.069   | 0.072   | 0.083    | 0.085    | 0.084   | 0.081   |
| 8B           | 0.070  | 0.073    | 0.079    | 0.070   | 0.071   | 0.085    | 0.087    | 0.073   | 0.081   |
| 16B          | 0.070  | 0.072    | 0.079    | 0.070   | 0.076   | 0.084    | 0.085    | 0.074   | 0.081   |
| 32B          | 0.070  | 0.072    | 0.079    | 0.070   | 0.070   | 0.076    | 0.094    | 0.074   | 0.082   |
| 64B          | 0.071  | 0.073    | 0.075    | 0.071   | 0.071   | 0.075    | 0.086    | 0.109   | 0.081   |
| 128B         | 0.056  | 0.055    | 0.063    | 0.061   | 0.071   | 0.076    | 0.088    | 0.073   | 0.083   |
| 256B         | 0.058  | 0.058    | 0.060    | 0.058   | 0.071   | 0.076    | 0.083    | 0.083   | 0.081   |
| 512B         | 0.058  | 0.057    | 0.060    | 0.058   | 0.070   | 0.076    | 0.076    | 0.082   | 0.079   |
| 1KB          | 0.058  | 0.058    | 0.060    | 0.063   | 0.070   | 0.076    | 0.076    | 0.073   | 0.082   |
| 2KB          | 0.062  | 0.058    | 0.060    | 0.058   | 0.070   | 0.087    | 0.082    | 0.075   | 0.082   |
| 4KB          | 0.057  | 0.058    | 0.067    | 0.059   | 0.071   | 0.078    | 0.089    | 0.076   | 0.083   |
| 8KB          | 0.057  | 0.059    | 0.073    | 0.059   | 0.071   | 0.083    | 0.090    | 0.080   | 0.081   |
| 16KB         | 0.058  | 0.061    | 0.067    | 0.061   | 0.073   | 0.088    | 0.091    | 0.085   | 0.084   |
| 32KB         | 0.061  | 0.066    | 0.078    | 0.066   | 0.077   | 0.099    | 0.091    | 0.096   | 0.088   |
| 64KB         | 0.065  | 0.078    | 0.110    | 0.079   | 0.087   | 0.121    | 0.115    | 0.118   | 0.105   |
| 128KB        | 0.079  | 0.102    | 0.157    | 0.101   | 0.109   | 0.164    | 0.177    | 0.161   | 0.157   |
| 256KB        | 0.104  | 0.146    | 0.259    | 0.147   | 0.152   | 0.250    | 0.260    | 0.245   | 0.241   |
| 512KB        | 0.147  | 0.232    | 0.420    | 0.229   | 0.240   | 0.418    | 0.421    | 0.418   | 0.410   |
| 1MB          | 0.233  | 0.402    | 0.768    | 0.400   | 0.407   | 0.778    | 1.727    | 0.762   | 1.294   |
| 2MB          | 0.404  | 0.746    | 1.434    | 0.744   | 1.720   | 1.450    | 2.059    | 1.450   | 2.142   |
| 4MB          | 0.747  | 1.438    | 2.806    | 1.429   | 1.960   | 2.824    | 9.078    | 2.826   | 8.787   |
| 8MB          | 1.432  | 2.807    | 5.555    | 2.804   | 4.260   | 5.573    | 18.673   | 5.573   | 19.496  |

## Quick start

### Requirements

#### GDR (GPUDirect RDMA)
You must use the `nvidia_peermem` kernel module for GPUDirect RDMA support.
  
  Example:
  ```bash
  lsmod | grep nvidia_peermem
  ```
  
  If the module is not loaded, run:
  ```bash
  sudo modprobe nvidia_peermem
  ```


---

#### Configure NVIDIA driver

This configuration enables traditional IBGDA support.

Modify `/etc/modprobe.d/nvidia.conf`:

```
options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"
```

Update kernel configuration:

```
sudo update-initramfs -u
sudo reboot
```


### Build from Source

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd p2p_ibgda
```

#### 2. Build the Project
```bash
cmake -S . -B build/ && cd build && make -j32
```

### Testing and Examples

The `examples/` directory includes two sample binaries:

- `p2p_one_to_one` – minimal client/server data exchange
- `p2p_multi` – multi-endpoint latency benchmark with verification

Build them with the top-level CMake project or from the `examples/` folder:

```bash
cd examples
cmake -S . -B build && cmake --build build
```

Run the provided helper script for quick experiments (logs are written alongside the binaries):

```bash
./run_tests.sh one_to_one   # one-to-one example
./run_tests.sh multi        # multi-endpoint benchmark
```

You can also launch the binaries manually:

```bash
./p2p_one_to_one server
./p2p_one_to_one client

./p2p_multi server <gpu_id> <tag> 0 <remote_tags>
./p2p_multi client <gpu_id> <tag> <client_index> <remote_tags>
```

---

## Architecture

### P2P Communication Pattern

The library uses a **client-driven** communication model:

1. **Client side**: Initiates PUT operations to send data
2. **Server side**: Waits for synchronization flags
3. **Synchronization**: Uses AMO (Atomic Memory Operations) for coordination
4. **Data flow**: 
   - Client → Server (PUT)
   - Server → Client (response via flags)
   - Client → Server (GET to verify data)

### Key Components

- **Warp-level operations**: `ibgda_put_nbi_warp`, `ibgda_get_nbi_warp`
- **Atomic operations**: `ibgda_amo_nonfetch_add` for flag updates


### Multi-endpoint buffer layout

`p2p_multi.cu` uses a fixed slot size constant, `ALLOC_BUF_PER_ENDPOINT = 10 << 20` (10 MiB). Every client and server reserves one slot of that size for each remote peer; the slot is large enough to hold the biggest message and keeps addressing simple (slot index × 10 MiB).

For a 4-client × 4-server configuration the buffers look like this (slot count × 10 MiB):

```
Client c1 buffer (4 × 10 MiB + reserved tail)
┌──────────────────────────────────────────────┐
│ Slot 0 : c1 → s1 payload (10 MiB)            │
├──────────────────────────────────────────────┤
│ Slot 1 : c1 → s2 payload (10 MiB)            │
├──────────────────────────────────────────────┤
│ Slot 2 : c1 → s3 payload (10 MiB)            │
├──────────────────────────────────────────────┤
│ Slot 3 : c1 → s4 payload (10 MiB)            │
├──────────────────────────────────────────────┤
│ Reserved (few KiB)                           │
└──────────────────────────────────────────────┘

Server s1 buffer (4 × 10 MiB + reserved tail)
┌──────────────────────────────────────────────┐
│ Slot 0 : c1 payload + flag (10 MiB)          │
├──────────────────────────────────────────────┤
│ Slot 1 : c2 payload + flag (10 MiB)          │
├──────────────────────────────────────────────┤
│ Slot 2 : c3 payload + flag (10 MiB)          │
├──────────────────────────────────────────────┤
│ Slot 3 : c4 payload + flag (10 MiB)          │
├──────────────────────────────────────────────┤
│ Reserved (few KiB)                           │
└──────────────────────────────────────────────┘
```


