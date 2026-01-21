// Portions derived from NVSHMEM (https://developer.nvidia.com/nvshmem)
// Copyright (c) NVIDIA Corporation.
// Licensed under the NVSHMEM Software License Agreement (version: September 3,
// 2019). See full license at: https://docs.nvidia.com/nvshmem/api/sla.html
//
// Modified from original source:
//  - nvshmem/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh
#pragma once

#include "configs.cuh"
#include "exception.cuh"
#include "utils.cuh"
#include "utils_device.cuh"

#define IBGDA_STATIC_ASSERT(cond, reason) static_assert(cond, reason)

namespace ibgda_p2p {
#ifndef likely
#define likely(x) (__builtin_expect(!!(x), 1))
#endif

#ifndef unlikely
#define unlikely(x) (__builtin_expect(!!(x), 0))
#endif

#ifndef ACCESS_ONCE
#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

#ifndef READ_ONCE
#define READ_ONCE(x) ACCESS_ONCE(x)
#endif

#ifndef WRITE_ONCE
#define WRITE_ONCE(x, v) (ACCESS_ONCE(x) = (v))
#endif

#ifdef NVSHMEM_IBGDA_DEBUG
struct mlx5_err_cqe_ex {
  uint8_t rsvd0[32];
  __be32 srqn;
  uint8_t rsvd1[16];
  uint8_t hw_err_synd;
  uint8_t hw_synd_type;
  uint8_t vendor_err_synd;
  uint8_t syndrome;
  __be32 s_wqe_opcode_qpn;
  __be16 wqe_counter;
  uint8_t signature;
  uint8_t op_own;
};
typedef struct mlx5_err_cqe_ex ibgda_mlx5_err_cqe_t;
#else
typedef struct mlx5_err_cqe ibgda_mlx5_err_cqe_t;
#endif

IBGDA_STATIC_ASSERT(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64,
                    "Invalid QP minimum depth");

__device__ static __forceinline__ uint64_t HtoBE64(uint64_t x) {
  uint64_t ret;
  asm("{\n\t"
      ".reg .b32 ign;\n\t"
      ".reg .b32 lo;\n\t"
      ".reg .b32 hi;\n\t"
      ".reg .b32 new_lo;\n\t"
      ".reg .b32 new_hi;\n\t"
      "mov.b64 {lo,hi}, %1;\n\t"
      "prmt.b32 new_hi, lo, ign, 0x0123;\n\t"
      "prmt.b32 new_lo, hi, ign, 0x0123;\n\t"
      "mov.b64 %0, {new_lo,new_hi};\n\t"
      "}"
      : "=l"(ret)
      : "l"(x));
  return ret;
}

__device__ static __forceinline__ uint32_t HtoBE32(uint32_t x) {
  uint32_t ret;
  asm("{\n\t"
      ".reg .b32 ign;\n\t"
      "prmt.b32 %0, %1, ign, 0x0123;\n\t"
      "}"
      : "=r"(ret)
      : "r"(x));
  return ret;
}

__device__ static __forceinline__ uint16_t HtoBE16(uint16_t x) {
  // TODO: simplify PTX using 16-bit instructions
  auto a = static_cast<uint32_t>(x);
  uint32_t d;
  asm volatile("{\n\t"
               ".reg .b32 mask;\n\t"
               ".reg .b32 ign;\n\t"
               "mov.b32 mask, 0x4401;\n\t"
               "mov.b32 ign, 0x0;\n\t"
               "prmt.b32 %0, %1, ign, mask;\n\t"
               "}"
               : "=r"(d)
               : "r"(a));
  return static_cast<uint16_t>(d);
}

typedef struct mlx5_wqe_ctrl_seg
    __attribute__((__aligned__(8))) ibgda_ctrl_seg_t;

typedef struct {
  uint32_t add_data;
  uint32_t field_boundary;
  uint64_t reserved;
} __attribute__((__packed__)) ibgda_atomic_32_masked_fa_seg_t;

__device__ static __forceinline__ nvshmemi_ibgda_device_state_t *
ibgda_get_state() {
  return &nvshmemi_ibgda_device_state_d;
}
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint8_t
ibgda_atomic_read(uint8_t *ptr) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
  uint16_t ret;
  asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b8 %0, [%1];"
               : "=h"(ret)
               : "l"(ptr));
  return (uint8_t)ret;
#else
  return READ_ONCE(*ptr);
#endif
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint16_t
ibgda_atomic_read(uint16_t *ptr) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
  uint16_t ret;
  asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];"
               : "=h"(ret)
               : "l"(ptr));
  return ret;
#else
  return READ_ONCE(*ptr);
#endif
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint32_t
ibgda_atomic_read(uint32_t *ptr) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
  uint32_t ret;
  asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];"
               : "=r"(ret)
               : "l"(ptr));
  return ret;
#else
  return READ_ONCE(*ptr);
#endif
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t
ibgda_atomic_read(uint64_t *ptr) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
  uint64_t ret;
  asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];"
               : "=l"(ret)
               : "l"(ptr));
  return ret;
#else
  return READ_ONCE(*ptr);
#endif
}
// Prevent code reordering from both compiler and GPU
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void IBGDA_MFENCE() {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_MFENCE
  asm volatile("fence.acq_rel.cta;" ::: "memory");
#else
  __threadfence_block();
#endif /* NVSHMEMI_IBGDA_PTX_OPTIMIZATION_MFENCE */
}
__device__ static __forceinline__ nvshmemi_ibgda_device_qp_t *
ibgda_get_rc(int pe, int id, p2pcomm_ibgda_device_state_t *state) {
  // auto state = ibgda_get_state();
  const auto num_rc_per_pe = state->num_rc_per_pe;
  return &state->rcs[pe * num_rc_per_pe + id % num_rc_per_pe];
}

__device__ static __forceinline__ void ibgda_lock_acquire(int *lock) {
  while (atomicCAS(lock, 0, 1) == 1)
    ;

  // Prevent reordering before the lock is acquired
  memory_fence_cta();
}

__device__ static __forceinline__ void ibgda_lock_release(int *lock) {
  memory_fence_cta();

  // Prevent reordering before lock is released
  st_na_relaxed(lock, 0);
}

__device__ static __forceinline__ void
ibgda_update_dbr(nvshmemi_ibgda_device_qp_t *qp, uint32_t dbrec_head) {
  // `DBREC` contains the index of the next empty `WQEBB`
  __be32 dbrec_val;
  __be32 *dbrec_ptr = qp->tx_wq.dbrec;

  // This is equivalent to `WRITE_ONCE(dbrec_ptr, HtoBE32(dbrec_head & 0xffff))`
  asm("{\n\t"
      ".reg .b32 dbrec_head_16b;\n\t"
      ".reg .b32 ign;\n\t"
      "and.b32 dbrec_head_16b, %1, 0xffff;\n\t"
      "prmt.b32 %0, dbrec_head_16b, ign, 0x123;\n\t"
      "}"
      : "=r"(dbrec_val)
      : "r"(dbrec_head));
  st_na_release(dbrec_ptr, dbrec_val);
}

__device__ static __forceinline__ void
ibgda_ring_db(nvshmemi_ibgda_device_qp_t *qp, uint16_t prod_idx) {
  auto bf_ptr = reinterpret_cast<uint64_t *>(qp->tx_wq.bf);
  ibgda_ctrl_seg_t ctrl_seg = {.opmod_idx_opcode = HtoBE32(prod_idx << 8),
                               .qpn_ds = HtoBE32(qp->qpn << 8)};

  IBGDA_STATIC_ASSERT(sizeof(decltype(&ctrl_seg)) == sizeof(uint64_t), "");
  st_na_release(bf_ptr, *(reinterpret_cast<uint64_t *>(&ctrl_seg)));
}

__device__ static __forceinline__ void
ibgda_post_send(nvshmemi_ibgda_device_qp_t *qp, uint64_t new_prod_idx) {
  nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
  uint64_t old_prod_idx;

  // Update `prod_idx` before ringing the doorbell, so that we know which index
  // is needed in quiet/fence
  ibgda_lock_acquire(&mvars->post_send_lock);

  old_prod_idx = atomicMax(
      reinterpret_cast<unsigned long long int *>(&mvars->tx_wq.prod_idx),
      new_prod_idx);
  if (new_prod_idx > old_prod_idx) {
    ibgda_update_dbr(qp, new_prod_idx);
    ibgda_ring_db(qp, new_prod_idx);
  }
  ibgda_lock_release(&mvars->post_send_lock);
}

template <bool kAlwaysDoPostSend>
__device__ static __forceinline__ void
ibgda_submit_requests(nvshmemi_ibgda_device_qp_t *qp, uint64_t base_wqe_idx,
                      uint32_t num_wqes, int message_idx = 0) {
  nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
  uint64_t new_wqe_idx = base_wqe_idx + num_wqes;

  // WQE writes must be finished first
  __threadfence();

  // Wait for prior WQE slots to be filled first
  auto *ready_idx =
      reinterpret_cast<unsigned long long int *>(&mvars->tx_wq.ready_head);
  while (atomicCAS(ready_idx, base_wqe_idx, new_wqe_idx) != base_wqe_idx)
    ;

  // Always post, not in batch
  constexpr int kNumRequestInBatch = 4;
  if (kAlwaysDoPostSend or (message_idx + 1) % kNumRequestInBatch == 0)
    ibgda_post_send(qp, new_wqe_idx);
}

__device__ static __forceinline__ uint64_t
ibgda_reserve_wqe_slots(nvshmemi_ibgda_device_qp_t *qp, uint32_t num_wqes) {
  auto mvars = &qp->mvars;
  return atomicAdd(
      reinterpret_cast<unsigned long long *>(&mvars->tx_wq.resv_head),
      static_cast<unsigned long long>(num_wqes));
}

__device__ static __forceinline__ void *
ibgda_get_wqe_ptr(nvshmemi_ibgda_device_qp_t *qp, uint16_t wqe_idx) {
  uint16_t cnt = qp->tx_wq.nwqes;
  uint16_t idx = wqe_idx & (cnt - 1);
  return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(qp->tx_wq.wqe) +
                                  (idx << MLX5_SEND_WQE_SHIFT));
}

__device__ static __forceinline__ void
ibgda_write_rdma_write_wqe(nvshmemi_ibgda_device_qp_t *qp, uint64_t laddr,
                           __be32 lkey, uint64_t raddr, __be32 rkey,
                           uint32_t bytes, uint16_t wqe_idx, void **out_wqes) {
  ibgda_ctrl_seg_t ctrl_seg;
  struct mlx5_wqe_raddr_seg raddr_seg;
  struct mlx5_wqe_data_seg data_seg;

  auto *ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t *>(out_wqes[0]);
  void *av_seg_ptr = reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
  struct mlx5_wqe_raddr_seg *raddr_seg_ptr;
  struct mlx5_wqe_data_seg *data_seg_ptr;

  raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg *>(
      reinterpret_cast<uintptr_t>(av_seg_ptr));
  data_seg_ptr = reinterpret_cast<mlx5_wqe_data_seg *>(
      reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));

  raddr_seg.raddr = HtoBE64(raddr);
  raddr_seg.rkey = rkey;
  raddr_seg.reserved = 0;

  data_seg.byte_count = HtoBE32(bytes);
  data_seg.lkey = lkey;
  data_seg.addr = HtoBE64(laddr);

  ctrl_seg = {0};
  ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 3);
  ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
  ctrl_seg.opmod_idx_opcode = HtoBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);
  IBGDA_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == 16,
                      "sizeof(*ctrl_seg_ptr) == 16");
  IBGDA_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == 16,
                      "sizeof(*raddr_seg_ptr) == 16");
  IBGDA_STATIC_ASSERT(sizeof(*data_seg_ptr) == 16,
                      "sizeof(*data_seg_ptr) == 16");
  st_na_relaxed(reinterpret_cast<int4 *>(ctrl_seg_ptr),
                *reinterpret_cast<const int4 *>(&ctrl_seg));
  st_na_relaxed(reinterpret_cast<int4 *>(raddr_seg_ptr),
                *reinterpret_cast<const int4 *>(&raddr_seg));
  st_na_relaxed(reinterpret_cast<int4 *>(data_seg_ptr),
                *reinterpret_cast<const int4 *>(&data_seg));
}
// 设备端函数，用于在warp级别执行非阻塞的RDMA写操作（PUT）
// 该函数由warp中的所有线程协作执行
__device__ static __forceinline__ void
nvshmemi_ibgda_put_nbi_warp(uint64_t req_rptr,           // 远程端地址（目标地址）
              uint64_t req_lptr,           // 本地端地址（源地址）
              size_t bytes,                 // 需要传输的字节数
              int dst_pe,                   // 目标处理单元ID
              int qp_id,                    // 队列对ID
              int lane_id,                  // 当前线程在warp中的ID（0-31）
              int message_idx,              // 消息索引
              p2pcomm_ibgda_device_state_t *state) {  // 设备状态指针

  // 初始化WQE（Work Queue Element）相关变量
  uint32_t num_wqes = 0;                // WQE数量计数器
  __be32 my_lkey = 0;                   // 本地内存注册钥匙（大端格式）
  uint64_t my_laddr = 0;                // 本地地址
  __be32 my_rkey = 0;                   // 远程内存注册钥匙（大端格式）
  uint64_t my_raddr = 0;                // 远程地址
  uint64_t my_chunk_size = 0;           // 当前块的大小

  // 定义最小块大小为128KB
  static size_t min_chunk_size = 128 * 1024;
  // 计算平均块大小：将总字节数分成32个部分
  size_t average_chunk_size = (bytes + 32 - 1) / 32;

  // 从状态获取本地和远程内存注册钥匙
  my_lkey = state->lkey.key;
  my_rkey = state->rkey.key;

  // 循环计算需要多少个WQE来完成数据传输（理论上最多32个）
  auto remaining_bytes = bytes;
  while (remaining_bytes > 0) {
    // 仅当前线程ID等于WQE数量时，才计算该线程负责的数据块信息
    if (lane_id == num_wqes) {
      my_laddr = req_lptr;                // 设置本地地址
      my_raddr = req_rptr;                // 设置远程地址
      // 计算当前块的大小：取剩余字节数、最小块大小、平均块大小中的较大值
      my_chunk_size =
        min(remaining_bytes, max(min_chunk_size, average_chunk_size));
    }
    
    // 使用warp级别的shuffle指令，将当前WQE对应线程的chunk_size广播给warp中的所有线程
    auto chunk_size =
      /**
       * @brief 使用shuffle操作在warp中广播chunk_size值
       * 
       * 该指令使用NVIDIA的warp级别shuffle原语从lane num_wqes的线程读取my_chunk_size
       * 并将其广播给warp中的所有线程（全32个线程）。
       * 掩码0xffffffff使能warp中的所有线程参与shuffle操作。
       * 
       * @param mask 0xffffffff - 使能所有32个线程参与shuffle
       * @param my_chunk_size - 源值，从源lane读取
       * @param num_wqes - 源lane索引（0-31），指定从哪个线程读取值
       * 
       * @return 从lane num_wqes的线程读取的my_chunk_size值，分发给所有线程
       */
      __shfl_sync(0xffffffff, my_chunk_size, static_cast<int>(num_wqes));
    
    // 减少剩余字节数
    remaining_bytes -= chunk_size;
    // 移动本地地址指针
    req_lptr += chunk_size;
    // 移动远程地址指针
    req_rptr += chunk_size;
    // 增加WQE计数器
    ++num_wqes;
  }
  
  // 断言：WQE数量不能超过32个（warp中的线程数）
  EP_DEVICE_ASSERT(num_wqes <= 32);
  
  // 获取目标PE对应的队列对
  auto qp = ibgda_get_rc(dst_pe, qp_id, state);
  
  // 初始化基础WQE索引
  uint64_t base_wqe_idx = 0;
  
  // 仅线程0预留足够的WQE槽位
  if (lane_id == 0)
    base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes);
  
  // 将基础WQE索引从线程0广播给warp中的所有线程
  base_wqe_idx = __shfl_sync(0xffffffff, base_wqe_idx, 0);
  
  // 每个线程处理分配给它的WQE（线程ID < WQE数量）
  if (lane_id < num_wqes) {
  // 获取该线程对应WQE的内存指针
    auto wqe_ptr = ibgda_get_wqe_ptr(qp, base_wqe_idx + lane_id);
    // 写入RDMA写操作的WQE信息
    ibgda_write_rdma_write_wqe(qp, my_laddr, my_lkey, my_raddr, my_rkey,
                  my_chunk_size, base_wqe_idx, &wqe_ptr);
  }
  
  // 同步warp中的所有线程，确保所有WQE都已写入
  __syncwarp();

  // 提交请求到硬件队列
  // 仅线程0执行提交操作
  if (lane_id == 0)
  // 模板参数false表示不总是立即post_send，可能进行批处理
  ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes, message_idx);
  
  // 再次同步warp中的所有线程，确保提交操作完成
  __syncwarp();
}

__device__ static __forceinline__ void
ibgda_write_amo_add_wqe(nvshmemi_ibgda_device_qp_t *qp, const int &value,
                        uint64_t laddr, __be32 lkey, uint64_t raddr,
                        __be32 rkey, uint16_t wqe_idx, void **out_wqes) {
  ibgda_ctrl_seg_t ctrl_seg = {0};
  struct mlx5_wqe_raddr_seg raddr_seg;
  struct mlx5_wqe_atomic_seg atomic_seg_1;
  struct mlx5_wqe_data_seg data_seg;

  auto ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t *>(out_wqes[0]);
  auto raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg *>(
      reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
  auto atomic_seg_ptr = reinterpret_cast<mlx5_wqe_atomic_seg *>(
      reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));
  auto data_seg_ptr = reinterpret_cast<mlx5_wqe_data_seg *>(
      reinterpret_cast<uintptr_t>(atomic_seg_ptr) + sizeof(*atomic_seg_ptr));

  raddr_seg.raddr = HtoBE64(raddr);
  raddr_seg.rkey = rkey;
  raddr_seg.reserved = 0;

  // NOTES: `0x08000000` means `IBGDA_4_BYTE_EXT_AMO_OPMOD`
  ctrl_seg.opmod_idx_opcode =
      HtoBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) | 0x08000000);
  auto atomic_32_masked_fa_seg =
      reinterpret_cast<ibgda_atomic_32_masked_fa_seg_t *>(&atomic_seg_1);
  atomic_32_masked_fa_seg->add_data = HtoBE32(value);
  atomic_32_masked_fa_seg->field_boundary = 0;

  ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 4);
  ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;

  data_seg.byte_count = HtoBE32(sizeof(int));
  data_seg.lkey = lkey;
  data_seg.addr = HtoBE64(laddr);

  IBGDA_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == sizeof(int4),
                      "Invalid vectorization");
  IBGDA_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == sizeof(int4),
                      "Invalid vectorization");
  IBGDA_STATIC_ASSERT(sizeof(*atomic_seg_ptr) == sizeof(int4),
                      "Invalid vectorization");
  IBGDA_STATIC_ASSERT(sizeof(*data_seg_ptr) == sizeof(int4),
                      "Invalid vectorization");
  st_na_relaxed(reinterpret_cast<int4 *>(ctrl_seg_ptr),
                *reinterpret_cast<int4 *>(&ctrl_seg));
  st_na_relaxed(reinterpret_cast<int4 *>(raddr_seg_ptr),
                *reinterpret_cast<int4 *>(&raddr_seg));
  st_na_relaxed(reinterpret_cast<int4 *>(atomic_seg_ptr),
                *reinterpret_cast<int4 *>(&atomic_seg_1));
  st_na_relaxed(reinterpret_cast<int4 *>(data_seg_ptr),
                *reinterpret_cast<int4 *>(&data_seg));
}

__device__ __forceinline__ void
nvshmemi_ibgda_amo_nonfetch_add(void *rptr, const int &value, int pe, int qp_id,
                                p2pcomm_ibgda_device_state_t *state) {
  nvshmemi_ibgda_device_qp_t *qp = ibgda_get_rc(pe, qp_id, state);

  __be32 rkey;
  uint64_t raddr;

  rkey = state->rkey.key;
  raddr = reinterpret_cast<uint64_t>(rptr);

  uint64_t my_wqe_idx = ibgda_reserve_wqe_slots(qp, 1);
  void *wqe_ptrs = ibgda_get_wqe_ptr(qp, my_wqe_idx);
  ibgda_write_amo_add_wqe(qp, value, reinterpret_cast<uint64_t>(qp->ibuf.buf),
                          qp->ibuf.lkey, raddr, rkey, my_wqe_idx, &wqe_ptrs);

  ibgda_submit_requests<true>(qp, my_wqe_idx, 1);
}
__device__ static __forceinline__ void
ibgda_read_rdma_read_wqe(nvshmemi_ibgda_device_qp_t *qp, uint64_t laddr,
                         __be32 lkey, uint64_t raddr, __be32 rkey,
                         uint32_t bytes, uint16_t wqe_idx, void **out_wqes) {
  ibgda_ctrl_seg_t ctrl_seg;
  struct mlx5_wqe_raddr_seg raddr_seg;
  struct mlx5_wqe_data_seg data_seg;

  auto *ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t *>(out_wqes[0]);
  void *av_seg_ptr = reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
  struct mlx5_wqe_raddr_seg *raddr_seg_ptr;
  struct mlx5_wqe_data_seg *data_seg_ptr;

  raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg *>(
      reinterpret_cast<uintptr_t>(av_seg_ptr));
  data_seg_ptr = reinterpret_cast<mlx5_wqe_data_seg *>(
      reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));

  raddr_seg.raddr = HtoBE64(raddr);
  raddr_seg.rkey = rkey;
  raddr_seg.reserved = 0;

  data_seg.byte_count = HtoBE32(bytes);
  data_seg.lkey = lkey;
  data_seg.addr = HtoBE64(laddr);

  ctrl_seg = {0};
  ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 3);
  ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
  ctrl_seg.opmod_idx_opcode = HtoBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_READ);

  EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == 16, "sizeof(*ctrl_seg_ptr) == 16");
  EP_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == 16,
                   "sizeof(*raddr_seg_ptr) == 16");
  EP_STATIC_ASSERT(sizeof(*data_seg_ptr) == 16, "sizeof(*data_seg_ptr) == 16");
  st_na_relaxed(reinterpret_cast<int4 *>(ctrl_seg_ptr),
                *reinterpret_cast<const int4 *>(&ctrl_seg));
  st_na_relaxed(reinterpret_cast<int4 *>(raddr_seg_ptr),
                *reinterpret_cast<const int4 *>(&raddr_seg));
  st_na_relaxed(reinterpret_cast<int4 *>(data_seg_ptr),
                *reinterpret_cast<const int4 *>(&data_seg));
}
__device__ __forceinline__ void
nvshmemi_ibgda_poll_cq_until(nvshmemi_ibgda_device_cq_v1 *cq,
                             uint32_t num_wqes) {
  uint32_t polled = 0;

  // Dynamically calculate log2(cq->ncqes)
  int log2_cq_size = __ffs(cq->ncqes) - 1;

  while (polled < num_wqes) {
    uint64_t cons_idx = atomicAdd((unsigned long long *)cq->cons_idx, 0);
    uint32_t index = cons_idx & (cq->ncqes - 1); // CQ is ring-buffered

    // Get current CQE pointer
    volatile uint32_t *cqe = &((volatile uint32_t *)cq->cqe)[index];
    uint32_t val = *cqe;

    // Check if this is a new CQE using owner bit (refer to Mellanox CQ owner
    // bit documentation)
    uint32_t owner_bit = val >> 31;
    if (owner_bit == ((cons_idx >> log2_cq_size) & 0x1)) {
      // This indicates a valid, unconsumed CQE
      polled++;

      // Write back to notify hardware that this CQE has been processed
      // (optional) NOTE: This step depends on your CQ design, some need ack,
      // some don't
      *cqe = 0;

      // Update cons_idx
      atomicAdd((unsigned long long *)cq->cons_idx, 1);
    } else {
      // If no new CQE, wait a bit to avoid aggressive busy waiting
      __nanosleep(100);
    }
  }
}
__device__ static __forceinline__ void
nvshmemi_ibgda_get_nbi_warp(uint64_t req_rptr, uint64_t req_lptr, size_t bytes,
                            int dst_pe, int qp_id, int lane_id, int message_idx,
                            p2pcomm_ibgda_device_state_t *state) {
  // Get lkey and rkey, store them into lanes
  uint32_t num_wqes = 0;
  __be32 my_lkey = 0;
  uint64_t my_laddr = 0;
  __be32 my_rkey = 0;
  uint64_t my_raddr = 0;
  uint64_t my_chunk_size = 0;

  static size_t min_chunk_size = 128 * 1024;
  size_t average_chunk_size = (bytes + 32 - 1) / 32;

  // auto state = ibgda_get_state();
  my_lkey = state->lkey.key;
  my_rkey = state->rkey.key;

  // Decide how many messages (theoretically 3 for maximum)
  auto remaining_bytes = bytes;
  while (remaining_bytes > 0) {
    if (lane_id == num_wqes) {
      my_laddr = req_lptr;
      my_raddr = req_rptr;
      my_chunk_size =
          min(remaining_bytes, max(min_chunk_size, average_chunk_size));
    }
    // Move one more message
    auto chunk_size =
        __shfl_sync(0xffffffff, my_chunk_size, static_cast<int>(num_wqes));
    remaining_bytes -= chunk_size;
    req_lptr += chunk_size;
    req_rptr += chunk_size;
    ++num_wqes;
  }
  EP_DEVICE_ASSERT(num_wqes <= 32);
  // Process WQE
  auto qp = ibgda_get_rc(dst_pe, qp_id, state);
  uint64_t base_wqe_idx = 0;
  if (lane_id == 0)
    base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes);
  base_wqe_idx = __shfl_sync(0xffffffff, base_wqe_idx, 0);
  if (lane_id < num_wqes) {
    auto wqe_ptr = ibgda_get_wqe_ptr(qp, base_wqe_idx + lane_id);
    ibgda_read_rdma_read_wqe(qp, my_laddr, my_lkey, my_raddr, my_rkey,
                             my_chunk_size, base_wqe_idx, &wqe_ptr);
  }
  __syncwarp();

  // Submit
  if (lane_id == 0)
    ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes, message_idx);
  __syncwarp();
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE int
ibgda_poll_cq(nvshmemi_ibgda_device_cq_t *cq, uint64_t idx, int *error) {
  int status = 0;
  struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *)cq->cqe;

  const uint32_t ncqes = cq->ncqes;

  uint8_t opown;
  uint8_t opcode;
  uint16_t wqe_counter;
  uint16_t new_wqe_counter;

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
  uint64_t start = ibgda_query_globaltimer();
  uint64_t now;
#endif

  uint64_t cons_idx = ibgda_atomic_read(cq->cons_idx);
  uint64_t new_cons_idx;

  // assert(likely(cq->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI ||
  //               cq->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC));

  if (unlikely(cons_idx >= idx))
    goto out;

#ifdef NVSHMEM_IBGDA_DEBUG
  // We can skip opcode == MLX5_CQE_INVALID check because we have already
  // initialized the CQ buffer to 0xff. With the QP depth range we enforce,
  // cons_idx cannot progress unless wqe_counter read from the CQ buffer is
  // a valid value.
  do {
    opown = ibgda_atomic_read(&cqe64->op_own);
    opcode = opown >> 4;

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    // TODO: Integrate timeout handler with the core NVSHMEM
    now = ibgda_query_globaltimer();
    status = ibgda_check_poll_timeout(cq, now, start, idx, error);
    if (status != 0)
      goto check_opcode;
#endif /* NVSHMEM_TIMEOUT_DEVICE_POLLING */
  } while (unlikely(opcode == MLX5_CQE_INVALID));

  // Prevent reordering of the opcode wait above
  IBGDA_MFENCE();
#endif /* NVSHMEM_IBGDA_DEBUG */

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
  start = ibgda_query_globaltimer();
#endif

  // If idx is a lot greater than cons_idx, we might get incorrect result due
  // to wqe_counter wraparound. We need to check prod_idx to be sure that idx
  // has already been submitted.
  while (unlikely(ibgda_atomic_read(cq->prod_idx) < idx))
    ;
  IBGDA_MFENCE();

  do {
    new_wqe_counter = ibgda_atomic_read(&cqe64->wqe_counter);
    new_wqe_counter = BSWAP16(new_wqe_counter);
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    now = ibgda_query_globaltimer();
    status = ibgda_check_poll_timeout(cq, now, start, idx, error);
    if (status != 0)
      goto check_opcode;

    // Observe progress. Reset the timer.
    if (new_wqe_counter != wqe_counter)
      start = now;
#endif
    wqe_counter = new_wqe_counter;

    // Another thread may have updated cons_idx.
    cons_idx = ibgda_atomic_read(cq->cons_idx);
    if (likely(cons_idx >= idx))
      goto out;
  }
  // NOTE: This while loop is part of do while above.
  // wqe_counter is the HW consumer index. However, we always maintain index
  // + 1 in SW. To be able to compare with idx, we need to use wqe_counter +
  // 1. Because wqe_counter is uint16_t, it may wraparound. Still we know for
  // sure that if idx - wqe_counter - 1 < ncqes, wqe_counter + 1 is less than
  // idx, and thus we need to wait. We don't need to wait when idx ==
  // wqe_counter + 1. That's why we use - (uint16_t)2 here to make this case
  // wraparound.
  while (unlikely(
      ((uint16_t)((uint16_t)idx - wqe_counter - (uint16_t)2) < ncqes)));

  // new_cons_idx is uint64_t but wqe_counter is uint16_t. Thus, we get the
  // MSB from idx. We also need to take care of wraparound.
  ++wqe_counter;
  new_cons_idx = (idx & ~(0xffffULL) | wqe_counter) +
                 (((uint16_t)idx > wqe_counter) ? 0x10000ULL : 0x0);
  atomicMax((unsigned long long int *)cq->cons_idx,
            (unsigned long long int)new_cons_idx);

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
check_opcode:
#endif

  // NVSHMEM always treats CQE errors as fatal.
  // Even if this error doesn't belong to the CQE in cons_idx,
  // we will just report and terminate the process.
  opown = ibgda_atomic_read(&cqe64->op_own);
  opcode = opown >> 4;

  if (unlikely(opcode == MLX5_CQE_REQ_ERR)) {
    ibgda_mlx5_err_cqe_t *cqe_err = (ibgda_mlx5_err_cqe_t *)cqe64;
    *error = cqe_err->syndrome;
#ifdef NVSHMEM_IBGDA_DEBUG
    __be16 wqe_counter = ibgda_atomic_read(&cqe_err->wqe_counter);
    __be32 s_wqe_opcode_qpn = ibgda_atomic_read(&cqe_err->s_wqe_opcode_qpn);
    printf("[%d] got completion with err:\n"
           "   syndrome=%#x, vendor_err_synd=%#x, hw_err_synd=%#x, "
           "hw_synd_type=%#x,\n"
           "   wqe_counter=%#x, s_wqe_opcode_qpn=%#x,\n"
           "   cqn=%#x, cons_idx=%#lx, prod_idx=%#lx, idx=%#lx\n",
           nvshmemi_device_state_d.mype, cqe_err->syndrome,
           cqe_err->vendor_err_synd, cqe_err->hw_err_synd,
           cqe_err->hw_synd_type, BSWAP16(wqe_counter),
           BSWAP32(s_wqe_opcode_qpn), cq->cqn, cons_idx,
           ibgda_atomic_read(cq->prod_idx), idx);
#endif /* NVSHMEM_IBGDA_DEBUG */
    status = -1;
  }

out:
  // Prevent reordering of this function and subsequent instructions
  IBGDA_MFENCE();

  return status;
}
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t ibgda_quiet(
    nvshmemi_ibgda_device_qp_t *qp, p2pcomm_ibgda_device_state_t *state) {
  // nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
  // uint64_t prod_idx = state->use_async_postsend ?
  // ibgda_atomic_read(qp->tx_wq.prod_idx)
  //                                              :
  //                                              ibgda_atomic_read(&qp->mvars.tx_wq.ready_head);
  uint64_t prod_idx = ibgda_atomic_read(&qp->mvars.tx_wq.ready_head);
  nvshmemi_ibgda_device_cq_t cq = *qp->tx_wq.cq;

  int err = 0;
  int status = ibgda_poll_cq(&cq, prod_idx, &err);
  // TODO: Integrate the error handler with the core NVSHMEM
#ifdef NVSHMEM_IBGDA_DEBUG
  if (status) {
    printf("ibgda_poll_cq failed with error=%d.\n", err);
  }
#endif
  // assert(likely(status == 0));
  return prod_idx;
}
} // namespace ibgda_p2p