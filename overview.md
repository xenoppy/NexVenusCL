# NexVenusCL 项目逻辑速览

## 核心思路
- 直接在 GPU 内核中以 IBGDA 发起 RDMA：内核通过 warp 级 `put/get/amo` 原语写入 RC 队列、轮询 CQ，避免主机代理线程。
- 两端（PE 0/PE 1）对称初始化：每个进程调用 `P2PComm::init()` 载入 CUDA 符号、初始化 NVSHMEM 选项、动态加载 IBGDA 传输插件，随后创建 RC QP 并分发连接信息。
- 内存注册 + 句柄交换：每个进程对 GPU 缓冲区执行 `register_memory`，序列化 mem handle 与指针写入临时文件，再由对端读取并设置为远端句柄。
- 设备态同步：`update_device_state_d()` 将 host-side IBGDA 状态（CQ/RC 地址、lkey/rkey）拷贝到设备侧结构体，供内核直接使用。

相关文件：
- API 入口与初始化流程： [src/p2pcomm.cpp](src/p2pcomm.cpp)
- CUDA/驱动符号加载： [src/host/init/cudawrap.cpp](src/host/init/cudawrap.cpp)
- 传输插件装载与 finalize： [src/host/transport/transport.cpp](src/host/transport/transport.cpp)
- NVSHMEM 选项与调试设施： [src/host/util/env_vars.cpp](src/host/util/env_vars.cpp)、[src/host/util/debug.cpp](src/host/util/debug.cpp)
- 示例（内核 + host 流程）：单对单 [examples/p2p_one_to_one.cu](examples/p2p_one_to_one.cu)，多端基准 [examples/p2p_multi.cu](examples/p2p_multi.cu)，通用工具 [examples/p2p_utils.cuh](examples/p2p_utils.cuh)

## 主机侧数据流
1. **全局初始化**：`global_init()` 解析环境、初始化 NVSHMEM 调试、动态加载 CUDA driver 符号并抓取当前 CUcontext。
2. **传输初始化**：`nvshmemi_transport_init()` 通过 `dlopen` 加载 `nvshmem_transport_ibgda.so.*`，填充 `host_ops` 并创建 RC QP；连接句柄 `rc_handles` 连续存放本端/对端块。
3. **连接握手**：调用 `get_local_conn_handle()` 按块切片导出本端 rc_handles，写入临时文件；对端读取后调用 `connect_endpoint()` 完成双向 RC 建立。
4. **内存注册**：`register_memory()` 通过 transport `get_mem_handle` 取得 lkey/rkey；`set_remote_mem_handle()` 把对端句柄拷入本地存储。
5. **设备态更新**：`update_device_state_d()` 提取 lkey/rkey、CQ/RC 指针并复制到设备可见的 `p2pcomm_ibgda_device_state_t`，供内核使用。
6. **资源清理**：`fini()` 依次 finalize transport、释放 rc_handles 与 device state。

## 设备侧逻辑
- 内核直接调用 IBGDA 原语：`nvshmemi_ibgda_put_nbi_warp`、`nvshmemi_ibgda_get_nbi_warp`、`nvshmemi_ibgda_amo_nonfetch_add`，并在必要时通过 `ibgda_quiet` 等待 CQ 完成。
- 同步与可见性：常见序列为 PUT → `__threadfence_system()` → AMO 写旗标 → 轮询对端旗标 GET/LOAD → 再次 fence 确保数据可见。
- 缓冲布局：示例中客户端/服务端按固定 slot 划分（如多端 10 MiB/slot），尾部留若干同步字段。

## 示例流程
- **一对一**（`p2p_one_to_one.cu`）：
  1. 双端写入/读取连接与 mem 句柄文件；
  2. 客户端 PUT 数据 + 原地写 flag；服务端 polling flag → 处理数据（乘 2）→ 客户端 GET 回读并校验。
- **多端基准**（`p2p_multi.cu`）：
  1. 为每个远端创建独立 `P2PComm`，注册同一大缓冲区并交换句柄；
  2. 客户端按块发送到各远端 slot，使用 AMO+GET 做完备同步；
  3. 服务端仅轮询并回写完成标记；客户端对每个消息规模做多轮 warmup/测量与校验。

## 可以继续改进的点
- 把基于临时文件的句柄交换替换为真正的 out-of-band 控制通道（例如 gRPC/Redis/ZMQ）。
- 提供错误码映射和重试策略，在连接/注册失败时给出可恢复路径。
- 增加自动化自检：检查 `nvidia_peermem`、CUDA driver 版本与 IBGDA 插件是否可加载。
