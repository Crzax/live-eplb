# Stage 3 实验记录

8×H20，NV18 全互联（每对 18 条 NVLink），CUDA 12.9，torch 2.7.0+cu126。

```bash
pip install -e . --no-build-isolation
torchrun --nproc_per_node=8 tutorials/03_p2p_bandwidth.py
torchrun --nproc_per_node=4 tests/test_p2p.py
```

## 要解决的问题

Stage 2 的 `stage_expert_async` 是从本地显存拷贝，等于作弊——真实场景下，专家 e 的权重在**别的进程、别的卡**上。

EP 推理每张卡一个进程，各进程的显存地址空间互相不可见。所以需要 CUDA IPC：让 A 进程把自己显存块的"句柄"导出，B 进程拿着句柄映射成自己可访问的指针，之后直接 `cudaMemcpyPeerAsync` 走 NVLink 拷贝，**源进程完全不参与**。

对比 NCCL 的 `send/recv`：那需要两边都发起调用、都要进通信算子，源 rank 必须配合。IPC + peer copy 是**单边操作**，只有目标 rank 动作。这对 Stage 4 的异步状态机很关键——各 rank 的迁移时机可以完全独立。

## 实验 1：正确性

```
rank0 pulled slot 7 from rank 1: got 0.107 want 0.107
ranks with bit-identical pull: 8/8
```

每个 rank 把自己的 pool 填成可识别的模式（`rank*100 + slot`），然后拉取邻居的某个槽，逐字节比对。8 个 rank 全部 bit-identical。

这里用 `torch.equal` 而不是 `allclose`——拷贝是字节搬运，不存在精度问题，任何偏差都说明地址算错了。

## 实验 2：带宽

```
1 slot            6.29 MB    0.043 ms    147.6 GB/s
4 contiguous     25.17 MB    0.092 ms    272.3 GB/s
40 scattered     251.7 MB    1.507 ms    167.0 GB/s
```

参照值：单独测 `torch.copy_` 在同一对卡上搬 252 MB 连续内存，实测 **376 GB/s**，这是单对的上限。

三行数字说明两件事：

**小拷贝是启动开销主导。** 6.29 MB 只有 147.6 GB/s，不到上限的 40%。一次 `cudaMemcpyPeerAsync` 的固定开销在几十微秒量级，对 6 MB 这种小块占比太大。

**合并连续槽位能显著改善。** 4 个连续槽（25 MB）一次拷贝达到 272 GB/s，比单槽快 1.8 倍。所以 `pull_batch` 里加了 `_coalesce`，把 `(dst, src_rank, src_slot)` 三元组里**源和目标都连续、且同一个源 rank** 的任务合并成一次调用。

**40 个分散槽位合并不了。** 它们来自不同的源 rank、源槽位也不连续，只能发 40 次，所以掉回 167 GB/s。这正是真实场景的形态——Stage 1 的规划结果本来就是分散的。

> 所以"40 层迁移耗时"的实际数字是 **1.5 ms**，不是按 376 GB/s 理论值算出的 0.67 ms。ROADMAP 里原先写的"< 2 ms"碰巧对了，但推理过程是错的（我之前按 350 GB/s 估算成 19 ms，那个估算假设了 6.57 GB 的全量，而单卡只需搬 252 MB）。

## 实验 3：路由表提交

```
table applied          : True
addresses unchanged    : True
commit latency         : 12.6 us
```

`commit_routing_table` 是个一次 launch 的 kernel，把新表整体覆盖旧表。12.6 us，相对一个 decode step（Stage 2 实测 1.7-2.3 ms）是 **0.6%**。

**不需要 atomic。** 同一个 stream 上的 kernel 天然有序：commit kernel 入队后，后续入队的 `graph.replay()` 必然在它之后执行，不可能看到半写状态。原规格要求"原子更新"是多余的，atomic 只会拖慢速度。

`addresses unchanged` 这条断言保护 Stage 2 的前提：表的地址被 CUDA Graph 固化了，commit 必须是原地覆盖，不能换指针。

## 实验 4：拷贝对计算的干扰 —— 预期之外的结果

```
step alone                  : 0.280 ms
step during 252 MB pull     : 0.667 ms  (2.38x)
total extra latency for the whole migration: 0.39 ms
```

**这个结果和我的预期相反。** 我原本以为 `cudaMemcpyPeerAsync` 走 copy engine、不占 SM，所以对计算零干扰。实测慢了 2.1-2.4 倍（多次运行波动）。

单独验证了一下更极端的情况（3 GB 拷贝 + GEMM）：慢 **28 倍**。

### 原因

copy engine 不占 SM 是对的，但它**占 HBM 带宽**。拷贝进来的数据要写入本地 HBM，而 GEMM 正在从同一个 HBM 读权重和激活。两者争抢同一条内存总线。

H20 的 HBM 带宽约 4 TB/s，NVLink 单向 376 GB/s 只占 9%，看起来不多。但 GEMM 本身就是带宽受限的（尤其 decode 阶段的 GEMV 形态），任何额外的内存流量都会直接反映在延迟上。

### 这改变了什么

**不改变"零停机"这个结论**，但要修正表述：

| | 错误表述 | 实测 |
|---|---|---|
| 阻塞 | "拷贝完全不影响推理" | 不阻塞 stream，但 step 会变慢 2.15× |
| 代价 | "免费" | 252 MB 迁移 = 1.5 ms 降速窗口，实测多付 0.39 ms |

和 stop-the-world 对比才是重点：

```
Stop-the-world:  一次重录完全停住（Stage 2 单卡实测 25-57 ms，取决于显存碎片）
LiveEPLB:        1.5 ms 窗口内 step 变慢 2.38×，整次迁移实测多付 0.39 ms
```

> 重录耗时波动很大：Stage 2 两次运行分别是 25.2 ms 和 57.3 ms。它受 graph 内部结构分配和
> cuBLAS workspace 申请影响，而这些取决于当时的显存碎片状况。所以**可信的是量级（一个
> decode step 的 10-30 倍），不是具体数值**。两个实验环境也不同（Stage 2 单进程单卡，
> Stage 3 八进程），严格来说不该直接相减。

即便取最乐观的 25 ms，0.39 ms 的额外延迟也是**两个数量级**的改善，而且降速是渐进的，不是断崖式卡住。

> 这个 0.39 ms 偏乐观：拷贝窗口（1.5 ms）里只放得下 1 个 step（0.28 ms 正常 / 0.67 ms 受扰），
> 所以受影响的 step 很少。真实 decode step 要长得多（Stage 2 单层实测 1.7-2.3 ms），
> 拷贝可能在一个 step 内就完成，额外延迟会更接近"一个 step 慢 2 倍"。Stage 4 用真实步长测。

Stage 4 可以进一步优化：把 40 层的迁移**分摊到多个 step**，每拍只搬一两层，把降速幅度压到可忽略。这是"有界迁移量"带来的额外自由度——反正最多 32 槽，怎么分摊都可控。

## 测试

`tests/test_p2p.py`，11 个，需要 `torchrun --nproc_per_node>=2`。

| 测试 | 断言 |
|---|---|
| `test_handles_exchanged` | 每个 rank 映射到 world-1 个 peer |
| `test_pull_is_bit_exact` | 拉取结果与源端逐字节相同 |
| `test_pull_leaves_other_slots_alone` | 只写目标槽，base 槽不受影响 |
| `test_multi_slot_pull` | 多槽连续拷贝每个槽内容都对 |
| `test_coalesce_merges_runs` | 连续的合并、不连续的不合并、跨 rank 的不合并 |
| `test_pull_batch_correct` | 批量拉取（含合并路径）结果正确 |
| `test_commit_applies_table` | 表被覆盖且地址不变 |
| `test_commit_rejects_bad_dtype` | int64 表被拒绝 |
| `test_local_tasks_filters_by_rank` | 规划器输出能正确过滤出本 rank 的任务 |
| `test_local_tasks_rejects_self_copy` | `src_rank == dst_rank` 会被断言拦下 |
| `test_oob_slot_rejected` | 越界槽位被拒绝 |

`test_local_tasks_rejects_self_copy` 测的是一个**规划器层面已经保证的不变量**：专家 e 的本体固定在
卡 `e // epr` 的 base 槽里，而贪心会排除已持有 e 的卡（`peak[holds.T] = inf`），所以
`src_rank == dst_rank` 的任务根本不可能产生。断言是为了防止将来改规划器时破坏这个性质——
真发生了就是自己拷自己，白占一个冗余槽且毫无收益。

`test_pull_leaves_other_slots_alone` 是针对地址计算错误的——如果 `dst_off` 算错，很可能覆盖到 base 槽，而 base 槽是 Stage 1 承诺"全程只读"的。

每个测试之间会重置 scratch 槽位，避免前一个测试的残留数据让后一个假通过。

## 已知局限

- **单节点。** IPC 句柄只能在同一台机器的进程间传递。跨节点要换成 RDMA（`nvshmem` 或 NCCL），那是完全不同的实现。
- **句柄在 `exchange_handles` 时一次性交换**，之后 pool 不能重新分配。这和 Stage 2 的地址锁死是同一个约束，`P2PSwapper.__init__` 里断言了 `storage_offset() == 0`。
- **没测并发 pull。** 多个 rank 同时从同一个源 rank 拉取时的带宽竞争没有测量，Stage 4 的 8 卡同时迁移会遇到。
- **`peer_base_ptr` 暴露了裸指针**，只为调试用，正常路径不该调用。
- `enable_peer_access` 失败时静默跳过那个 peer（拓扑不支持 P2P 的情况），但目前 8 卡全互联，这条分支没被实际测到。
