# Stage 3 调试记录

两个 bug，一个是边界检查写错，一个是**我对硬件行为的假设错了**。

---

## Bug 1：多槽拷贝时边界检查误判

### 症状

`test_multi_slot_pull` 和 `test_pull_batch_correct` 失败：

```
FAIL test_multi_slot_pull: RuntimeError: dst_slot 8 overruns local pool
FAIL test_pull_batch_correct: RuntimeError: dst_slot 8 overruns local pool
```

pool 有 12 个槽，拉取到槽 8 完全合法，却被拒了。

### 根因

最初的参数设计是"一次拷一个槽"，所以检查写成：

```cpp
TORCH_CHECK((dst_slot + 1) * slot_bytes <= local_pool.nbytes(),
            "dst_slot ", dst_slot, " overruns local pool");
```

后来为了合并连续槽位加了 `n_slots`，Python 侧传的是：

```python
_C.async_copy_expert_slot(self.pool, dst_slot, src_rank, src_slot,
                          self.slot_bytes * n_slots, s.cuda_stream)
```

**第五个参数的语义从"单槽字节数"变成了"总字节数"**，但 C++ 侧的检查没跟着改。于是 `(8+1) * (4槽的字节数)` 远超 pool 大小，误判越界。

### 修复

把参数名改成 `nbytes`，让语义明确，并用 pool 自己的形状推导单槽大小：

```cpp
int64_t slot_bytes = local_pool.nbytes() / local_pool.size(0);
int64_t dst_off = dst_slot * slot_bytes;
TORCH_CHECK(dst_off >= 0 && dst_off + nbytes <= local_pool.nbytes(),
            "copy of ", nbytes, " bytes at slot ", dst_slot,
            " overruns local pool of ", local_pool.nbytes(), " bytes");
```

这样 `slot_bytes` 不再依赖调用方传入，少一个出错的地方。

### 教训

**参数语义变了就要改名。** `slot_bytes` 这个名字在加了 `n_slots` 之后就不再准确，但它还叫这个名字，所以所有读代码的人（包括我）都会继续按旧语义理解它。改成 `nbytes` 之后，"这是总字节数"一目了然。

这个 bug 是测试抓到的，而且抓得很及时——如果边界检查是朝另一个方向错（该拦的没拦），就会变成静默的内存越界，排查成本完全不同。

---

## Bug 2：以为 copy engine 不干扰计算 —— 假设错了

### 我的原始假设

ROADMAP 里我自己写的：

> `cudaMemcpyPeerAsync`，走 NVLink Copy Engine 异步拉取单个 6.29 MB 槽位（**不占 SM**）

"不占 SM" 是对的，但我由此推出了一个错误结论：**既然不占 SM，对计算就没有干扰**。

实验 4 一开始的输出是：

```
step alone             : 0.281 ms
step during 252 MB pull: 0.648 ms over 1 steps (2.31x)
```

慢了 2.31 倍。我第一反应是测量方法有问题（毕竟 Stage 2 刚踩过异步计时的坑），于是单独写了个探针，用更极端的 3 GB 拷贝：

```
mm alone                : 0.281 ms
mm during 3 GB peer copy: 8.058 ms  (28.68x)
```

慢了 **28 倍**。不是测量误差，是真的。

### 真正的原因

copy engine 和 SM 是独立的硬件单元，这部分没错。但拷贝进来的数据**必须写入本地 HBM**，而 GEMM 正在从同一个 HBM 读取权重和激活。

两者争抢的不是计算单元，是**内存带宽**。

H20 的 HBM 带宽约 4 TB/s，NVLink 单向 376 GB/s 看起来只占 9%。但：
- decode 阶段的 MoE 计算本身就是带宽受限的（GEMV 形态，计算密度低）
- peer copy 的写入是大块连续写，会挤占内存控制器的队列
- 实际争抢比带宽占比反映的更严重

### 这暴露了什么问题

不是代码 bug，是**我把"不占某个资源"错误推广成了"不占任何资源"**。

更值得反省的是：我在 ROADMAP 里把 "不占 SM" 写成了设计要点，等于把一个硬件事实包装成了性能承诺。如果不做这个实验，Stage 4 的 TPOT 曲线跑出来会莫名其妙地有抖动，那时候再回头查会困难得多。

### 修正后的表述

```
错误：拷贝完全不影响推理
实测：不阻塞 stream，但 step 会在拷贝期间变慢 2.1-2.4 倍（多次运行）
```

关键是和 stop-the-world 对比，结论没变：

| | 代价 |
|---|---|
| Stop-the-world | **完全停住 25-57 ms**（Stage 2 两次实测，波动来自显存碎片） |
| LiveEPLB | 1.5 ms 窗口内降速 2.1-2.4×，实测多花 0.39 ms |

即便取最乐观的 25 ms，仍是一个数量级的改善，而且是渐进降速而非断崖卡死。

> 这里本来写的是"57 ms"，而且在 Stage 3 的脚本里硬编码了这个常数。那是从 Stage 2 的某一次
> 运行手抄来的——另一次是 25.2 ms，差一倍多。跨实验抄数字本身就不严谨（Stage 2 单进程单卡，
> Stage 3 八进程），所以已改成只报本实验实测的额外延迟，重录耗时只提量级区间。

### 带来的新设计空间

既然干扰来自带宽争抢，那就可以**把迁移摊薄**：40 层不一次搬完，每个 decode step 只搬一两层，把瞬时带宽压力降下来。

这正是"有界迁移量"的额外好处——总量最多 32 槽/层是确定的，怎么分摊完全可控。全局重排方案就没这个自由度，因为它的迁移量不确定，没法规划分摊节奏。

这条写进 Stage 4 的 TODO 了。

### 教训

**凡是"因为 X 所以不会 Y"的推理，都要实测 Y。**

硬件层面的"独立单元"几乎总共享某一层资源：copy engine 和 SM 共享 HBM，不同 SM 共享 L2，不同卡共享 NVLink switch。只要有共享，就会有干扰，只是程度不同。

另外一点：这次是因为**数字看起来不对**（2.31× 太明显了）才去查的。如果干扰只有 1.05 倍，很可能就当成噪声忽略了。所以 Stage 4 的 TPOT 测量要有足够的采样和分位数统计，不能只看均值。

---

## 两个 bug 对比

| | 性质 | 发现方式 | 后果 |
|---|---|---|---|
| Bug 1 边界检查 | 代码错误（参数语义漂移） | 单元测试 | 误拦合法调用，立即可见 |
| Bug 2 带宽干扰 | **认知错误**（假设未验证） | 实验数据异常 | 会让 Stage 4 的曲线出现无法解释的抖动 |

Bug 1 是典型的"改了一半"——加功能时改了调用方没改被调方。Bug 2 性质完全不同，它不是代码写错，是**设计时的物理模型错了**。

累计到 Stage 3，八个 bug 的分类：

- **目标函数没对齐**：S1-1（代理指标）、S1-3（算错目标）
- **内部状态与输出不一致**：S1-4（容量溢出）
- **测量方法错**：S2-1（异步当同步）
- **实现偏离设计意图**：S2-2（全局 cursor）
- **参数语义漂移**：S3-1（slot_bytes → nbytes）
- **物理模型假设错**：S3-2（copy engine 干扰）
- 会报错的只有 S1-2（槽号越界）和 S3-1

发现手段的有效性，到目前为止：

1. **实测验证假设**（抓到 S3-2）—— 写代码前先验证硬件行为
2. **端到端指标异常**（抓到 S1-3）
3. **对照实验**（抓到 S1-4）
4. **人工 review**（抓到 S2-2）
5. **量纲自洽检查**（抓到 S2-1）
6. 单元测试（抓到 S1-2、S3-1）

单元测试能抓的都是"会报错"的那类。真正危险的静默错误，靠的是前五种。
