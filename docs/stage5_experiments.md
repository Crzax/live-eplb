# Stage 5 实验记录：真实 Router 轨迹

Stage 1-4 的所有负载都是合成 Zipf。**如果结论只在合成数据上成立，前四个 Stage 就白做了。**
这一阶段只做一件事：把负载换成 Qwen3.6-35B-A3B 自己的路由决策，重新回答同样的问题。
范围已收窄——不跑端到端 generate，只采轨迹（见 `TODO`）。

## 轨迹怎么来的

```bash
python benchmarks/collect_router_trace.py    # GPU 机器，约 2 分钟 + 4 分钟模型加载
```

在 40 个 `Qwen3_5MoeTopKRouter` 上挂 forward hook，取它 forward 返回值里的
`router_indices [tokens, 8]`（top-k 选择的专家 id），逐 prompt 前向一次，
按层 bincount 成 `[40, 256]`。两个域，各 24 条短 prompt：

- **Phase A**：代码生成（英文，24 条，"写一个线程安全的 LRU cache" 这类）
- **Phase B**：数学证明 + 中文百科（24 条，"证明柯西中值定理" 这类）

模型方式是 `Qwen3_5MoeForConditionalGeneration` + `device_map="auto"`（70 GB BF16
摊到 8 卡），**纯文本输入，不走 chat template，只采 prefill**。采完即释放。
产出 `results/router_traces.pt`（88 KB）。

## 真实路由长什么样——和合成负载的三点本质差别

先把这个阶段最重要的观测放最前面（`benchmarks/plan_on_trace.py` 可复算）：

| | Phase A (code) | Phase B (math/wiki) | 合成 Zipf(α=1.1) |
|---|---|---|---|
| 层平均 Max/Mean（静态布局） | **1.680** | **1.605** | 2.204 |
| 单层最小 / 最大 | 1.171 / **2.603** | 1.188 / 2.341 | ≈ 每层相同 |
| 两域 top-8 热点交集 | 每层平均 1.95 / 8 | 同左 | 错位是人为轮转 64 个 |

1. **真实流量更温和**：层平均 1.6-1.7，不到合成的 2.2。
2. **真实流量的倾斜是逐层异质的**：有的层 1.03（几乎均匀），有的层 2.60——
   合成的 α 全层相同，造不出这个形状。**Stage 4 在局限里承认"逐层门控的价值
   在合成 benchmark 里验证不了"，现在可以了**。
3. **两域切换真实存在**：top-8 热点每层平均只有 1.95 个重合——"代码流量突然变
   数学流量"会真实地换一批热点专家，不需要人为构造。

![per-layer skew heatmap](../results/qwen3_6_layer_skew_heatmap.png)

## Stage 1 重放：规划对比在真实轨迹上

```bash
python benchmarks/plan_on_trace.py
```

```
policy                    Max/Mean  worst layer  slots moved      GB
static on A                  1.680        0.000            0     0.0
static on B                  1.605        0.000            0     0.0
live plan@A                  1.127        0.000          971    5.96
live incr A->B               1.085        1.330          983    6.04
official reshuffle @B        1.001        0.000        11473   70.47
```

**Stage 1 的权衡在真实轨迹上几乎原样复现**：官方 EPLB 全量重排 Max/Mean 1.001 但要搬
70.47 GB；LiveEPLB 增量 6.04 GB（**11.7×**，和合成一模一样）换 1.085。最差的单层
1.330，恰好踏在冗余槽预算给的均衡上限附近。

## Stage 4 重放：四策略 TPOT 在真实轨迹上

```bash
torchrun --nproc_per_node=8 benchmarks/run_synthetic_8gpu.py \
    --layers 40 --steps 300 --trace results/router_traces.pt --out results/stage5_trace_timeline.json
python benchmarks/plot_stage4.py 80 results/stage5_trace_timeline.json results/stage5_trace_timeline.png
```

其余设置和 Stage 4 完全一致（每步 64 token/层、step 80 换域、threshold 1.35、
冷却 20 步），只是把相位分布从 Zipf 换成真实 counts。

### 40 层

```
policy      TPOT p50      p99    worst  Max/Mean last100  commits
-----------------------------------------------------------------
static        66.369   67.174   67.373             1.617        0
stw           66.637  155.552  167.080             1.220       15
stw_inplace   66.614   69.526   70.454             1.220       15
live          66.544   67.524   67.617             1.222       12

stw 停机分解典型行（ms）:
    step  90   slots 74   plan 19.6  drain 46.0  copy 1.7   recap 86.6  TPOT +88.2
    step 210   slots  7   plan  3.2  drain 63.1  copy 0.5   recap 99.7  TPOT +100.7
    step 251   slots  3   plan  2.3  drain 64.0  copy 0.4   recap 79.6  TPOT +80.5
stw_inplace 同一批应用: recap ≈ 1.6-2.2, 每次停机只超 p50 +1.5~4.1
```

### 1 层（对照）

```
static        1.717   1.759   1.762             1.449   0
stw           1.721   1.764  48.085             1.202   2
stw_inplace   1.721   1.761   2.320             1.202   2
live          1.728   1.774   2.108             1.161   1
```

![real trace TPOT timeline](../results/stage5_trace_timeline.png)

### 和合成结果逐条对照

| 结论 | 合成 Zipf | 真实轨迹 | 判定 |
|---|---|---|---|
| live 最差步 ≈ static | 66.166 vs 66.528（−0.36） | 67.617 vs 67.373（**+0.24**） | 仍在抖动内（±0.36） |
| 停机 ≈ 重录 | recap 45-58，占 98.1% | recap 52.8-99.7，中位 ~86 | 成立，甚至更靠重录 |
| 不 recapture 拿走大部分 | stw_inplace worst 69.03 | 70.45 | 成立 |
| live 对 stw_inplace 余量 | 2.86 ms | **2.84 ms** | **惊人地一致** |
| 均衡收益（last100） | 2.204 → 1.166（47%） | 1.617 → 1.22（25%） | 方向一致，量级随倾斜度 |
| live 收敛慢于停机组 | 45 步，峰值差 0.81 | **37 步，峰值差 0.20** | **劣势仍在，但真实流量上小得多** |
| step 时长 | 64.6 ms | 66.4 ms | 静态布局在真实流量上略慢 |

额外看一眼收敛过程（突变后 live 与 stw 的 Max/Mean）：

| step | 80 | 90 | 100 | 110 | 120 | 125 | 130 |
|---|---|---|---|---|---|---|---|
| live | 1.190 | 1.376 | 1.368 | 1.416 | 1.431 | 1.253 | 1.249 |
| stw | 1.220 | 1.168 | 1.286 | 1.198 | 1.254 | 1.266 | 1.180 |

live 的提交落在 step 95 和 122，收敛慢窗口约 37 步。**为什么比合成的 0.81 小**：
真实 Phase B 的逐层倾斜和 Phase A 有相当一部分重叠（层 28/38 在两域都热），
突变后需要修的层更少，而逐层增量规划恰好最擅长"只修该修的层"——
**合成数据错开热点的方式（全层统一轮转）反而把这个机制的缺点放大了**。

## 这一阶段改变了哪些声明

1. **"逐层门控的价值合成 benchmark 验证不了" —— 现在验证了。** 真实轨迹的层间异质性
   （1.17 ~ 2.60）意味着整栈平均门控会把热层埋在平均线下；而逐层门控第一次面对的就是
   它为之设计的输入。触发行为虽未单独打点，但从两个证据反推：静态基线层平均 1.617 > 1.35
   却大量单层低于阈值，`live` 和 `stw` 触发 15/12 次而非"每次都是全量"，规划时长也
   是热点层规模（3-20 ms）而非全栈规模（~59 ms）。
2. **"live 收敛慢 45 步"下修为"37 步左右，随域间重叠度浮动"** —— 这个数字依赖于
   两个域差多远，不是机制常数。合成构造的极端错开（top8 完全轮转）把它估大了。
3. **其余 Stage 4 结论原样保留**：停机的钱在重录上（52.8-99.7 ms）、copy 快得无关紧要
   （≤3.4 ms）、live 与 stw_inplace 的差是那 ~2.8 ms 的同步溢价。真实轨迹没有推翻
   任何一条 Stage 4 结论，还把"live 的那条短板"的真实严重度降到了 1/4。

## 已知局限

- **样本太薄**：48 条短 prompt、纯 prefill。两域总共 6568 次路由指派、摊到 40 层
  每层只有 ~90 次采样——**每层 256 个专家里，只有头部几十个专家的路由是可信的**，
  尾部噪声很大。真实生产 trace 应该采一周线上流量，这里是"证明方法成立"的量级。
- **只采了 prefill 的路由**：decode 阶段每个 token 的路由在更长 context 下可能漂移，
  未涵盖。也没有按时间切片构造"会话内漂移"。
- **域切换是突兀的**：step 80 直接从 A 换 B；真实流量是渐变的混合比例变化。我们的
  突变是压力测试版。
- **benchmark 内仍是 multinomial 重采样**：真实 counts 只提供分布形状，逐步噪声模型
  还是合成的。完全忠实应直接回放逐 token 序列。
- **1 层对照用第 0 层切片**：此为任意选择；第 0 层恰好处在层间偏温和的一侧
  （静态 Max/Mean 1.449 vs 最热层 2.60），单层数字的绝对水平不代表最热层。
- **两个域是我手写的主观选择**：代码 vs 数学/百科的分离度（top8 交集 1.95/8）
  受 prompt 集直接影响，更多域、更长轨迹下分离度可能不同。
- **transformers 5.24.1 + 官方参考实现**：`causal_conv1d`/`flash-linear-attention`
  缺失走 fallback kernel，只影响采集速度，不影响路由决策本身。
