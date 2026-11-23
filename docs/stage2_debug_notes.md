# Stage 2 调试记录

两个 bug。第一个是测量方法错，第二个是设计错——而且是用户看代码时发现的，不是测试发现的。

---

## Bug 1：计时测的是入队，不是执行

### 症状

第一版脚本输出：

```
40-layer batch (252 MB): 53 replays overlapped, 0.5 ms wall, zero blocked steps
```

同一个脚本里另一处写着单次 `replay()` 是 2.226 ms。**53 次 × 2.226 ms = 118 ms，怎么可能是 0.5 ms？**

数字自相矛盾，但看起来很漂亮（"53 次推理和拷贝完全重叠"），差点就这么写进文档了。

### 定位

直接测三种写法：

```
with sync per 100:  2.219 ms/replay    ← 100 次后统一同步
sync each:          2.230 ms/replay    ← 每次都同步
enqueue only:       0.009 ms/replay    ← 不同步
```

差了 **247 倍**。

### 根因

CUDA 的 kernel launch 是**异步**的。`graph.replay()` 只是把整张图的 kernel 序列塞进 stream 队列就立刻返回，GPU 什么时候真正执行是另一回事。

所以原来的轮询循环：

```python
while not ev_bulk.query():
    shard.replay()          # 0.009 ms 入队，立刻返回
    bulk_spins += 1
```

53 次迭代其实是**往队列里塞了 53 张图**，GPU 可能连第一张都没跑完。拷贝完成时 `query()` 返回 True，循环退出，而那 53 个"decode step"还在队列里排队。

这个测量根本没有证明"推理不被阻塞"——它只证明了"CPU 能很快地提交任务"。

### 修复

每次迭代都真同步，确保测的是完整的 GPU 执行：

```python
while not ev_bulk.query():
    t0 = time.perf_counter()
    shard.replay()
    torch.cuda.synchronize()     # 等这一步真跑完
    step_ms.append((time.perf_counter() - t0) * 1e3)
    overlapped += 1
```

修复后输出变成：

```
40-layer batch (252 MB): overlapped 1 full decode steps, step time 3.97-3.97 ms
```

数字小了很多（1 次而不是 53 次），但这才是真的：**252 MB 的拷贝和一个完整的 decode step 重叠，那个 step 仍然只花 3.97 ms，没有被拖慢。**

同时在计时表里把两个数都列出来，提醒后面读代码的人：

```
graph.replay() GPU       1.733 ms   (one decode step, synced)
graph.replay() launch    0.009 ms   (CPU-side enqueue only)
```

### 教训

**GPU 上凡是计时，没有 `synchronize()` 就是在测 CPU 提交速度。**

更值得警惕的是：这个错误产生的数字**方向是对的**（确实重叠了）、**量级是错的**（53 vs 1），而且错的方向刚好对我们有利。如果不是两处数字对不上，很可能就蒙过去了。

所以凡是"结果特别漂亮"的测量，先做一次量纲检查：53 次 × 2.2 ms 应该是 118 ms，和 0.5 ms 差两个数量级，这种矛盾是最好的报警器。

---

## Bug 2：全局 cursor —— 副本复制实际没生效

### 发现过程

这个 bug 不是测试发现的，是**用户读代码时问出来的**：

> 理论上不是应该一个专家一个 cursor 吗才是 round robin，你的 cursor 不是一个一维的值吗

问得对。当时的实现：

```python
self.cursor = torch.zeros(1, dtype=torch.int32, device=device)   # 一个标量
...
k = self.cursor.long() % cnt.clamp(min=1)
```

`cursor` 是**全局标量，所有专家共享**。

### 后果

同一拍内，所有 token 算出来的 `k` 都一样。比如 `cursor=0`：

```
64 个 token 路由到专家 23（它有 2 个副本）
→ 所有 token 的 k 都是 0
→ 全部挤到 replica 0 所在的卡
→ replica 1 完全空闲
```

要等下一拍 `cursor=1`，才轮到 replica 1 承担全部负载。

### 为什么这个偏差很严重

Stage 1 的规划器假设"专家有 c 个副本，每个承担 w/c"。全局 cursor 下，这个假设**只在多拍平均的意义上成立**，单拍内始终是 100% 压在一个副本上。

而 **MoE 的瓶颈是单拍内最慢的那张卡**。all-to-all 必须等最慢的卡算完，其他卡空等。所以：

- 平均下来 w/c：好看
- 每一拍都是 w/1：实际延迟一点没改善

**复制等于白做。** Stage 1 那些 Max/Mean 数字在运行时根本兑现不了。

### 为什么测试没抓到

当时的 `test_replica_round_robin`：

```python
for c, w in [(0, wa), (1, wb), (2, wa), (3, wb)]:
    sh.set_cursor(c)
    y = sh.replay()
    assert torch.allclose(y[0].float(), ref(x[0], w).float())   # 只看 token 0
```

它验证的是"**改变 cursor 会切换副本**"。坏的实现完全满足这一点——全局 cursor 确实能在 step 之间轮换副本。

漏掉的性质是"**同一拍内，多个 token 要落到不同副本**"。测试只检查了 `y[0]` 一个 token，根本不可能发现分散问题。

### 修复的难点：不能照字面实现

正确的 round-robin 直觉上是这样：

```
token A → 读 cursor[5]=0 → 用副本 0 → cursor[5]=1
token B → 读 cursor[5]=1 → 用副本 1 → cursor[5]=2
```

但这**隐含顺序执行**。GPU 上 64 个 token 是同时路由的，所有线程会同时读到 `cursor[5]=0`，还是全选副本 0。

要真串行化就得用 atomic，而 atomic 的执行顺序不确定，结果不可复现——CUDA Graph 里更没法保证。

### 解法：把"我是第几个"算出来

关键转换：**"token B 是第 2 个访问专家 5 的"这个信息不需要累加得到，可以直接计算。**

定义 `rank[t]` = 在 t 之前、且路由到同一专家的 token 数量。这是个纯函数，可完全向量化：

```python
same = expert.unsqueeze(1) == expert.unsqueeze(0)   # [T, T] 谁和谁同专家
rank = same.tril(diagonal=-1).sum(1)                # [T] 我前面有几个同专家的
```

`tril(diagonal=-1)` 取严格下三角，排除自己。

然后组合成最终公式：

```python
k = (self.cursor[e] + rank) % cnt.clamp(min=1)
```

两部分分工明确：

| 量 | 作用 | 维度 |
|---|---|---|
| `rank` | 本拍内分散 | 由 batch 内容算出 |
| `cursor[e]` | 跨拍轮转余数 | per-expert，持久化 |

### 为什么 cursor 还必须保留

考虑 3 个 token、2 个副本：

```
rank = 0, 1, 2  →  k = 0, 1, 0
副本 0 拿 2 个，副本 1 拿 1 个     ← 有余数
```

`advance_cursor` 把 `cursor[e]` 前进 3，下一拍起点变成 1：

```
k = (1 + [0,1,2]) % 2 = 1, 0, 1
副本 0 拿 1 个，副本 1 拿 2 个     ← 余数轮到对方
```

所以 cursor 负责把除不尽的那部分在时间上抹平。

`advance_cursor` 用 `index_add_` 一次算完，不需要逐个加：

```python
self.cursor.index_add_(0, e, torch.ones_like(e))
```

### 验证

实测 64 个 token 全部路由到同一个有 2 个副本的专家：

```
64 tokens all on expert 23 -> slots [32, 33], counts [32, 32]
spread within one step (not just across steps): True
```

完美对半分。

补了三个测试，都是针对原来漏掉的性质：

| 测试 | 断言 |
|---|---|
| `test_batch_spreads_across_replicas` | 8 token / 4 副本，**单拍内**必须 2/2/2/2 |
| `test_cursor_carries_remainder_across_steps` | 3 token / 2 副本，必须从 2:1 翻转成 1:2 |
| `test_advance_cursor_counts_tokens` | 游标前进量 == 该专家的 token 数 |

### 教训

**测试写的是"功能能用"，不是"功能对"。**

`test_replica_round_robin` 测的是"cursor 变了输出会变"，这是功能的**表象**。真正要保证的是"负载被分散"，这是功能的**目的**。表象对了但目的没达成，测试照样全绿。

还有一点：这个 bug 是**人读代码发现的**，不是任何自动化手段发现的。它不违反任何结构不变量，不产生任何异常，性能数字也看不出异常（因为单卡实验里没有跨卡负载对比）。要到 Stage 4 的 8 卡 TPOT 压测才可能暴露——那时候排查成本就高多了。

所以"让人读一遍核心逻辑"仍然是不可替代的手段，尤其对这种"实现偏离了设计意图"的问题。

---

## 两个 bug 的共性

| | 表现 | 测试能发现吗 | 靠什么发现的 |
|---|---|---|---|
| Bug 1 计时 | 数字漂亮但自相矛盾 | 否（不是正确性问题） | 量纲检查：53 × 2.2 ≠ 0.5 |
| Bug 2 全局 cursor | 功能"能用"但目的没达成 | 否（测了表象） | **人工 review** |

和 Stage 1 的四个 bug 合起来看，六个里有五个是"安静地给出错误结果"。分类：

- **目标函数没对齐**：Stage 1 Bug 1（代理指标）、Bug 3（算错目标）
- **内部状态与输出不一致**：Stage 1 Bug 4（容量溢出）
- **测量方法错**：Stage 2 Bug 1（异步当同步）
- **实现偏离设计意图**：Stage 2 Bug 2（全局 vs per-expert）

只有 Stage 1 Bug 2（槽号越界）是会报错的那种。

发现手段的有效性排序，从实践看是：

1. **端到端指标异常**（抓到 Stage 1 Bug 3）
2. **对照实验**（抓到 Stage 1 Bug 4）
3. **人工 review**（抓到 Stage 2 Bug 2）
4. **量纲/自洽检查**（抓到 Stage 2 Bug 1）
5. 结构性单元测试（只抓到 Stage 1 Bug 2）

单元测试排在最后，这个结果值得记下来。
