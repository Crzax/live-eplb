# Stage 5 调试记录

四个，最后一个才是真的值得记的，和前四个 Stage 抽出来的类型完全不一样——
这一阶段的问题全在"环境"和"手快"，没有一个在机制本身。

---

## S5-1：列表推导的变量不会泄漏

### 症状

第一次采集跑到 Phase B 收尾时 `NameError: name 'sel' is not defined`，
直接被你从日志里抓了出来。

### 根因

```python
c = torch.stack([torch.bincount(sel.flatten().long(), ...) for sel in picked])
counts = c if counts is None else counts + c
ntok += sel.numel()          # Pick 3 以后这是未定义变量：推导式作用域不外泄
```

### 教训

crashed loudly、一修就好（`ids.numel() * 8`），但**这行能过第一次提交前的自测**
靠的是它恰好写在循环尾部——采集脚本这种"跑一次就扔"的代码恰恰是静态检查最少覆盖
的部分。以后这类快脚本写完先在 3 条 prompt 上跑个冒烟版，再上完整 48 条——
这条规则本来就有，统计一下：这是第三次没守。

---

## S5-2：和正在跑的作业抢文件

### 症状（没酿成事的）

`--trace` 刚加进 `run_synthetic_8gpu.py` 就 nohup 起了全套（40 层 + 1 层），
马上发现 1 层分支会拿 `[40, 256]` 的 counts 去匹配 `L=1` 的 ids——函数里没按
`args.layers` 切片。40 层在跑的正是同一份文件，但 1 层的 torchrun 是 40 层结束
之后才被 shell 拉起来的。

### 处理

在 40 层还在跑的时候原地改文件（`counts[:L]`），赌 1 层进程启动时读到的是修后的
文本——**赌赢了**，但这是一次把"竞态"当工作流用的操作，不该是常态。

### 教训

正确姿势是写完文件、只在你确定没有副作用的粒度上起跑后台作业；或者把两个
torchrun 拆成两条命令、先跑小的那个验证路径。**能赌对不等于该赌。**

---

## S5-3：`transformers_version: 4.57.1` 说它能跑，4.57.1 说它没有这个类

### 症状

`ModuleNotFoundError: No module named 'transformers.models.qwen3_5_moe'`。
反复核对了装的是 config 声称的 4.57.1，依然缺。

### 根因

模型目录的 `config.json` 写 `transformers_version: 4.57.1`，但那是**保存时**的版本
字符串，`Qwen3_5MoeForConditionalGeneration` 这个类在 4.57.1 里还没进主干
（只有 qwen3_moe / qwen3_next 等一族）。它在 5.x 才成为公开模块，升到 5.24.1
后一切正常。

### 教训

**`transformers_version` 是"写文件用的版本"，不是"跑这个 checkpoints 需要的最低版本"。**
以后看到缺模块先去镜像上扫一遍最高版本里有没有这个包名，再考虑自己拼 shim。
另外 `config.json` 里没有 `auto_map`、模型目录里没有 `.py`——这类 checkpoint
只能靠 transformers 主干认它，识别不出来时 trust_remote_code 也救不了你。

---

## S5-4：小样本会同时往两个方向骗你

### 症状

先交的 48 条手写 prompt 被"样本太少、结论怎么会鲁棒"的批评打回。重采 4066 条标准
数据集之后，**同一个估计值沿完全相反的两个方向移动了**：

- 层平均倾斜（Max/Mean mean）：A 1.680 → 1.527，B 1.605 → 1.642（不变）；
- 收敛窗口峰值差：**0.20 → 0.31（更差）** —— 数学域最热层在小样本里没被采到；
- 增量规划后的最差单层：**1.330 → 1.263（更好）** —— 尾部专家的统计被点亮后，
  规划器能用的信息更多了。

### 根因与教训

48 × 90 次指派摊到 10240 个 (层, 专家) 格子上，**头部几十个专家以外的格子全是
17-90 次的泊松噪声**。薄样本不是"不精确"，是**系统性偏向低估尾层倾斜、同时
系统性高估规划器的覆盖质量**——因为你只看到你采到的。

我以前把"样本太少"写进过局限就以为这事结了。教训是：**写进"已知局限"不是保险，
该修的鲁棒性还是要修**；只有等量级实测（这里是 85×）把两条方向相反的误差都拉出来，
才知道自己当时谎报了多少。

---

## S5-5：全量重排的应用是多阶段操作，不是一次拷贝

### 症状

给 benchmark 加第五个对照 `stw_official`（官方 EPLB 规划器 + 全量重排 + staging +
重录）之后，权重校验随机失败：122 / 21 / 15 个槽 tag 错位，每次跑的数量都不一样，
而且失败的（层, 专家, 槽）三元组每次换一批。

三段排错各自都"装对"过：任务生成在 CPU 复算 0 错位、staging 的标签 dump 全对、
而 pool 最终的内容仍能错。**"过程每一环抽查都对，结果随机坏"是跨进程竞态的
指纹。**

### 根因

官方重排**不把源槽冻结**：一个 expert 的权重可以在任意 GPU 的任意槽。benchmark 的
应用路径是"各 rank 分头把上一个布局的权重拉进本地 staging，再写回本地 pool"——
但 rank A 的 pull 和 rank B 的 pool 写之间没有任何同步，B 完全可能把别的权重
写进 A 正在拉的槽：

```
rank 0: 正在 cudaMemcpyPeerAsync 从 rank3/pool13 拉 expert 109
rank 3: 与此同时 pass2 在把 expert 49 写进自己的 pool13
        → rank0 的 staging 里是半新半旧的破坏内容 (tag=49)
```

我们自己的全部对照（stw / stw_inplace / live）**天然免疫**，因为 Pinned-Base 把
源槽钉死在冻结的 base 区——任何进程都不会写一个"别的 rank 可能正在读"的槽，
于是就根本不需要任何集群级两阶段协议。官方方案做不到这件事，**它的"拿到全局
最优"的成本里就包括这个 barrier**。

### 修复

两次 barrier + 一次本地 stream 清空：

```
pull staging 前: dist.barrier()        # 所有人先把任务算完
pull staging 后: s.synchronize()        # 本 rank 的读全部落地（GPU 上，不是 CPU 上）
写 pool 前:     dist.barrier()        # 全集群的读都落地了，才准开始写
```

**注意中间那一次 `s.synchronize()` 不能省**——`barrier()` 只在 CPU 上排队，CA
异步 peer copy 还在 GPU 上飞的时候就开始写，一样是撞。

### 教训

全量重排的成本模型有"两台发动机"：一次拷贝量、一次两阶段同步。我们一直量的
是前一台（GB）；"后一台"一直被 EPLB 的项目代码藏在 prepare/activate 状态机里。
这次写基准把后一台也逼到了明面上——也正是为什么 Pinned-Base 愿意用 1.091 的均衡
上限去换：**它把第二台发动机整个拆掉了**。
