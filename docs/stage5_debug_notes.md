# Stage 5 调试记录

三个，都不隐蔽、不阴间，和前四个 Stage 抽出来的类型完全不一样——
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
