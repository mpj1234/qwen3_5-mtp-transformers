# 给 Qwen3.5 接上 DFlash：一次猜一块 Token，为什么草稿 Cache 不用回退？

## 为什么又写一篇投机解码

上一篇 [MTP 文章](ZHIHU_MTP_ARTICLE.md) 写到最后，最绕的地方是“两套 cache 都可能走过头，所以要各自回退”。把同样的问题带到 DFlash，我一开始也以为：既然草稿模型同样猜了几个 token，它的 cache 当然也要保存快照、验证失败后再裁掉。

写完 Qwen3.5-4B 的 [DFlash 推理示例](demo_dflash.py)，我发现答案取决于**究竟把什么写进草稿 cache**。这份实现只缓存主模型已经确认的上下文；当前这一轮的候选只参与临时计算。于是主模型验证失败时要回退，草稿 cache 却没有错误候选可回退。

这篇文章从一轮推理开始，顺着 `prefill → draft → verify` 走一遍，再解释此前容易卡住的几个问题：`output_hidden_states=True` 为什么会返回 33 项？为什么上下文只算 K/V、不算 Q？第一次和第二次调用 `cache_context` 的序列长度是多少？`cache.length` 为什么只增不减？

本文讲的是**单条输入、贪心解码**的教学实现。采样模式需要另外的接受与拒绝规则；“一次草稿前向”也不等于一定更快，实际收益要用目标硬件测量。DFlash 的模型与方法可参考 [项目仓库](https://github.com/z-lab/dflash) 和 [Qwen3.5-4B-DFlash 模型卡](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash)。

## 先用一句话看懂这一轮

假设 Qwen3.5 已经读完提示词，预测出下一个 token `t0`。注意：`t0` **尚未作为输入经过主模型**，所以主模型 cache 仍停在提示词末尾。

DFlash 拿 `t0` 当作起点，在它后面放几个特殊的 `MASK` 位置，一次前向并行猜出候选 `d1、d2、d3`。主模型再把 `[t0,d1,d2,d3]` 当成已知输入，一次前向验证：读完 `t0` 是否会选 `d1`，读完 `d1` 是否会选 `d2`，依次比较，遇到第一个不一致就停。

```text
提示词 p0...p18  ──主模型 prefill──>  t0（pending）
       │                                │
       └──若干层 hidden ──> DFlash      └─> [t0, MASK, MASK, MASK]
                                    一次草稿前向 ──> [d1, d2, d3]
                                                    │
主模型 cache + [t0, d1, d2, d3] ──一次验证───────────┘
                         │
                         └─只接受连续正确的前缀，回退主模型多算的后缀
```

这个流程是“无损”的条件很具体：本文使用贪心解码，只有与主模型自身贪心预测相同的草稿才会被接受。它不靠草稿模型改变最终答案，只尝试减少主模型逐 token 的串行轮数。

## 先认清手里的两个模型和 `block_size`

这里的主模型是 `Qwen/Qwen3.5-4B`，草稿权重是 `z-lab/Qwen3.5-4B-DFlash`。后者**不是能独立聊天的 4B 模型**；它是与指定主模型配套的草稿模型，需要主模型的中间层特征、词嵌入和输出头。模型卡也明确写了这个配对关系。[模型卡](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash)

本地草稿配置有 6 层：前 5 层是 `sliding_attention`，最后 1 层是 `full_attention`；滑动窗口为 4096。`target_layer_ids` 是 `[1,5,9,13,17,21,25,29]`，表示要读取主模型的 8 个指定层。`dflash_config.block_size` 是 16。[配置文件](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash/blob/main/config.json)

**这里要纠正一个很容易说错的数字：**在本文 demo 中，`block_size` 是整个草稿输入块的长度，包含已经由主模型选出的 anchor `t0`。代码构造的是 `[t0, MASK × (block_size-1)]`，只对 `hidden[:, 1:]` 计算候选 logits。因此 `block_size=16` 表示**一次草稿前向提出最多 15 个新候选 token**；运行时设 `--block-size 4`，就是一次提出 3 个候选。它不是让草稿模型连续自回归调用 16 次。

这与上一篇的 MTP 不一样：那份单层 MTP demo 要连着调用多次草稿层，才能依次得到 `d1、d2、d3、d4`。DFlash 的 `MASK` 块则在一次草稿前向里并行给出各位置预测。

## 第一件事：主模型 prefill 为何要打开 `output_hidden_states`

主模型 prefill 做两件事：把提示词写进主模型 cache，并用最后位置的 logits 选出 `pending=t0`。DFlash 还需要主模型指定层在**提示词每个位置**上的隐藏向量，所以调用时传 `output_hidden_states=True`：

```python
outputs = target(
    input_ids=prompt_ids,
    past_key_values=cache,
    use_cache=True,
    output_hidden_states=True,
    logits_to_keep=1,
)
pending = outputs.logits[:, -1:, :].argmax(dim=-1)
features = torch.cat(
    [outputs.hidden_states[i + 1] for i in target_layer_ids], dim=-1
)
```

有人看到 `len(outputs.hidden_states) == 33` 会问：多出来的第 33 项是不是 logits？**不是。** Qwen3.5-4B 文本主模型有 32 个解码层；`hidden_states[0]` 是进入第 0 层前的 embedding 输出，后面对应 32 层的输出，所以总共 33 项。Logits 在单独的 `outputs.logits` 字段，由最终隐藏向量通过 `lm_head` 得到。代码中的 `i+1` 正是为了跳过第 0 项 embedding 输出。最后一项还会按 Transformers 的输出捕获逻辑与最终归一化结果对齐；本文所选的最高层是 29，不涉及这一项。[Qwen3.5 模型实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py) · [输出捕获实现](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/output_capturing.py)

这 8 层特征拼接后，形状是 `[batch, prompt_length, 8 × 2560]`。DFlash 的 `fc` 把最后一维投影回草稿 hidden size 2560，随后 `hidden_norm` 做 RMSNorm：

```python
context = draft_model.hidden_norm(draft_model.fc(target_features))
```

为什么是**先 `fc`、再 `hidden_norm`**？因为草稿模型就是按这个结构训练、保存权重的。先拼接并投影，得到草稿模型宽度的向量，再对这个投影结果归一化。把顺序换过来不是等价的代数变换，也不再对应现有权重。[DFlash 参考实现](https://github.com/z-lab/dflash/blob/main/dflash/model.py)

## 第二件事：DFlash 的“上下文预填充”到底算了什么

上一篇 MTP demo 有显式的 `prefill_mtp()`。这份 DFlash demo 没有独立的 `prefill_dflash()` 函数：首次 `draft()` 会把完整提示词特征交给 `DFlashDraft.forward()`；以后每轮则只交给它上一轮新接受的 token 特征。函数名可以合并，但**为草稿模型建立上下文状态这件事不能省**。

`DFlashDraft.forward()` 先做 `fc → hidden_norm`，再让每个草稿层调用 `cache_context()`：

```python
for index, layer in enumerate(self.layers):
    layer.self_attn.cache_context(context, context_cos, context_sin, cache, index)
cache.length += context.shape[1]
```

`cache_context()` 只为主模型已经接受的上下文计算 **Key 和 Value**，并把它们保存到该草稿层的 cache。它不为这些历史位置计算 Query，也不把这些位置重新送过完整的草稿 Transformer 层。原因很直白：本轮要预测的是新 `MASK` 位置，历史位置只需要提供“可被读取的资料”；真正发起查询的是本轮的 anchor/MASK 块。

这回答了“是不是等同于 MTP 的 prefill”：**目的相似，计算不同。** MTP prefill 要用移位的 token 与主模型 hidden 执行 MTP 层前向；DFlash 在这里把主模型特征投影为各草稿层需要的 K/V。它们都为后续草稿准备历史，但不是相同的一次模型计算。

### 第一次、第二次调用的 `context` 多长？

默认提示词“你好，请用一句话介绍你自己。”经过 Qwen3.5 聊天模板后，本地 tokenizer 得到 **19 个 token**。第一次 `draft` 时，`context.shape == [1,19,2560]`，每个草稿层都要为这 19 个位置建自己的 K/V。

“第二次调用”有两种数法，别混在一起：

| 你说的第二次 | `context` 序列长度 |
| --- | --- |
| 同一轮 `draft` 中，第二个草稿层调用 `cache_context` | 仍是 19；六层各调用一次，处理的是同一段提示词 |
| 第二轮 `draft` 再进入 `cache_context` | `1 + 上轮接受的草稿数`；1 是主模型先前选出的 `pending` |

例如上一轮 3 个草稿中有 2 个连续猜对，新接受 `[t0,d1,d2]`，下一轮 `context` 序列长度就是 3。原来的 19 个提示词 K/V 已在草稿 cache，不会再重复投影。自定义提示词时，首轮长度也会随分词结果变化。

## 为什么 `forward` 里看着有“两次 forward”？

看代码容易误以为 DFlash 对同一段内容跑了两遍完整模型。实际上这是两类不同的工作：

1. `cache_context()`：为**已接受的主模型上下文**，在每个草稿层只计算 K/V，写入持久 cache。
2. `layer.forward()`：让本轮 `[t0,MASK,...]` 计算 Q 和临时 K/V。它的 Q 同时读取“缓存的上下文 K/V + 本轮块内 K/V”，随后做注意力、MLP 和残差，最后预测 `MASK` 位置。

下面这行特别重要：

```python
k = torch.cat([cache.keys[layer_idx], k], dim=-2)
v = torch.cat([cache.values[layer_idx], v], dim=-2)
```

它只是把缓存和本轮临时 K/V 拼成**局部变量**，供这次注意力读取；代码没有把拼接结果重新赋给 `cache.keys[layer_idx]` 或 `cache.values[layer_idx]`。所以本轮猜出的 token 并没有“偷偷进入草稿 cache”。

前 5 个滑动注意力层还会用因果加窗口 mask：只能看到当前位置及左边、且距离小于 4096 的 Key。最后的全注意力层在这个草稿块里可以双向看块内位置。虽然草稿块能并行互相参考，最后仍由**因果的主模型**逐位置验证，只有连续正确前缀才进入正式输出。这些注意力规则来自草稿权重对应的层结构，不能任意换掉。[DFlash 参考实现](https://github.com/z-lab/dflash/blob/main/dflash/model.py)

### 共享词嵌入与 `lm_head`，还要看缩放参数

本轮输入是 `[t0,MASK,...]`。示例用主模型原始 embedding 表做查找，避开主模型常规 embedding 路径中的额外归一化；草稿层输出后，也复用主模型的 `lm_head` 把 `MASK` 位置的 hidden 转成 logits。

参考实现还允许配置 `input_embedding_scale`、`output_multiplier` 和 `final_logit_softcapping`。示例从 `dflash_config` 优先读取，也兼容顶层字段：输入 embedding 乘第一个倍率，输出头后的 logits 乘第二个倍率；若 softcap 为正，再做 `tanh(logits/softcap) × softcap`。本地 Qwen3.5-4B-DFlash 配置没有这三个字段，因此使用 `1.0、1.0、不做 softcap` 的默认行为。[DFlash 参考实现](https://github.com/z-lab/dflash/blob/main/dflash/model.py) · [模型配置](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash/blob/main/config.json)

## 第三件事：主模型怎样验收一整块草稿

`verify()` 把 `[pending, proposed...]` 整块交给主模型，各解码层只做一次块前向。假设草稿是 `[d1,d2,d3]`，主模型产生的位置对应关系是：

```text
主模型读完 t0 的 logits  →  应该等于 d1
主模型读完 d1 的 logits  →  应该等于 d2
主模型读完 d2 的 logits  →  应该等于 d3
```

从左到右只接受**连续匹配**的前缀。如果 `d1、d2` 对，`d3` 错，接受 `[t0,d1,d2]`。主模型在 `d2` 位置预测的正确 token 成为下一轮的 `pending`。代码里 `accepted_count = 1 + matches`；那个 `1` 是主模型自己先前预测的 `t0`，不算“草稿猜对”。

`output_hidden_states=True` 在 prompt prefill 中负责拿各层输出；验证块时，示例为了回退 cache，手动遍历主模型 decoder layer，并在指定层取本轮每个候选位置的 hidden。验证结束只返回：

```python
features[:, :accepted_count]
```

这就是“是不是沿 seq 维度裁掉错误 token 对应的主模型 feature”的答案：**是的**。即使主模型计算过整块候选，只有已接受前缀的 feature 会交给下一轮 `draft()`，随后才进入 DFlash cache。错误草稿的特征不会被缓存。

## 现在说最容易绕的 Cache 回退

一轮开始前，主模型 cache 只保存正式接受的 token。为了验收，主模型会暂时把 `[t0,d1,d2,d3]` 整块算进去。若 `d3` 错了，主模型 cache 就多包含了错误候选，必须回退到 `[t0,d1,d2]`。

Qwen3.5 主模型混用全注意力和线性注意力，两种回退方式不同：

| 主模型层 | 验证时多算的状态 | 拒绝后怎么回退 |
| --- | --- | --- |
| 全注意力 | 按 token 排列的 K/V | 截到验证前长度 `L + accepted_count` |
| 线性注意力 | 卷积窗口与 Gated DeltaNet 递归状态 | 恢复到最后接受 token 的状态快照 |

线性注意力不能只把一个长度减小。它的最终递归状态已经被候选块后面的 token 改写；要恢复到某个前缀，需要该前缀结束时的状态。示例的 `linear_prefix_states()` 根据候选块每个位置的输入，额外保存卷积与递归状态快照。`verify()` 只接受几个，就用对应快照替换当前状态。这里的额外计算是教学实现的成本，不该误认为 DFlash 草稿本身的开销。[Qwen3.5 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

**草稿 cache 为什么没有对应的 `restore()`？** 它只在当前 `draft()` 开头写入上轮已经接受的 `target_features`；本轮 anchor/MASK 的临时 K/V 没有写入。验证失败时，`draft_cache` 仍停在本轮开始时的已接受上下文，下轮再追加实际接受的部分即可。主模型 cache 则真的处理过整块候选，所以必须退。

### `cache.length` 只增加，难道 cache 不会越积越长？

这里的 `cache.length` 是**累计已接受 token 的位置编号**，用于给 RoPE 确定绝对位置。它只增加，因为历史上已经接受过的位置不会因草稿拒绝而“没发生过”。这并不意味着每个草稿层的物理 K/V 张量都一样长。

`cache_context()` 对滑动注意力层做了这两个切片：

```python
key = key[..., -self.sliding_window :, :]
value = value[..., -self.sliding_window :, :]
```

所以滑动层最多保存最近 4096 个上下文位置的 K/V；旧 K/V 被丢掉，`cache.length` 仍继续增加。全注意力层没有这一切片，K/V 随已接受上下文线性增长。比如累计已经接受 5000 个 token，滑动层最多留 4096 个位置，全注意力层仍留 5000 个位置，`cache.length` 是 5000。

验证**之后**找不到 `draft_cache.length -= ...` 是正常的：被拒绝候选从没进入草稿 cache。真正会“缩小”的是主模型 cache 在 `verify()` 里截掉未接受后缀；草稿的滑动层则是在每次追加上下文时按窗口丢弃旧 K/V。两种缩小原因完全不同。

## 拿 MTP 做一次对照

如果刚读完上一篇 MTP，最容易把两套草稿流程混在一起。抓住下面几处区别就够了：

| 问题 | 上篇单层 MTP demo | 本篇 DFlash demo |
| --- | --- | --- |
| 草稿怎么出 | 自回归，多次调用 MTP 逐个提出 token | anchor 加 `MASK` 块，一次草稿前向并行提出多个位置 |
| prompt 历史怎么进草稿模型 | `prefill_mtp` 用移位 token 和主 hidden 跑 MTP 层 | 首轮 `draft` 从主模型特征为各草稿层计算上下文 K/V |
| 草稿 cache 存什么 | 已经作为 MTP 输入处理过的 token 状态 | **已接受**主模型特征投影得到的各层 K/V |
| 草稿猜错后 | 恢复 MTP cache 快照 | 本轮临时块没进 cache，无需恢复 |
| 主模型猜错后 | 截 K/V 或替换线性层状态 | 同样截 K/V 或替换线性层状态 |

这也回答“为什么删除独立的 `prefill_dflash` 还能运行”：**预填充工作仍然在**，只不过合并到首次 `draft()` 中。后续每轮 `draft()` 先把新接受的特征增量写进 cache，再处理新的 anchor/MASK 块。

顺带回答一个常见混淆：有些项目会用 Qwen 底座加 LoRA 和额外任务头，例如 [Kev](https://github.com/jaredpalmer/kev) 的决策模型；本文的 DFlash 权重是专门与 Qwen3.5-4B 配对的**投机草稿模型**，它的推理接口需要主模型中间特征和共享 embedding/head。两者都可以利用 Qwen 权重，但任务与前向流程不同。[Kev 实现说明](https://github.com/jaredpalmer/kev#how-it-works)

## 本地运行：接受数不等于加速比

使用本地 Qwen3.5-4B 和 DFlash 权重，可以用下面的命令在 `cuda:1` 上启动。`HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1` 保证只读本地文件：

```bash
MODEL_PATH=/data04/modelzoo/models/llm/Qwen3.5-4B \
DFLASH_PATH=/data04/modelzoo/models/llm/Qwen3.5-4B-DFlash \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python demo_dflash.py --device cuda:1
```

这条命令使用脚本默认参数：最多生成 256 个 token，`block_size=8`，即每轮最多提出 7 个新草稿 token。脚本会打印生成的文本和 `DFlash accepted drafts`，后者统计被主模型接受的草稿 token 总数。这个数会随生成长度、提示词和草稿命中情况变化，**不能直接当作加速比**。要讨论速度，还需要分别测主模型普通生成与 DFlash 生成的延迟，并记录接受长度分布、显存和上下文长度。尤其这份示例为了展示线性注意力回退，额外计算了逐前缀状态快照；它不等于已经完成优化的推理服务。

## 最后留一张心智图

如果只记住一个边界，我会选这个：**正式确认的历史写入 cache，尚待验证的猜测只在临时计算里**。本文 DFlash cache 遵守这个边界，所以不回退；主模型验证时暂时越过了边界，所以必须把拒绝的后缀退回去。

从这个边界往回看，前面的问题就连起来了：主模型的 33 个 hidden states 给了草稿模型指定层的资料；`fc → hidden_norm` 把资料变成草稿层宽度；上下文只提供 K/V，本轮 anchor/MASK 才产生 Q；`cache.length` 记录累计位置，滑动层的物理 K/V 可以更短；验证只把接受前缀的 feature 交给下一轮。代码中的每一次切片，最终都是在守住“哪些 token 已经被主模型正式接受”。
