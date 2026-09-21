# Qwen3.5 MTP 推理流程图

图中 `t0` 是主模型 prefill 后预测的 token，`d1` 至 `d4` 是 MTP 提出的四个候选，`b` 是主模型在验证后给出的下一 token。`h_x` 表示主模型处理 token `x` 后的 hidden state。

## 总览

![MTP 投机解码总览](assets/mtp/01_overview.svg)

## 1. 主模型 prefill

![主模型 prefill](assets/mtp/02_main_prefill.svg)

## 2. MTP prefill

![MTP prefill](assets/mtp/03_mtp_prefill.svg)

## 3. 连续生成候选

![四次 MTP draft 与 cache 快照](assets/mtp/04_draft.svg)

## 4. 主模型验证

![主模型 chunk 验证与两类 attention cache](assets/mtp/05_verify.svg)

## 情况 A：部分候选被接受

![d3 不匹配时的主模型与 MTP cache 回退](assets/mtp/06_partial_reject.svg)

## 情况 B：全部候选被接受

![全部接受后的双 token MTP 输入](assets/mtp/07_full_accept.svg)

## 情况 C：第一个候选就不匹配

![第一个 draft 不匹配时的 cache 回退](assets/mtp/08_zero_match.svg)

## 两类 Attention Cache 的回退对比

![全注意力 KV 截断与线性注意力状态替换](assets/mtp/09_cache_rollback.svg)

这些图对应 [简洁版代码](demo_qwen3_5.py) 和 [注释版代码](demo_qwen3_5_annotated.py)。
