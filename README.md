# Qwen3.5 MTP 推理示例

本目录有两个功能相同的脚本：

- `demo_qwen3_5.py`：简洁版。
- `demo_qwen3_5_annotated.py`：带注释的学习版，说明 prefill、draft、verify 和 cache 回退。

两个脚本都直接调用模型 `forward()`，不使用 `generate()`。默认每轮最多预测 4 个 draft token。模型从本地目录读取，不下载权重。

## 运行

在当前 Transformers 仓库目录、已安装 PyTorch 和所需依赖的环境中执行：

```bash
export MODEL_PATH=/data/modelscope/Qwen3.5-0.8B
HF_HUB_OFFLINE=1 python demo_qwen3_5.py --prompt "你好" --max-new-tokens 64
```

查看详细注释或指定设备：

```bash
HF_HUB_OFFLINE=1 python demo_qwen3_5_annotated.py --device cuda:0 --draft-tokens 4
```

`--device` 默认为 `auto`，选择一张空闲显存最多的 CUDA 卡；没有可用 CUDA 时使用 CPU。也可以显式传入 `--device cpu`。`MODEL_PATH` 必须指向包含模型配置和权重的本地目录；示例还需要该目录中的 `mtp.*` 权重。

## 流程

1. 主模型对提示词执行一次 prefill，得到首个待提交 token 和主模型 cache。
2. MTP 对移位后的提示词执行一次 prefill，建立 MTP cache。
3. MTP 提出最多 4 个候选 token；主模型把候选作为一个 chunk 验证。
4. 若候选不匹配，逐层回退 cache：全注意力层截断 KV，线性注意力层恢复已保存的卷积和递归状态。

当前示例只处理单条输入，使用贪心解码和一个 MTP 层。BF16 下，chunk 验证与逐 token 前向的舍入结果可能不同。
