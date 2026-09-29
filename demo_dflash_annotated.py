"""Qwen3.5 + Qwen3.5-DFlash 投机解码教学版：与 demo_dflash.py 使用相同的推理逻辑。

运行示例：
    MODEL_PATH=/data04/modelzoo/models/llm/Qwen3.5-4B \
    DFLASH_PATH=/data04/modelzoo/models/llm/Qwen3.5-4B-DFlash \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python demo_dflash_annotated.py --device cuda:1

一、两个模型分工：Qwen3.5 是目标模型，决定最终输出；DFlash 是小得多的
草稿模型，一次并行猜测多个 token。目标模型一次验证整块候选。贪心解码只
接受从块开头连续与目标模型一致的候选，所以结果仍由目标模型决定。

二、prefill：目标模型处理整段提示词，把各层状态写入 main_cache，并给出
第一个待输出 token pending。DFlash 需要目标模型若干指定层的 hidden
states；按配置取出并拼接为 target_features。

三、draft：DFlash.forward 首先把尚未写入草稿 cache 的、已接受的
target_features 经 fc 和 hidden_norm 投影。每个草稿层只为这些上下文
计算 K/V，追加到 draft_cache。之后把 [pending, MASK, MASK, ...]
作为查询块：每层计算该块的 Q/K/V，让 Q 关注已缓存上下文的 K/V 与块内
临时 K/V。块内的 K/V 不写入 draft_cache。最终用主模型共享的 lm_head
将 MASK 位置的 hidden 转为 logits，并一次选出整块候选。

四、verify：主模型逐层处理 [pending, 草稿候选...]。若接受 m 个草稿，
只保留前 1+m 个位置的主模型 cache：全注意力层截断 K/V；线性注意力层
恢复到第 1+m 个位置的卷积与循环状态。主模型在最后一个接受位置预测的
token 是下一轮 pending。只有已接受位置的 target_features 会送入下轮
draft，因此 draft_cache 无需回退。

五、两个“长度”：draft_cache.length 是已接受上下文的累计位置，只增加，
用于 RoPE 位置编号。滑动注意力层实际保存的 K/V 最多为最近 4096 个
token；全注意力层保存所有已接受 token 的 K/V。两者长度可以不同。

六、限制：示例采用 batch=1、单设备、贪心解码。线性注意力层为了支持
部分接受，会额外计算候选块每个位置的状态快照。BF16 的块计算与单 token
计算可能有微小舍入差异。
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, Qwen3Config  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn  # noqa: E402


def rotate_half(x):
    left, right = x.chunk(2, dim=-1)
    return torch.cat([-right, left], dim=-1)


def draft_config_value(config, name, default=None):
    # DFlash 参数可能在 dflash_config 子字典，也可能是 config 的顶层字段。
    # 前者优先；旧 checkpoint 缺字段时用默认值。
    return config.dflash_config.get(name, getattr(config, name, default))


class LayerCache:
    def __init__(self):
        # full_attention 用 keys/values；linear_attention 用 conv/recurrent。
        # 两种层共享这个简单容器，未使用的字段保持 None。
        self.keys = None
        self.values = None
        self.conv_states = {0: None}
        self.recurrent_states = {0: None}
        self.conv_kernel_size = {0: None}
        self.previous = False
        self.record_past = False


class LayerwiseCache:
    """Small Qwen3.5 cache that can restore a partially accepted block."""

    def __init__(self, layer_types):
        self.layer_types = layer_types
        self.layers = [LayerCache() for _ in layer_types]

    def get_seq_length(self, layer_idx=None):
        # 线性注意力没有逐 token 的 K/V 张量，因此用任一全注意力层
        # 的 K/V 长度代表目标模型已经处理的 token 数。
        if layer_idx is None:
            layer_idx = self.layer_types.index("full_attention")
        keys = self.layers[layer_idx].keys
        return 0 if keys is None else keys.shape[-2]

    def get_query_offset(self, layer_idx=0):
        return self.get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length, layer_idx):
        return self.get_seq_length(layer_idx) + query_length, 0

    def update(self, keys, values, layer_idx):
        # 全注意力层每处理一块 token，就把它们的 K/V 接到旧 cache 后面。
        layer = self.layers[layer_idx]
        layer.keys = keys if layer.keys is None else torch.cat([layer.keys, keys], dim=-2)
        layer.values = values if layer.values is None else torch.cat([layer.values, values], dim=-2)
        return layer.keys, layer.values

    def has_previous_state(self, layer_idx, state_idx=0):
        return self.layers[layer_idx].previous

    def update_conv_state(self, new_states, layer_idx, conv_kernel_size, state_idx=0):
        # 线性注意力的因果卷积只需保留最近一个卷积核宽度的输入。
        layer = self.layers[layer_idx]
        layer.conv_kernel_size[state_idx] = conv_kernel_size
        if layer.previous:
            full_states = torch.cat([layer.conv_states[state_idx], new_states], dim=-1)
        else:
            full_states = new_states
            layer.previous = True
        layer.conv_states[state_idx] = F.pad(
            full_states[..., -conv_kernel_size:], (max(0, conv_kernel_size - full_states.shape[-1]), 0)
        )
        return full_states

    def update_recurrent_state(self, new_state, layer_idx, state_idx=0):
        # 线性注意力的递归状态没有“按 token 存一串 K/V”这种结构。
        self.layers[layer_idx].recurrent_states[state_idx] = new_state.clone()
        return new_state

    def restore(self, states):
        """Truncate K/V and restore the recurrent state at the accepted prefix."""
        # verify 已经计算过整块候选，但只接受前缀。这里删除后缀影响。
        for layer_type, layer, (marker, conv, recurrent) in zip(self.layer_types, self.layers, states):
            if layer_type == "full_attention":
                # marker 是前缀结束后的 K/V 长度；切掉拒绝的候选。
                layer.keys = layer.keys[..., :marker, :]
                layer.values = layer.values[..., :marker, :]
            else:
                # 递归状态不能靠截张量回退，必须替换为对应前缀的快照。
                layer.previous = marker
                layer.conv_states[0] = conv
                layer.recurrent_states[0] = recurrent


class DFlashContextCache:
    """Draft K/V for accepted target tokens; the temporary mask block is never stored."""

    def __init__(self, num_layers):
        # length 是“累计已接受数”，不等于所有层实际 K/V 长度：
        # 滑动窗口层会扔掉旧 K/V，但 RoPE 位置仍需要累计值。
        self.length = 0
        self.keys = [None] * num_layers
        self.values = [None] * num_layers


class DFlashAttention(nn.Module):
    """Attend from the draft block to target context and the whole draft block."""

    def __init__(self, config, layer_idx):
        super().__init__()
        # 权重名称与 safetensors 中的 q_proj/k_proj/... 一一对应。
        # 注意：每个草稿层各有自己的投影矩阵，因此各有一份上下文 K/V。
        width = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.layer_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window
        self.q_proj = nn.Linear(width, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(width, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(width, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, width, bias=config.attention_bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def cache_context(self, context, cos, sin, cache, layer_idx):
        """Project each newly accepted target feature just once."""
        batch, length, _ = context.shape
        # 已接受的上下文不需要生成新预测，只作为本轮查询的 K/V。
        # K 在进入 cache 前先做 head 归一化和 RoPE；V 不需要 RoPE。
        key = self.k_norm(self.k_proj(context).view(batch, length, self.num_kv_heads, self.head_dim))
        key = key.transpose(1, 2)
        key = key * cos[:, None] + rotate_half(key) * sin[:, None]
        value = self.v_proj(context).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if cache.keys[layer_idx] is not None:
            # target_features 在首轮是完整 prompt，此后只是本轮新接受
            # 的少量 token，所以旧上下文不用重复计算投影。
            key = torch.cat([cache.keys[layer_idx], key], dim=-2)
            value = torch.cat([cache.values[layer_idx], value], dim=-2)
        if self.layer_type == "sliding_attention":
            # 物理 K/V 最多保留最近 sliding_window 个上下文 token。
            # cache.length 不在这里减小，它记录的是绝对位置。
            key = key[..., -self.sliding_window :, :]
            value = value[..., -self.sliding_window :, :]
        cache.keys[layer_idx] = key
        cache.values[layer_idx] = value

    def forward(self, hidden, cos, sin, cache, layer_idx):
        batch, block_size, _ = hidden.shape
        # hidden 是 [pending, MASK, ...] 的当前草稿块，不是整段上下文。
        # 查询 Q 只为这几个位置计算；块内 K/V 是本轮临时数据。
        q = self.q_norm(self.q_proj(hidden).view(batch, block_size, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(hidden).view(batch, block_size, self.num_kv_heads, self.head_dim))
        v = self.v_proj(hidden).view(batch, block_size, self.num_kv_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        cached_length = cache.keys[layer_idx].shape[-2]
        # 注意这里的 cat 只生成局部变量 k/v；绝不赋回 cache。
        # 因此验证拒绝草稿后，draft_cache 没有候选状态需要删除。
        k = torch.cat([cache.keys[layer_idx], k], dim=-2)
        v = torch.cat([cache.values[layer_idx], v], dim=-2)
        repeats = self.num_heads // self.num_kv_heads
        # GQA：查询头比 K/V 头多；把每个 K/V 头共享给一组查询头。
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        mask = None
        if self.layer_type == "sliding_attention":
            # 该层既因果（不能看到右边未来位置），又只能看到窗口内位置。
            # cache 可能已截掉旧 K/V，所以用累计位置重建绝对 key 位置。
            query_positions = cache.length + torch.arange(block_size, device=hidden.device)
            key_positions = torch.arange(
                cache.length - cached_length, cache.length + block_size, device=hidden.device
            )
            distance = query_positions[:, None] - key_positions[None, :]
            mask = ((distance >= 0) & (distance < self.sliding_window))[None, None]
        # full_attention 层的 mask=None：草稿块可双向关注，用并行方式
        # 预测所有 MASK；目标模型随后会逐位置做因果验证。
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0)
        return self.o_proj(output.transpose(1, 2).reshape(batch, block_size, -1))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden, cos, sin, cache, layer_idx):
        # 与常规 Transformer 一样：注意力残差、MLP 残差。
        # 上下文在此层只供注意力取 K/V，hidden 才沿草稿层向前传递。
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), cos, sin, cache, layer_idx)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class DFlashDraft(nn.Module):
    """The checkpoint's six-layer drafter; it shares embeddings and LM head with Qwen3.5."""

    def __init__(self, config):
        super().__init__()
        # target_layer_ids 指出从主模型哪几层取 hidden；本地 4B 权重
        # 取 8 层。fc 把这些层拼接后的宽向量压回草稿 hidden_size。
        settings = config.dflash_config
        self.config = config
        self.target_layer_ids = settings["target_layer_ids"]
        self.block_size = settings["block_size"]
        self.mask_token_id = settings["mask_token_id"]
        # 有些 DFlash checkpoint 对 embedding/logits 另有缩放约定。
        # 当前 4B 权重没写这些字段，默认缩放为 1 且不做 softcap。
        self.input_embedding_scale = float(draft_config_value(config, "input_embedding_scale", 1.0))
        self.output_multiplier = float(draft_config_value(config, "output_multiplier", 1.0))
        self.final_logit_softcapping = draft_config_value(config, "final_logit_softcapping")
        self.layers = nn.ModuleList([DFlashDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def forward(self, cache, target_features, noise_embedding):
        # Only accepted target tokens enter the cache. The anchor and mask block
        # are temporary and are recomputed on every draft call.
        # 首轮 target_features 是 prompt 的特征；后续只是上轮新接受 token
        # 的特征。投影只在它们第一次进入草稿 cache 时执行。
        context = self.hidden_norm(self.fc(target_features))
        context_positions = torch.arange(cache.length, cache.length + context.shape[1], device=context.device)[None]
        context_cos, context_sin = self.rotary_emb(context, context_positions)
        for index, layer in enumerate(self.layers):
            layer.self_attn.cache_context(context, context_cos, context_sin, cache, index)
        # 先更新累计位置，再为 anchor/MASK 块编号；块从上下文后面开始。
        cache.length += context.shape[1]

        hidden = noise_embedding
        positions = torch.arange(cache.length, cache.length + hidden.shape[1], device=hidden.device)[None]
        cos, sin = self.rotary_emb(hidden, positions)
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, cos, sin, cache, index)
        # 这里返回的是草稿块的 hidden，不是 token ID。
        return self.norm(hidden)

    def compute_logits(self, hidden, output_head):
        # 草稿模型共享目标模型的 lm_head，因此自身权重文件无需另存一份。
        logits = output_head(hidden) * self.output_multiplier
        softcap = self.final_logit_softcapping
        if softcap is not None and float(softcap) > 0:
            # tanh softcap 限制 logits 幅度；只在配置明确要求时执行。
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        return logits


def load_draft(path, target):
    # 只读取本地配置与权重。配置声明 32 层目标模型，避免错配其他尺寸。
    config = Qwen3Config.from_pretrained(path, local_files_only=True)
    if config.num_target_layers != target.config.text_config.num_hidden_layers:
        raise ValueError("DFlash checkpoint does not match the target model's layer count")
    draft = DFlashDraft(config)
    weights_file = path / "model.safetensors"
    parameters = draft.state_dict()
    with safe_open(weights_file, framework="pt", device="cpu") as checkpoint:
        # 教学示例要求参数名称和形状完全匹配，避免悄悄漏载权重。
        missing = set(parameters) - set(checkpoint.keys())
        unexpected = set(checkpoint.keys()) - set(parameters)
        if missing or unexpected:
            raise RuntimeError(f"DFlash weights do not match: missing={missing}, unexpected={unexpected}")
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = checkpoint.get_tensor(name)
                if parameter.shape != value.shape:
                    raise RuntimeError(f"Wrong shape for {name}: {value.shape} != {parameter.shape}")
                parameter.copy_(value)
    return draft.to(device=target.device, dtype=target.dtype).eval()


def linear_prefix_states(attn, hidden, layer_cache):
    """Compute the conv and recurrent state after every candidate token."""
    # 主模型线性注意力层一次处理整块候选后，只在 cache 留下块末状态。
    # 若只接受块的前 m 个 token，就需要知道第 m 个位置的中间状态。
    # 此函数根据该层的输入逐 token 复原这些状态快照，不再跑一遍整层。
    batch_size, seq_len, _ = hidden.shape
    raw_qkv = attn.in_proj_qkv(hidden).transpose(1, 2)
    previous_conv = layer_cache.conv_states[0]
    # 因果卷积需要紧邻候选块之前的若干输入；从旧卷积缓存接上本块 QKV。
    full_qkv = raw_qkv if previous_conv is None else torch.cat([previous_conv, raw_qkv], dim=-1)
    convolved = causal_conv1d_fn(
        full_qkv, attn.conv1d.weight.squeeze(1), attn.conv1d.bias, attn.activation
    )[..., -seq_len:].transpose(1, 2)
    _, key, value = torch.split(convolved, [attn.key_dim, attn.key_dim, attn.value_dim], dim=-1)
    key = key.reshape(batch_size, seq_len, attn.num_k_heads, attn.head_k_dim)
    value = value.reshape(batch_size, seq_len, attn.num_v_heads, attn.head_v_dim).float()
    if attn.num_v_heads != attn.num_k_heads:
        # 与实际线性注意力计算保持一致：K 头复用到各个 V 头。
        key = key.repeat_interleave(attn.num_v_heads // attn.num_k_heads, dim=2)
    key = key.float()
    key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)

    beta = attn.in_proj_b(hidden).sigmoid().float()
    decay = (-attn.A_log.float().exp() * F.softplus(attn.in_proj_a(hidden).float() + attn.dt_bias)).float()
    previous_recurrent = layer_cache.recurrent_states[0]
    # 递归从旧缓存状态开始；prompt prefill 后它通常已经存在。
    state = (
        previous_recurrent.float().clone()
        if previous_recurrent is not None
        else value.new_zeros(batch_size, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim)
    )
    states = []
    kernel_size = attn.conv_kernel_size
    for index in range(seq_len):
        # 对候选块每个位置执行与 Gated DeltaNet 相同的状态更新，
        # 顺手保存这一位置后的卷积尾巴和递归状态。
        k, v = key[:, index], value[:, index]
        state = state * decay[:, index].exp()[..., None, None]
        predicted = (state * k.unsqueeze(-1)).sum(dim=-2)
        correction = (v - predicted) * beta[:, index].unsqueeze(-1)
        state = state + k.unsqueeze(-1) * correction.unsqueeze(-2)
        conv = full_qkv[..., : full_qkv.shape[-1] - seq_len + index + 1][..., -kernel_size:].clone()
        states.append((True, conv, state.clone()))
    return states


def validation_forward(target, candidate_ids, main_cache, target_layer_ids):
    """Verify one block and save enough state to roll back its rejected tail."""
    # main_cache 当前恰好覆盖 prompt + 已接受的生成 token；pending
    # 还未进入它。candidate_ids 是 [pending, d1, d2, ...]。
    decoder = target.model.language_model
    start = main_cache.get_seq_length()
    positions = torch.arange(start, start + candidate_ids.shape[1], device=candidate_ids.device)[None]
    rope_positions = positions[None, ...].expand(3, -1, -1)
    # 验证时直接逐层调用主模型，便于在每层前保存回退所需状态。
    hidden = decoder.embed_tokens(candidate_ids)
    rope = decoder.rotary_emb(hidden, rope_positions)
    mask = create_causal_mask(
        config=decoder.config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=main_cache,
        position_ids=positions,
        layer_idx=main_cache.layer_types.index("full_attention"),
    )

    rollback_data = []
    selected_hidden = {}
    for index, layer in enumerate(decoder.layers[: decoder.config.num_hidden_layers]):
        layer_cache = main_cache.layers[index]
        if main_cache.layer_types[index] == "full_attention":
            # 全注意力层只记下旧 K/V 长度。layer 调用后会把整块
            # candidate 的 K/V 写入 cache，回退时按长度切片即可。
            old_length = main_cache.get_seq_length(index)
            hidden = layer(
                hidden,
                position_embeddings=rope,
                position_ids=positions,
                attention_mask=mask,
                past_key_values=main_cache,
            )
            rollback_data.append(old_length)
        else:
            # 线性层写入的是最终递归状态；先备份每个 token 的前缀
            # 状态，稍后若有拒绝，就恢复到最后接受的位置。
            prefix_states = linear_prefix_states(layer.linear_attn, layer.input_layernorm(hidden), layer_cache)
            hidden = layer(
                hidden,
                position_embeddings=rope,
                position_ids=positions,
                attention_mask=None,
                past_key_values=main_cache,
            )
            rollback_data.append(prefix_states)
        if index in target_layer_ids:
            # DFlash 只需要配置指定的主模型层输出。这里保存候选块
            # 每个位置的特征；verify 之后还会裁掉拒绝的位置。
            selected_hidden[index] = hidden

    logits = target.lm_head(decoder.norm(hidden))
    features = torch.cat([selected_hidden[i] for i in target_layer_ids], dim=-1)
    return logits, features, rollback_data


def prefill(target, prompt_ids, target_layer_ids):
    """Cache the prompt in Qwen3.5 and choose the first pending token."""
    # prefill 只处理 prompt；pending 是 prompt 最后位置的 logits 选出
    # 的下一个 token，尚未写进 main_cache。
    cache = LayerwiseCache(target.config.text_config.layer_types)
    outputs = target(
        input_ids=prompt_ids,
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
    )
    pending = outputs.logits[:, -1:, :].argmax(dim=-1)
    # hidden_states[0] 是进入第 0 层前的 embedding；第 i 层输出
    # 位于 hidden_states[i + 1]。只拼接 DFlash 配置指定的层。
    features = torch.cat([outputs.hidden_states[i + 1] for i in target_layer_ids], dim=-1)
    return pending, features, cache


def draft(draft_model, target, cache, target_features, pending, block_size):
    """Predict the masked positions in one DFlash forward pass."""
    # The anchor is a token already chosen by the target. The remaining slots
    # all contain the mask token and are predicted together in one draft pass.
    # 例如 block_size=4，输入是 [pending, MASK, MASK, MASK]，
    # 草稿将并行猜测 pending 后面的三个 token。
    noise_ids = torch.cat([pending, pending.new_full((1, block_size - 1), draft_model.mask_token_id)], dim=1)
    # Qwen3.5 normalizes its usual input embeddings. DFlash expects the raw table.
    # 直接查共享 embedding 表，避免 Qwen3.5 常规 embedding 路径的额外
    # 归一化；配置若有 input_embedding_scale，则再乘相同倍率。
    noise_embedding = F.embedding(noise_ids, target.get_input_embeddings().weight)
    noise_embedding = noise_embedding * draft_model.input_embedding_scale
    # 这次调用会先把新接受的 target_features 写入草稿 K/V cache，
    # 再让 anchor/MASK 块作为查询读取 cache，但块本身不写入 cache。
    hidden = draft_model(cache, target_features, noise_embedding)
    # hidden[:, 0] 是 anchor 位置；它不作为草稿候选输出。
    logits = draft_model.compute_logits(hidden[:, 1:, :], target.lm_head)
    return logits.argmax(dim=-1)


def verify(target, pending, proposed, main_cache, target_layer_ids, eos_ids):
    """Accept the greedy matching prefix and roll back rejected cache states."""
    # 目标模型一次处理整块。按自回归语义，pending 位置的 logits
    # 预测 d1，d1 位置的 logits 预测 d2，以此类推。
    candidate_ids = torch.cat([pending, proposed], dim=1)
    logits, features, rollback_data = validation_forward(target, candidate_ids, main_cache, target_layer_ids)
    greedy = logits[:, :-1, :].argmax(dim=-1)
    # cumprod 让首次不匹配之后的全部位置变为 0。即便后面偶然又
    # 匹配，也不能跳过前面的错误 token 去接受它们。
    matches = (greedy == proposed).to(torch.int32).cumprod(dim=1).sum().item()
    accepted_count = matches + 1  # The target's pending token is always accepted.
    if matches < proposed.shape[1]:
        # 目标模型刚才把整块写入 main_cache。只接受前缀时，必须
        # 去掉后缀的状态，否则下一轮会把错误草稿当作已生成上下文。
        restore_states = []
        for layer_type, state in zip(main_cache.layer_types, rollback_data):
            if layer_type == "full_attention":
                # state 是验证前长度，追加 accepted_count 就是新长度。
                restore_states.append((state + accepted_count, None, None))
            else:
                # state 中第 accepted_count-1 项是最后接受位置的状态。
                restore_states.append(state[accepted_count - 1])
        main_cache.restore(restore_states)

    accepted_ids = candidate_ids[0, :accepted_count].tolist()
    eos_index = next((i for i, token_id in enumerate(accepted_ids) if token_id in eos_ids), None)
    if eos_index is not None:
        # EOS 不显示到最终文本；遇到它，本轮生成立即结束。
        accepted_ids = accepted_ids[:eos_index]
    # 最后一个接受位置的 logits 提供下一轮 pending。若第一个草稿
    # 就被拒绝，这就是目标模型给出的替换 token。
    next_pending = logits[:, matches : matches + 1, :].argmax(dim=-1)
    # 只返回接受前缀的 hidden。下轮 DFlash.forward 才会把它们写入
    # draft_cache，所以草稿 cache 不包含被拒绝候选，也无需回退。
    return accepted_ids, next_pending, features[:, :accepted_count], eos_index is not None


def generate(target, draft_model, prompt_ids, max_new_tokens, block_size, eos_ids):
    """把 prefill、draft、verify 串成完整的贪心投机生成。"""
    generated = []
    accepted_drafts = 0
    pending, target_features, main_cache = prefill(target, prompt_ids, draft_model.target_layer_ids)
    draft_cache = DFlashContextCache(len(draft_model.layers))

    while len(generated) < max_new_tokens and pending.item() not in eos_ids:
        # 最后剩 1 个 token 的预算时，没有空间放草稿候选；直接
        # 输出已经由目标模型确定的 pending。
        verify_size = min(block_size, max_new_tokens - len(generated))
        if verify_size == 1:
            generated.append(pending.item())
            break
        proposed = draft(draft_model, target, draft_cache, target_features, pending, verify_size)
        accepted_ids, pending, accepted_features, stopped = verify(
            target, pending, proposed, main_cache, draft_model.target_layer_ids, eos_ids
        )
        generated.extend(accepted_ids)
        # accepted_ids 中第一项是必定接受的 pending，其余才计入
        # “接受的草稿 token”统计。
        accepted_drafts += max(0, len(accepted_ids) - 1)
        if stopped:
            break

        # 本轮主模型可能算过被拒绝候选，但这些特征不进入草稿 cache。
        # 下轮 draft 只接收这里裁好的 accepted_features。
        target_features = accepted_features
    return generated, accepted_drafts


def main():
    # 与原 demo 相同：权重由本地目录提供，不在脚本中下载。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--device", default="auto", help="Single device, for example cuda:0 or cpu")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")

    model_path = Path(os.environ["MODEL_PATH"]).expanduser()
    draft_path = Path(os.environ["DFLASH_PATH"]).expanduser()
    if args.device == "auto":
        # 多卡时选空闲显存最多的一张；主模型和草稿模型必须同设备。
        device = (
            f"cuda:{max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])}"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = str(torch.device(args.device))
    print(f"Running on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # 目标模型提供最终答案，也提供草稿输入 embedding、输出 lm_head
    # 以及指定中间层的 hidden states。
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map={"": device}, local_files_only=True
    ).eval()
    draft = load_draft(draft_path, model)
    # checkpoint 训练时的最大块大小为 16；命令行可以选更小的块。
    if not 2 <= args.block_size <= draft.block_size:
        parser.error(f"--block-size must be between 2 and {draft.block_size}")
    inputs = tokenizer.apply_chat_template(
        # 与真实聊天推理一致，在 user 消息后添加 assistant 起始标记。
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    # generation_config 可能给单个 EOS，也可能给多个 EOS。
    eos_ids = set(eos) if isinstance(eos, list) else {eos}
    with torch.inference_mode():
        # 推理不保存反向传播计算图，减少内存开销。
        generated, accepted_drafts = generate(
            model, draft, inputs["input_ids"], args.max_new_tokens, args.block_size, eos_ids
        )
    print(tokenizer.decode(generated, skip_special_tokens=True))
    print(f"DFlash accepted drafts: {accepted_drafts}")


if __name__ == "__main__":
    main()
