"""Qwen3.8-27B + DFlash2 的单条输入投机解码教学示例。

不导入 dflash，也不调用 Transformers 的 generate()。流程明确分成
prefill（目标模型）、draft（增量缓存已接受特征，再选 top-k 候选路径）、
verify（目标模型整块验证并回退错误后缀）。目标模型与草稿都在同一张 GPU。

本脚本只读取本地权重，不会联网下载。

    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python demo_dflash2.py --trace
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

from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration, Qwen3Config  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn  # noqa: E402


def rotate_half(x):
    left, right = x.chunk(2, dim=-1)
    return torch.cat([-right, left], dim=-1)


def draft_config_value(config, name, default=None):
    return config.dflash_config.get(name, getattr(config, name, default))


def sampling_probs(logits, temperature, top_p=1.0, top_k=0):
    """按目标模型的温度、top-p、top-k 得到概率；也用于草稿选择器的温度。"""
    scores = logits.float() / temperature
    vocab_size = scores.shape[-1]
    if 0 < top_k < vocab_size:
        scores, indices = torch.topk(scores, top_k, dim=-1)
    else:
        indices = None
    probs = torch.softmax(scores, dim=-1)
    if top_p < 1:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    if indices is not None:
        probs = torch.zeros_like(logits, dtype=probs.dtype).scatter(-1, indices, probs)
    return probs


def sample_probs(probs):
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(probs.shape[:-1])


def select_target_token(logits, temperature, top_p, top_k):
    if temperature == 0:
        return logits.argmax(dim=-1)
    return sample_probs(sampling_probs(logits, temperature, top_p, top_k))


def rejection_sample(draft_tokens, target_probs, draft_probs, draft_indices):
    """逐位置以 min(1, p(token)/q(token)) 接受；首个拒绝处抽残差分布。"""
    gamma = draft_tokens.shape[1]
    p = target_probs[:, :gamma].gather(-1, draft_tokens[..., None])[..., 0]
    # q 只定义在每个位置的 selector_top_k 个 token 上；映射到词表 ID。
    q = (draft_probs * (draft_indices == draft_tokens[..., None])).sum(-1)
    accepted = (torch.rand_like(q) * q < p).to(torch.int32).cumprod(-1).sum(-1)[0].item()
    if accepted == gamma:
        return accepted, sample_probs(target_probs[:, -1])[0]

    # 拒绝时不能直接从 p 采样；要从 normalize(max(p-q, 0)) 采样。
    residual = target_probs[0, accepted].clone()
    residual.scatter_add_(0, draft_indices[0, accepted], -draft_probs[0, accepted])
    residual.clamp_min_(0)
    total = residual.sum()
    residual = torch.where(
        total > 0,
        residual / total.clamp_min(torch.finfo(residual.dtype).tiny),
        target_probs[0, accepted],
    )
    return accepted, sample_probs(residual[None])[0]


def grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    """两 tap 分组动态卷积：每个位置读自己和左边一个位置。"""
    batch, length, width = hidden.shape
    groups = width // group_size
    blocks = hidden.reshape(batch, length, groups, group_size)
    dynamic = dynamic.reshape(batch, length, base.shape[0], groups, 1)
    result = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        kernel = base[offset].reshape(1, 1, groups, group_size).to(hidden.dtype)
        result = result + kernel * values
        result = torch.addcmul(result, dynamic[:, :, offset], values)
    return result.reshape_as(hidden)


class GroupedDynamicCausalConv(nn.Module):
    """DFlash2 的块内卷积没有跨轮状态，也不写入草稿 cache。"""

    def __init__(self, width, kernel_size, group_size):
        super().__init__()
        groups = width // group_size
        self.group_size = group_size
        self.kernel_size = kernel_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, width))
        nn.init.zeros_(self.base_kernel)  # 真实运行会由 checkpoint 覆盖。
        self.kernel_projection = nn.Linear(width, 2 * kernel_size * groups, bias=False)

    def prepare(self, hidden):
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).reshape(*hidden.shape[:-1], 2, self.kernel_size, groups)
        return (
            grouped_dynamic_convolve(hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden, dynamic):
        return grouped_dynamic_convolve(hidden, dynamic, self.base_kernel[1], self.group_size)


class CandidateSelector(nn.Module):
    """从每个位置的 top-k 中，按前一个已选 ID 逐位置选出路径。"""

    def __init__(self, config):
        super().__init__()
        rank = int(draft_config_value(config, "selector_rank"))
        self.top_k = int(draft_config_value(config, "selector_top_k"))
        self.predecessor_codebook = nn.Embedding(config.vocab_size, rank)
        self.successor_codebook = nn.Embedding(config.vocab_size, rank)
        self.hidden_projection = nn.Linear(config.hidden_size, rank, bias=False)

    def forward(self, hidden, logits, anchor_ids, temperature):
        # [1, 7, 248320] -> 两个 [1, 7, 16]；unary 是分数，candidates 是 token ID。
        unary, candidates = torch.topk(logits, self.top_k, dim=-1, sorted=False)
        projected = self.hidden_projection(hidden)  # [1, 7, 256]
        predecessor = anchor_ids  # 第一个位置用目标模型选出的 anchor。
        chosen = []
        q_rows = []
        for position in range(hidden.shape[1]):
            # 原始 logit + 当前候选与前一个已选 ID 的相邻匹配分。
            scores = unary[:, position] + torch.einsum(
                "br,bkr->bk",
                self.predecessor_codebook(predecessor) * projected[:, position],
                self.successor_codebook(candidates[:, position]),
            )
            if temperature > 0:
                # q 的词表只有这 16 个候选；生成接口的 top-p/top-k 不作用于选择器。
                q = sampling_probs(scores[:, None], temperature)[:, 0]
                index = sample_probs(q)
                q_rows.append(q)
            else:
                index = scores.argmax(dim=-1)
            # index 是 top-k 内的下标，不是词表 token ID。
            predecessor = candidates[:, position].gather(-1, index[:, None])[:, 0]
            chosen.append(predecessor)
        return (
            torch.stack(chosen, dim=1),  # [1, 7]：真正的词表 token ID。
            candidates,  # [1, 7, 16]：供拒绝采样时把 q 映射到词表。
            torch.stack(q_rows, dim=1) if q_rows else None,
        )


class LayerCache:
    def __init__(self):
        self.keys = None
        self.values = None
        self.conv_states = {0: None}
        self.recurrent_states = {0: None}
        self.conv_kernel_size = {0: None}
        self.previous = False
        self.record_past = False


class LayerwiseCache:
    """目标模型 cache：全注意力存 K/V，线性注意力存卷积与递归状态。

    verify 会暂时把整个候选块写入它。遇到首个错误后，全注意力 K/V 截断，
    线性注意力恢复到已接受前缀末尾的状态快照。
    """

    def __init__(self, layer_types):
        self.layer_types = layer_types
        self.layers = [LayerCache() for _ in layer_types]

    def get_seq_length(self, layer_idx=None):
        if layer_idx is None:
            layer_idx = self.layer_types.index("full_attention")
        keys = self.layers[layer_idx].keys
        return 0 if keys is None else keys.shape[-2]

    def get_query_offset(self, layer_idx=0):
        return self.get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length, layer_idx):
        return self.get_seq_length(layer_idx) + query_length, 0

    def update(self, keys, values, layer_idx):
        layer = self.layers[layer_idx]
        layer.keys = keys if layer.keys is None else torch.cat([layer.keys, keys], dim=-2)
        layer.values = values if layer.values is None else torch.cat([layer.values, values], dim=-2)
        return layer.keys, layer.values

    def has_previous_state(self, layer_idx, state_idx=0):
        return self.layers[layer_idx].previous

    def update_conv_state(self, new_states, layer_idx, conv_kernel_size, state_idx=0):
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
        self.layers[layer_idx].recurrent_states[state_idx] = new_state.clone()
        return new_state

    def restore(self, states):
        """Truncate K/V and restore the recurrent state at the accepted prefix."""
        for layer_type, layer, (marker, conv, recurrent) in zip(self.layer_types, self.layers, states):
            if layer_type == "full_attention":
                layer.keys = layer.keys[..., :marker, :]
                layer.values = layer.values[..., :marker, :]
            else:
                layer.previous = marker
                layer.conv_states[0] = conv
                layer.recurrent_states[0] = recurrent


class DFlashContextCache:
    """草稿各层已接受上下文的 K/V。

    length 是自 prompt 开始的绝对 token 位置；滑动层实际 K/V 仅保留最近
    sliding_window 个位置。因此 length 可持续增长，物理 K/V 长度却有上限。
    当前 anchor/MASK 块的 K/V 只在 self_attn.forward 的局部变量中出现。
    """

    def __init__(self, num_layers):
        self.length = 0
        self.keys = [None] * num_layers
        self.values = [None] * num_layers


class DFlashAttention(nn.Module):
    """Attend from the draft block to target context and the whole draft block."""

    def __init__(self, config, layer_idx):
        super().__init__()
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
        """只为新接受的主模型特征计算 K/V；历史上下文不重复投影。"""
        batch, length, _ = context.shape
        key = self.k_norm(self.k_proj(context).view(batch, length, self.num_kv_heads, self.head_dim))
        key = key.transpose(1, 2)
        key = key * cos[:, None] + rotate_half(key) * sin[:, None]
        value = self.v_proj(context).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if cache.keys[layer_idx] is not None:
            key = torch.cat([cache.keys[layer_idx], key], dim=-2)
            value = torch.cat([cache.values[layer_idx], value], dim=-2)
        if self.layer_type == "sliding_attention":
            key = key[..., -self.sliding_window :, :]
            value = value[..., -self.sliding_window :, :]
        cache.keys[layer_idx] = key
        cache.values[layer_idx] = value

    def forward(self, hidden, cos, sin, cache, layer_idx):
        batch, block_size, _ = hidden.shape
        q = self.q_norm(self.q_proj(hidden).view(batch, block_size, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(hidden).view(batch, block_size, self.num_kv_heads, self.head_dim))
        v = self.v_proj(hidden).view(batch, block_size, self.num_kv_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        cached_length = cache.keys[layer_idx].shape[-2]
        k = torch.cat([cache.keys[layer_idx], k], dim=-2)
        v = torch.cat([cache.values[layer_idx], v], dim=-2)
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        mask = None
        if self.layer_type == "sliding_attention":
            query_positions = cache.length + torch.arange(block_size, device=hidden.device)
            key_positions = torch.arange(
                cache.length - cached_length, cache.length + block_size, device=hidden.device
            )
            distance = query_positions[:, None] - key_positions[None, :]
            # 此 checkpoint 的 is_causal=False：块内可双向看，但窗口仍限制为 2048。
            mask = (distance.abs() < self.sliding_window)[None, None]
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0)
        return self.o_proj(output.transpose(1, 2).reshape(batch, block_size, -1))


class DFlash2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        kernel_size = int(draft_config_value(config, "conv_kernel_size"))
        group_size = int(draft_config_value(config, "conv_group_size"))
        self.attention_conv = GroupedDynamicCausalConv(config.hidden_size, kernel_size, group_size)
        self.mlp_conv = GroupedDynamicCausalConv(config.hidden_size, kernel_size, group_size)

    def forward(self, hidden, cos, sin, cache, layer_idx):
        # 两处动态卷积都只读本轮块；first mask 的前一个位置是 anchor。
        attention_input, attention_kernel = self.attention_conv.prepare(self.input_layernorm(hidden))
        attention_output = self.self_attn(attention_input, cos, sin, cache, layer_idx)
        hidden = hidden + self.attention_conv.finish(attention_output, attention_kernel)
        mlp_input, mlp_kernel = self.mlp_conv.prepare(self.post_attention_layernorm(hidden))
        return hidden + self.mlp_conv.finish(self.mlp(mlp_input), mlp_kernel)


class DFlash2Draft(nn.Module):
    """本地 checkpoint 的 5 层草稿网络；复用目标模型 embedding 和 LM head。"""

    def __init__(self, config):
        super().__init__()
        settings = config.dflash_config
        self.config = config
        self.target_layer_ids = settings["target_layer_ids"]
        self.block_size = settings["block_size"]
        self.mask_token_id = settings["mask_token_id"]
        self.input_embedding_scale = float(draft_config_value(config, "input_embedding_scale", 1.0))
        self.output_multiplier = float(draft_config_value(config, "output_multiplier", 1.0))
        self.final_logit_softcapping = draft_config_value(config, "final_logit_softcapping")
        self.layers = nn.ModuleList([DFlash2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.candidate_selector = CandidateSelector(config)

    def forward(self, cache, target_features, noise_embedding):
        # 首轮 target_features 是完整 prompt；后续仅是上一轮 verify 接受的前缀。
        # 只为这些 token 计算每层 K/V，因此无需对草稿 cache 做验证后回退。
        context = self.hidden_norm(self.fc(target_features))
        context_positions = torch.arange(cache.length, cache.length + context.shape[1], device=context.device)[None]
        context_cos, context_sin = self.rotary_emb(context, context_positions)
        for index, layer in enumerate(self.layers):
            layer.self_attn.cache_context(context, context_cos, context_sin, cache, index)
        cache.length += context.shape[1]  # 只在“已接受上下文”写入后增加。

        hidden = noise_embedding  # [anchor, MASK × 7]；其 K/V 不保存到 cache。
        positions = torch.arange(cache.length, cache.length + hidden.shape[1], device=hidden.device)[None]
        cos, sin = self.rotary_emb(hidden, positions)
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, cos, sin, cache, index)
        return self.norm(hidden)

    def compute_logits(self, hidden, output_head):
        # lm_head 可能被 Accelerate 分片/卸载，此时 weight.device 会是 meta。
        # 直接调用模块，让它的设备钩子搬运输入；只把结果搬回草稿 GPU。
        logits = output_head(hidden).to(hidden.device) * self.output_multiplier
        softcap = self.final_logit_softcapping
        if softcap is not None and float(softcap) > 0:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        return logits


def load_draft(path, target, device):
    config = Qwen3Config.from_pretrained(path, local_files_only=True)
    if config.num_target_layers != target.config.text_config.num_hidden_layers:
        raise ValueError("DFlash2 checkpoint does not match the target model's layer count")
    if config.architectures != ["DFlash2DraftModel"]:
        raise ValueError(f"Expected a DFlash2 checkpoint, got {config.architectures}")
    draft = DFlash2Draft(config)
    weights_file = path / "model.safetensors"
    parameters = draft.state_dict()
    with safe_open(weights_file, framework="pt", device="cpu") as checkpoint:
        # DFlash2 的两张 codebook 在 safetensors 中省略了 ".weight" 后缀。
        def checkpoint_name(name):
            if name in (
                "candidate_selector.predecessor_codebook.weight",
                "candidate_selector.successor_codebook.weight",
            ):
                return name.removesuffix(".weight")
            return name

        names = {checkpoint_name(name) for name in parameters}
        missing = names - set(checkpoint.keys())
        unexpected = set(checkpoint.keys()) - names
        if missing or unexpected:
            raise RuntimeError(f"DFlash2 weights do not match: missing={missing}, unexpected={unexpected}")
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = checkpoint.get_tensor(checkpoint_name(name))
                if parameter.shape != value.shape:
                    raise RuntimeError(f"Wrong shape for {name}: {value.shape} != {parameter.shape}")
                parameter.copy_(value)
    return draft.to(device=device, dtype=target.dtype).eval()


def linear_prefix_states(attn, hidden, layer_cache):
    """保存验证块每个前缀结束时的卷积与 DeltaNet 状态，供拒绝后回退。"""
    batch_size, seq_len, _ = hidden.shape
    raw_qkv = attn.in_proj_qkv(hidden).transpose(1, 2)
    previous_conv = layer_cache.conv_states[0]
    full_qkv = raw_qkv if previous_conv is None else torch.cat([previous_conv, raw_qkv], dim=-1)
    convolved = causal_conv1d_fn(
        full_qkv, attn.conv1d.weight.squeeze(1), attn.conv1d.bias, attn.activation
    )[..., -seq_len:].transpose(1, 2)
    _, key, value = torch.split(convolved, [attn.key_dim, attn.key_dim, attn.value_dim], dim=-1)
    key = key.reshape(batch_size, seq_len, attn.num_k_heads, attn.head_k_dim)
    value = value.reshape(batch_size, seq_len, attn.num_v_heads, attn.head_v_dim).float()
    if attn.num_v_heads != attn.num_k_heads:
        key = key.repeat_interleave(attn.num_v_heads // attn.num_k_heads, dim=2)
    key = key.float()
    key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)

    beta = attn.in_proj_b(hidden).sigmoid().float()
    decay = (-attn.A_log.float().exp() * F.softplus(attn.in_proj_a(hidden).float() + attn.dt_bias)).float()
    previous_recurrent = layer_cache.recurrent_states[0]
    state = (
        previous_recurrent.float().clone()
        if previous_recurrent is not None
        else value.new_zeros(batch_size, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim)
    )
    states = []
    kernel_size = attn.conv_kernel_size
    for index in range(seq_len):
        k, v = key[:, index], value[:, index]
        state = state * decay[:, index].exp()[..., None, None]
        predicted = (state * k.unsqueeze(-1)).sum(dim=-2)
        correction = (v - predicted) * beta[:, index].unsqueeze(-1)
        state = state + k.unsqueeze(-1) * correction.unsqueeze(-2)
        conv = full_qkv[..., : full_qkv.shape[-1] - seq_len + index + 1][..., -kernel_size:].clone()
        states.append((True, conv, state.clone()))
    return states


def validation_forward(target, candidate_ids, main_cache, target_layer_ids, draft_device):
    """Verify one block and save enough state to roll back its rejected tail."""
    decoder = target.model.language_model
    candidate_ids = candidate_ids.to(decoder.embed_tokens.weight.device)
    start = main_cache.get_seq_length()
    positions = torch.arange(start, start + candidate_ids.shape[1], device=candidate_ids.device)[None]
    rope_positions = positions[None, ...].expand(3, -1, -1)
    hidden = decoder.embed_tokens(candidate_ids)
    rope_device = decoder.rotary_emb.inv_freq.device
    rope = decoder.rotary_emb(hidden.to(rope_device), rope_positions.to(rope_device))
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
        # 27B 分到两张 GPU 时，手动逐层验证必须把本层输入和 RoPE 搬到该层设备。
        layer_device = layer.input_layernorm.weight.device
        hidden = hidden.to(layer_device)
        layer_rope = tuple(tensor.to(layer_device) for tensor in rope)
        layer_positions = positions.to(layer_device)
        layer_cache = main_cache.layers[index]
        if main_cache.layer_types[index] == "full_attention":
            old_length = main_cache.get_seq_length(index)
            hidden = layer(
                hidden,
                position_embeddings=layer_rope,
                position_ids=layer_positions,
                attention_mask=mask.to(layer_device),
                past_key_values=main_cache,
            )
            rollback_data.append(old_length)
        else:
            prefix_states = linear_prefix_states(layer.linear_attn, layer.input_layernorm(hidden), layer_cache)
            hidden = layer(
                hidden,
                position_embeddings=layer_rope,
                position_ids=layer_positions,
                attention_mask=None,
                past_key_values=main_cache,
            )
            rollback_data.append(prefix_states)
        if index in target_layer_ids:
            selected_hidden[index] = hidden.to(draft_device)

    hidden = decoder.norm(hidden)
    logits = target.lm_head(hidden)
    features = torch.cat([selected_hidden[i] for i in target_layer_ids], dim=-1)
    return logits, features, rollback_data


def prefill(target, prompt_ids, target_layer_ids, draft_device, temperature, top_p, top_k):
    """主模型缓存完整 prompt，并给出 anchor 和草稿层所需的多层特征。"""
    cache = LayerwiseCache(target.config.text_config.layer_types)
    outputs = target(
        input_ids=prompt_ids,
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
    )
    pending = select_target_token(outputs.logits[:, -1:, :], temperature, top_p, top_k)
    # + 1 的 offset 移动来自官方 dflash
    features = torch.cat([outputs.hidden_states[i + 1].to(draft_device) for i in target_layer_ids], dim=-1)
    return pending, features, cache


def draft(draft_model, target, cache, target_features, pending, block_size, temperature):
    """先增量缓存已接受 context；再并行算块并按相邻 token 选出候选路径。"""
    pending = pending.to(draft_model.fc.weight.device)
    # The anchor is a token already chosen by the target. The remaining slots
    # all contain the mask token and are predicted together in one draft pass.
    noise_ids = torch.cat([pending, pending.new_full((1, block_size - 1), draft_model.mask_token_id)], dim=1)
    # Qwen3.8 常规输入 embedding 路径会额外归一化；DFlash2 需要原始词表权重。
    embeddings = target.get_input_embeddings().weight
    noise_embedding = F.embedding(noise_ids.to(embeddings.device), embeddings)
    noise_embedding = noise_embedding.to(draft_model.fc.weight.device) * draft_model.input_embedding_scale
    hidden = draft_model(cache, target_features, noise_embedding)
    logits = draft_model.compute_logits(hidden[:, 1:, :], target.lm_head)
    return draft_model.candidate_selector(hidden[:, 1:, :], logits, pending[:, 0], temperature)


def verify(
    target,
    pending,
    proposed,
    draft_indices,
    draft_probs,
    main_cache,
    target_layer_ids,
    eos_ids,
    draft_device,
    temperature,
    top_p,
    top_k,
):
    """主模型整块验收；错误后缀必须从主模型 cache 中撤销。"""
    candidate_ids = torch.cat([pending.to(proposed.device), proposed], dim=1)
    logits, features, rollback_data = validation_forward(
        target, candidate_ids, main_cache, target_layer_ids, draft_device
    )
    if temperature > 0:
        target_probs = sampling_probs(logits, temperature, top_p, top_k)
        matches, bonus = rejection_sample(
            proposed.to(logits.device), target_probs, draft_probs.to(logits.device), draft_indices.to(logits.device)
        )
    else:
        greedy = logits[:, :-1, :].argmax(dim=-1)
        matches = (greedy == proposed.to(greedy.device)).to(torch.int32).cumprod(dim=1).sum().item()
        # 全部草稿都正确时，下一 token 来自验证块最后位置的 logits。
        bonus = logits[:, matches, :].argmax(dim=-1)[0]
    accepted_count = matches + 1  # The target's pending token is always accepted.
    if matches < proposed.shape[1]:
        restore_states = []
        for layer_type, state in zip(main_cache.layer_types, rollback_data):
            if layer_type == "full_attention":
                # 全注意力：按 token 排列的 K/V 直接截断。
                restore_states.append((state + accepted_count, None, None))
            else:
                # Gated DeltaNet：递归状态已被错误后缀改写，须恢复前缀快照。
                restore_states.append(state[accepted_count - 1])
        main_cache.restore(restore_states)

    accepted_ids = candidate_ids[0, :accepted_count].tolist()
    eos_index = next((i for i, token_id in enumerate(accepted_ids) if token_id in eos_ids), None)
    if eos_index is not None:
        accepted_ids = accepted_ids[:eos_index]
    next_pending = bonus.reshape(1, 1)
    return accepted_ids, next_pending, features[:, :accepted_count], eos_index is not None


def generate(
    target, draft_model, prompt_ids, max_new_tokens, block_size, eos_ids,
    temperature=0.0, top_p=1.0, top_k=0, trace=False,
):
    generated = []
    accepted_drafts = 0
    draft_device = draft_model.fc.weight.device
    pending, target_features, main_cache = prefill(
        target, prompt_ids, draft_model.target_layer_ids, draft_device, temperature, top_p, top_k
    )
    draft_cache = DFlashContextCache(len(draft_model.layers))
    if trace:
        print(f"prefill: main_cache={main_cache.get_seq_length()}, draft_cache=0, pending={pending.item()}")

    round_index = 0
    while len(generated) < max_new_tokens and pending.item() not in eos_ids:
        round_index += 1
        verify_size = min(block_size, max_new_tokens - len(generated))
        if verify_size == 1:
            generated.append(pending.item())
            break
        context_length = target_features.shape[1]
        proposed, draft_indices, draft_probs = draft(
            draft_model, target, draft_cache, target_features, pending, verify_size, temperature
        )
        if trace:
            print(
                f"round {round_index} draft: context=+{context_length}, "
                f"draft_cache.length={draft_cache.length}, "
                f"physical_KV={draft_cache.keys[0].shape[-2]}, candidates={proposed.tolist()[0]}"
            )
        accepted_ids, pending, accepted_features, stopped = verify(
            target, pending, proposed, draft_indices, draft_probs, main_cache,
            draft_model.target_layer_ids, eos_ids, draft_device, temperature, top_p, top_k,
        )
        if trace:
            print(
                f"round {round_index} verify: accepted={accepted_ids}, "
                f"main_cache={main_cache.get_seq_length()}, next_pending={pending.item()}"
            )
        generated.extend(accepted_ids)
        accepted_drafts += max(0, len(accepted_ids) - 1)
        if stopped:
            break

        target_features = accepted_features
    return generated, accepted_drafts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="讲述下存算一体")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--quantization", choices=("4bit", "8bit"), default="8bit")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 is greedy; >0 enables rejection sampling")
    parser.add_argument("--top-p", type=float, default=1.0, help="Target-model sampling threshold")
    parser.add_argument("--top-k", type=int, default=0, help="Target-model sampling cutoff")
    parser.add_argument("--trace", action="store_true", help="Print cache lengths and accepted tokens after each step")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.temperature < 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error("temperature 和 top-k 必须非负，top-p 必须在 (0, 1] 内")

    model_path = Path(os.environ.get("MODEL_PATH", "/data04/modelzoo/models/llm/Qwen3.8-27B")).expanduser()
    draft_path = Path(os.environ.get("DFLASH_PATH", "/data04/modelzoo/models/llm/Qwen3.8-27B-DFlash2")).expanduser()
    if not torch.cuda.is_available():
        parser.error("需要一张 CUDA GPU")
    device = torch.device("cuda:0")

    # 从本地 BF16 权重加载时量化；两模型固定在同一张 GPU。
    print(f"Target quantization={args.quantization}; both models on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if args.quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        # NF4 量化权重，BF16 计算；这不是 W4A8。
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map={"": str(device)},
        quantization_config=quantization_config,
        local_files_only=True,
    ).eval()
    draft = load_draft(draft_path, model, device)
    if not 2 <= args.block_size <= draft.block_size:
        parser.error(f"--block-size must be between 2 and {draft.block_size}")
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.get_input_embeddings().weight.device)
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else {eos}
    with torch.inference_mode():
        generated, accepted_drafts = generate(
            model, draft, inputs["input_ids"], args.max_new_tokens, args.block_size,
            eos_ids, args.temperature, args.top_p, args.top_k, args.trace,
        )
    print(tokenizer.decode(generated, skip_special_tokens=True))
    print(f"生成 token 总数（不含 EOS）：{len(generated)}")
    print(f"Draft 命中 token 数：{accepted_drafts}")


if __name__ == "__main__":
    main()
