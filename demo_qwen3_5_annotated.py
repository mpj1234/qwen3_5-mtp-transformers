"""Qwen3.5 0.8B MTP 投机解码教学版：逐行标注了实际运行的实现。

运行：MODEL_PATH=/data/peijue.ma/modelscope/Qwen3.5-0.8B \
      HF_HUB_OFFLINE=1 python demo_qwen3_5_annotated.py --device cpu --max-new-tokens 12
默认尝试选择空闲显存最多的一张 CUDA 卡；模型和 MTP 始终放在同一设备。

一、状态对齐：主模型在 token x_i 上的 hidden h_i，与 MTP 输入 x_(i+1) 配对。
MTP(x_(i+1), h_i) 输出下一个候选 x_(i+2)。MTP 预测出的 token 并不会
自动写进 MTP cache；只有它在下一次作为 MTP 的输入时才会写入。

二、主模型 prefill：整段 prompt 只调用一次主模型 forward，得到主 cache、
每个 prompt token 的 hidden，以及第一个待提交 token pending。
MTP prefill：只调用一次 MTP，输入 prompt[1:] 和主模型 hidden[:-1]。
MTP cache 因而处理到 prompt 最后一个 token；pending 尚未进入 cache。
后续轮次始终沿用 MTP cache，不重复执行 prefill_mtp。

三、draft：第一次 MTP 调用处理当前输入，生成候选 d1；之后依次把前一
候选作为输入生成 d2、d3、d4。这个 checkpoint 只有一层 MTP，因此后续
候选的配对 hidden 来自该 MTP 层本身。保存每次调用后的 MTP cache 快照。
这些快照是“已处理输入”的状态，而不是“已处理输出候选”的状态。

四、validation_forward：把 [pending,d1,d2,d3,d4] 当成一个 chunk，逐层
调用主模型 decoder layer；每层仅进行一次 chunk forward。全注意力层缓存
完整 KV，可用长度切片回退。线性注意力层 forward 仅留下 chunk 末尾状态，
故额外根据该层归一化后的输入计算各 token 的卷积状态和递归状态快照。
这一步是轻量状态计算，没有再次执行 decoder layer forward。

五、verify：比较主模型在 pending 位置预测的 d1、在 d1 位置预测的 d2，
以此类推，遇到首次不一致便停止。若匹配 m 个 draft，主 cache 只保留
[pending,d1,...,dm]。全注意力层直接截 KV；线性注意力层直接替换为
第 1+m 个 token 对应的卷积/递归状态。主模型在不匹配位置给出的 token
成为下一轮 pending。无需为回退再调用主模型 forward。

六、MTP cache 回退：假设本轮生成四个候选 d1..d4，调用 MTP 时实际写入
的是本轮输入以及 d1..d3；d4 尚未写入。若 m<4，恢复第 m 个快照，
它恰好处理到了最后一个被接受的候选。若 m=4，保留最后一个快照，
它处理到了 d3；下一轮把 [d4, bonus] 一起送入 MTP。对应主模型 hidden
是 [h_d3,h_d4]。这解释了下一轮 draft 的 input_ids 长度可能为 2。
若 m<4，下一轮只送入主模型给出的替换 token bonus，并配 h_dm；长度为 1。
即使某轮输入长度为 2，第一次调用后保存的快照也是这两个输入都已进入
cache 的状态，所以仍可按匹配数定位回退快照。

七、限制：示例是 batch=1、贪心解码、一个 MTP 层。chunk 计算与逐 token
计算在 BF16 下可能产生不同舍入，因此不能无条件承诺逐 token 位级一致。
线性注意力的逐前缀快照通过显式递归计算，仍有额外计算和存储开销。
"""


import argparse
import copy
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: E402
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    causal_conv1d_fn,
)


MODEL_PATH = Path(os.environ["MODEL_PATH"]).expanduser()


# 单层 cache 同时准备全注意力 KV 和线性注意力状态字段。
class LayerCache:
    def __init__(self):
        self.keys = None
        self.values = None
        self.conv_states = {0: None}
        self.recurrent_states = {0: None}
        self.conv_kernel_size = {0: None}
        self.previous = False
        self.record_past = False


# 按 decoder 层顺序保存 cache，供模型 forward 和显式回退使用。
class LayerwiseCache:
    """Minimal cache interface used by Qwen3.5 attention layers in this demo."""

    def __init__(self, layer_types, mtp=False):
        self.layer_types = layer_types
        self.layers = [LayerCache() for _ in layer_types]
        self.mtp = mtp

    def get_seq_length(self, layer_idx=None):
        if layer_idx is None:
            layer_idx = self.layer_types.index("full_attention")
        keys = self.layers[layer_idx].keys
        return 0 if keys is None else keys.shape[-2]

    def get_query_offset(self, layer_idx=0):
        return self.get_seq_length(layer_idx) + int(self.mtp)

    def get_mask_sizes(self, query_length, layer_idx):
        return self.get_seq_length(layer_idx) + query_length, int(self.mtp)

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
        layer.conv_states[state_idx] = torch.nn.functional.pad(
            full_states[..., -conv_kernel_size:], (max(0, conv_kernel_size - full_states.shape[-1]), 0)
        )
        return full_states

    def update_recurrent_state(self, new_state, layer_idx, state_idx=0):
        self.layers[layer_idx].recurrent_states[state_idx] = new_state.clone()
        return new_state

    # 保存各层可恢复的状态：全注意力记录 KV 长度，线性注意力复制两个可变状态。
    def snapshot(self):
        """Keep K/V lengths and copy each linear layer's mutable states."""
        states = []
        for layer_type, layer in zip(self.layer_types, self.layers):
            if layer_type == "full_attention":
                states.append((self.get_seq_length(len(states)), None, None))
            else:
                conv = layer.conv_states[0]
                recurrent = layer.recurrent_states[0]
                states.append(
                    (layer.previous, None if conv is None else conv.clone(), None if recurrent is None else recurrent.clone())
                )
        return states

    # 逐层恢复给定快照；不会再次执行任一 decoder layer。
    def restore(self, states):
        """Rollback every decoder layer to the last accepted candidate."""
        for layer_type, layer, (marker, conv, recurrent) in zip(self.layer_types, self.layers, states):
            if layer_type == "full_attention":
                layer.keys = layer.keys[..., :marker, :]
                layer.values = layer.values[..., :marker, :]
            else:
                layer.previous = marker
                layer.conv_states[0] = conv
                layer.recurrent_states[0] = recurrent


# 重建 checkpoint 中的单层 MTP 模块，并复用主模型的词嵌入和输出头。
class Qwen35MTP(nn.Module):
    """The checkpoint's MTP block, with shared token embeddings and LM head."""

    def __init__(self, main_model):
        super().__init__()
        config = copy.deepcopy(main_model.config.text_config)
        if config.mtp_num_hidden_layers != 1 or config.mtp_use_dedicated_embeddings:
            raise ValueError("This demo expects one MTP layer with shared embeddings")
        config.num_hidden_layers = 1
        config.layer_types = ["full_attention"]
        width = config.hidden_size
        self.config = config
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(width, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(width, eps=config.rms_norm_eps)
        self.fc = nn.Linear(width * 2, width, bias=False)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, 0)])
        self.norm = Qwen3_5RMSNorm(width, eps=config.rms_norm_eps)
        self.embed_tokens = main_model.model.language_model.embed_tokens
        self.rotary_emb = main_model.model.language_model.rotary_emb
        self.lm_head = main_model.lm_head

    # MTP 可一次接收一个或两个输入，按位置生成最后一个输入之后的候选。
    def forward(self, token_ids, previous_hidden, cache):
        # MTP token at position i+1 is paired with main hidden state at i.
        start = cache.get_seq_length()
        positions = torch.arange(start + 1, start + 1 + token_ids.shape[1], device=token_ids.device)
        text_positions = positions[None, :]
        rope_positions = text_positions[None, ...].expand(3, -1, -1)
        embedding = self.embed_tokens(token_ids)
        hidden = self.fc(
            torch.cat(
                [self.pre_fc_norm_embedding(embedding), self.pre_fc_norm_hidden(previous_hidden)], dim=-1
            )
        )
        mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=cache,
            position_ids=text_positions,
            layer_idx=0,
        )
        hidden = self.layers[0](
            hidden,
            position_embeddings=self.rotary_emb(hidden, rope_positions),
            position_ids=text_positions,
            attention_mask=mask,
            past_key_values=cache,
        )
        next_token = self.lm_head(self.norm(hidden[:, -1:, :]))[:, -1, :].argmax(dim=-1, keepdim=True)
        return next_token, hidden[:, -1:, :]


# 从本地 safetensors 读取 mtp.* 权重，装入教学版单层 MTP。
def load_mtp(main_model):
    mtp = Qwen35MTP(main_model).to(device=main_model.device, dtype=main_model.dtype)
    weights_file = MODEL_PATH / "model.safetensors-00001-of-00001.safetensors"
    with safe_open(weights_file, framework="pt", device="cpu") as checkpoint:
        weights = {
            key.removeprefix("mtp."): checkpoint.get_tensor(key)
            for key in checkpoint.keys()
            if key.startswith("mtp.")
        }
    missing, unexpected = mtp.load_state_dict(weights, strict=False)
    allowed_missing = {key for key in missing if key.startswith(("embed_tokens.", "lm_head."))}
    if unexpected or set(missing) != allowed_missing:
        raise RuntimeError(f"MTP weights do not match: missing={missing}, unexpected={unexpected}")
    mtp.eval()
    return mtp


# 主模型一次处理全部 prompt，取得第一个待提交 token、全部 hidden、主 cache。
def prefill(model, inputs):
    cache = LayerwiseCache(model.config.text_config.layer_types)
    outputs = model(**inputs, past_key_values=cache, use_cache=True, logits_to_keep=1, output_hidden_states=True)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return next_token, outputs.hidden_states[-1], outputs.past_key_values


# MTP 只预填一次 prompt 的移位序列，不在此处生成首个 draft。
def prefill_mtp(mtp, prompt_ids, main_hidden, mtp_cache):
    # 对长度为 N 的 prompt，MTP 只消费 x_1 ... x_(N-1)，并分别搭配 h_0 ... h_(N-2)。
    # 最终 MTP cache 覆盖到 prompt 的最后一个 token；主模型新预测的 pending 留给 draft。
    # 这一步只执行一次：后续轮次靠 cache 接续，无需重复对整段 prompt 做 MTP prefill。
    shifted_ids = prompt_ids[:, 1:]
    if shifted_ids.shape[1] > 0:
        mtp(shifted_ids, main_hidden[:, :-1, :], mtp_cache)


# 从当前 MTP cache 出发连续提出最多 count 个候选，每次调用后保存快照。
def draft(mtp, input_ids, previous_hidden, mtp_cache, count, eos_ids):
    # 普通轮次 input_ids 只有 pending；全接受后，它是 [上一轮末候选, bonus]。
    # MTP 的一次 forward 可以顺序处理这两个输入，但只返回最后位置的预测。
    # cache_states[k] 对应第 k 次调用结束后的状态，保存的是已输入 token 的状态。
    first_token, first_hidden = mtp(input_ids, previous_hidden, mtp_cache)

    drafted = [first_token]
    # 第一次快照记录全部 input_ids 已写入 MTP cache；长度为 2 时两者均已写入。
    cache_states = [mtp_cache.snapshot()]
    token, hidden = first_token, first_hidden
    while len(drafted) < count and token.item() not in eos_ids:
        # 只有一个 MTP 层，后续候选使用该层上一步的 hidden 继续预测。
        token, hidden = mtp(token, hidden, mtp_cache)
        drafted.append(token)
        # 保存本次输入后的 cache；验证失败时可以定位最后一个已接受候选。
        cache_states.append(mtp_cache.snapshot())
    return torch.cat(drafted, dim=1), cache_states


# 从线性注意力层的归一化输入计算每个前缀的卷积和递归状态。
def linear_prefix_states(attn, hidden, layer_cache):
    # 真实线性注意力的 chunk forward 只保留最后一个 recurrent state。
    # 为了在候选中途拒绝时直接回退，这里额外计算每个前缀的状态快照。
    # hidden 必须已经过对应层的 input_layernorm，才能与真实层的输入一致。
    """Capture the conv and recurrent cache after each token in a chunk."""
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


# 一次 chunk 验证所有候选，保留足够的每层回退信息。
def validation_forward(model, candidate_ids, main_cache):
    # candidate_ids = [pending, d1, ..., dK]；每层 decoder 只 forward 一次整个 chunk。
    # 全注意力记录进入 chunk 之前的 KV 长度；线性注意力记录每个 token 结束时的状态。
    """Run the candidate chunk and retain states needed for partial acceptance."""
    decoder = model.model.language_model
    start = main_cache.get_seq_length()
    positions = torch.arange(start, start + candidate_ids.shape[1], device=candidate_ids.device)[None, :]
    rope_positions = positions[None, ...].expand(3, -1, -1)
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
    for index, layer in enumerate(decoder.layers[: decoder.config.num_hidden_layers]):
        layer_cache = main_cache.layers[index]
        if main_cache.layer_types[index] == "full_attention":
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
            # 必须先做该层 input_layernorm，再计算逐前缀状态，否则与真实线性注意力不一致。
            prefix_states = linear_prefix_states(layer.linear_attn, layer.input_layernorm(hidden), layer_cache)
            hidden = layer(
                hidden,
                position_embeddings=rope,
                position_ids=positions,
                attention_mask=None,
                past_key_values=main_cache,
            )
            rollback_data.append(prefix_states)

    hidden = decoder.norm(hidden)
    logits = model.lm_head(hidden)
    return hidden, logits, rollback_data


# 匹配连续正确的 draft，并把主 cache 回退到已接受前缀。
def verify(model, pending, drafted, main_cache, attention_mask):
    # logits[j] 是主模型处理 candidate_ids[j] 后对下一 token 的预测。
    # 因此 logits[0] 与 d1 比，logits[1] 与 d2 比，直到遇到首次不一致。
    # matched=m 时主 cache 仅提交 [pending, d1, ..., dm]，长度为 1+m。
    candidate_ids = torch.cat([pending, drafted], dim=1)
    # 主模型验证整个候选块一次，并取得每一层的回退数据。
    hidden, logits, rollback_data = validation_forward(model, candidate_ids, main_cache)
    proposed = logits[:, :-1, :].argmax(dim=-1)

    matched = 0
    for proposed_id, drafted_id in zip(proposed[0], drafted[0]):
        if proposed_id.item() != drafted_id.item():
            break
        matched += 1

    accepted_count = 1 + matched  # the pending main-model token is always accepted
    if matched < drafted.shape[1]:
        restore_states = []
        for layer_type, state in zip(main_cache.layer_types, rollback_data):
            if layer_type == "full_attention":
                # 全注意力层只保留旧 KV 长度加上已接受 token 数。
                restore_states.append((state + accepted_count, None, None))
            else:
                # 线性注意力层选择已接受前缀结束处的卷积和递归状态。
                restore_states.append(state[accepted_count - 1])
        # 直接替换/裁剪主 cache，不再调用 decoder layer。
        main_cache.restore(restore_states)
    accepted_mask = torch.cat([attention_mask, attention_mask.new_ones((1, accepted_count))], dim=1)
    next_token = logits[:, matched, :].argmax(dim=-1, keepdim=True)
    return next_token, hidden[:, :accepted_count, :], main_cache, accepted_mask, matched


# 加载本地模型后运行 prefill、一次 MTP prefill 和多轮 draft/verify。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--draft-tokens", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--device", default="auto", help="Single device, for example cuda:0 or cpu")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")

    if args.device == "auto":
        device = (
            f"cuda:{max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])}"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = str(torch.device(args.device))
    print(f"Running on {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype="auto", device_map={"": device}, local_files_only=True
    ).eval()
    mtp = load_mtp(model)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    eos = model.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else {eos}
    generated = []
    accepted_drafts = 0
    with torch.inference_mode():
        pending, main_hidden, main_cache = prefill(model, inputs)
        attention_mask = inputs["attention_mask"]
        mtp_cache = LayerwiseCache(mtp.config.layer_types, mtp=True)
        if args.max_new_tokens > 1 and pending.item() not in eos_ids:
            # 整次生成过程只做这一次 MTP prefill；后续通过 cache 续写。
            prefill_mtp(mtp, inputs["input_ids"], main_hidden, mtp_cache)
        mtp_next_ids = pending
        mtp_previous_hidden = main_hidden[:, -1:, :]
        while len(generated) < args.max_new_tokens:
            if pending.item() in eos_ids:
                break
            if len(generated) + 1 == args.max_new_tokens:
                generated.append(pending.item())
                break

            draft_count = min(args.draft_tokens, args.max_new_tokens - len(generated) - 1)
            drafted, mtp_states = draft(
                mtp,
                mtp_next_ids,
                mtp_previous_hidden,
                mtp_cache,
                draft_count,
                eos_ids,
            )

            next_token, accepted_hidden, main_cache, attention_mask, matched = verify(
                model, pending, drafted, main_cache, attention_mask
            )
            # MTP 逐层回退到最后被接受的输入；全接受时最后 draft 仍未进 cache。
            mtp_cache.restore(mtp_states[min(matched, drafted.shape[1] - 1)])
            accepted = torch.cat([pending, drafted[:, :matched]], dim=1)
            accepted_token_ids = accepted[0].tolist()
            eos_at = next((i for i, token_id in enumerate(accepted_token_ids) if token_id in eos_ids), None)
            if eos_at is not None:
                accepted_token_ids = accepted_token_ids[:eos_at]
            generated.extend(accepted_token_ids)
            accepted_drafts += min(matched, len(accepted_token_ids) - 1)
            if eos_at is not None:
                break
            if matched == drafted.shape[1]:
                # 例：drafted=[d1,d2,d3,d4] 全接受，主模型另外预测出 bonus=b。
                # 此时 MTP cache 只处理到 d3：d4 是上次 MTP 的输出，尚未作为输入写入。
                # 下一轮一次送入 [d4,b]；处理 d4 时用主模型的 h_d3，处理 b 时用 h_d4。
                # 这就是下一轮 draft() 的 input_ids 长度为 2 的原因。
                mtp_next_ids = torch.cat([drafted[:, -1:], next_token], dim=1)
                # accepted_hidden 此时末两项恰好是 [h_d3,h_d4]，与 [d4,b] 逐项配对。
                mtp_previous_hidden = accepted_hidden[:, -2:, :]
            else:
                # 例：d1、d2 正确，d3 错误。MTP cache 已回退到“处理完 d2”的快照。
                # 主模型在 d2 后给出的正确 token 是 b；下一轮只需送入 b，不能重送 d2。
                mtp_next_ids = next_token
                # accepted_hidden 的最后一项是 h_d2，因此 b 与 h_d2 配对。
                # 若一个 draft 都没接受，cache 处理到 pending，这里相应使用 h_pending。
                mtp_previous_hidden = accepted_hidden[:, -1:, :]
            pending = next_token

    print(tokenizer.decode(generated, skip_special_tokens=True))
    print(f"MTP accepted drafts: {accepted_drafts}")


if __name__ == "__main__":
    main()
