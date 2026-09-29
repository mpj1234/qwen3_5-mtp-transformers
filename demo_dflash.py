"""Greedy Qwen3.5 generation with an explicit DFlash draft/verify loop.

Set MODEL_PATH and DFLASH_PATH to local copies of Qwen/Qwen3.5-4B and
z-lab/Qwen3.5-4B-DFlash. The main model caches accepted tokens and rolls
back rejected draft suffixes. The drafter caches accepted context K/V states.
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
    return config.dflash_config.get(name, getattr(config, name, default))


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
    """Small Qwen3.5 cache that can restore a partially accepted block."""

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
    """Draft K/V for accepted target tokens; the temporary mask block is never stored."""

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
        """Project each newly accepted target feature just once."""
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
            mask = ((distance >= 0) & (distance < self.sliding_window))[None, None]
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
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), cos, sin, cache, layer_idx)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class DFlashDraft(nn.Module):
    """The checkpoint's six-layer drafter; it shares embeddings and LM head with Qwen3.5."""

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
        self.layers = nn.ModuleList([DFlashDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def forward(self, cache, target_features, noise_embedding):
        # Only accepted target tokens enter the cache. The anchor and mask block
        # are temporary and are recomputed on every draft call.
        context = self.hidden_norm(self.fc(target_features))
        context_positions = torch.arange(cache.length, cache.length + context.shape[1], device=context.device)[None]
        context_cos, context_sin = self.rotary_emb(context, context_positions)
        for index, layer in enumerate(self.layers):
            layer.self_attn.cache_context(context, context_cos, context_sin, cache, index)
        cache.length += context.shape[1]

        hidden = noise_embedding
        positions = torch.arange(cache.length, cache.length + hidden.shape[1], device=hidden.device)[None]
        cos, sin = self.rotary_emb(hidden, positions)
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, cos, sin, cache, index)
        return self.norm(hidden)

    def compute_logits(self, hidden, output_head):
        logits = output_head(hidden) * self.output_multiplier
        softcap = self.final_logit_softcapping
        if softcap is not None and float(softcap) > 0:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        return logits


def load_draft(path, target):
    config = Qwen3Config.from_pretrained(path, local_files_only=True)
    if config.num_target_layers != target.config.text_config.num_hidden_layers:
        raise ValueError("DFlash checkpoint does not match the target model's layer count")
    draft = DFlashDraft(config)
    weights_file = path / "model.safetensors"
    parameters = draft.state_dict()
    with safe_open(weights_file, framework="pt", device="cpu") as checkpoint:
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


def validation_forward(target, candidate_ids, main_cache, target_layer_ids):
    """Verify one block and save enough state to roll back its rejected tail."""
    decoder = target.model.language_model
    start = main_cache.get_seq_length()
    positions = torch.arange(start, start + candidate_ids.shape[1], device=candidate_ids.device)[None]
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
    selected_hidden = {}
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
            selected_hidden[index] = hidden

    logits = target.lm_head(decoder.norm(hidden))
    features = torch.cat([selected_hidden[i] for i in target_layer_ids], dim=-1)
    return logits, features, rollback_data


def prefill(target, prompt_ids, target_layer_ids):
    """Cache the prompt in Qwen3.5 and choose the first pending token."""
    cache = LayerwiseCache(target.config.text_config.layer_types)
    outputs = target(
        input_ids=prompt_ids,
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
    )
    pending = outputs.logits[:, -1:, :].argmax(dim=-1)
    # + 1 的 offset 移动来自官方 dflash
    features = torch.cat([outputs.hidden_states[i + 1] for i in target_layer_ids], dim=-1)
    return pending, features, cache


def draft(draft_model, target, cache, target_features, pending, block_size):
    """Predict the masked positions in one DFlash forward pass."""
    # The anchor is a token already chosen by the target. The remaining slots
    # all contain the mask token and are predicted together in one draft pass.
    noise_ids = torch.cat([pending, pending.new_full((1, block_size - 1), draft_model.mask_token_id)], dim=1)
    # Qwen3.5 normalizes its usual input embeddings. DFlash expects the raw table.
    noise_embedding = F.embedding(noise_ids, target.get_input_embeddings().weight)
    noise_embedding = noise_embedding * draft_model.input_embedding_scale
    hidden = draft_model(cache, target_features, noise_embedding)
    logits = draft_model.compute_logits(hidden[:, 1:, :], target.lm_head)
    return logits.argmax(dim=-1)


def verify(target, pending, proposed, main_cache, target_layer_ids, eos_ids):
    """Accept the greedy matching prefix and roll back rejected cache states."""
    candidate_ids = torch.cat([pending, proposed], dim=1)
    logits, features, rollback_data = validation_forward(target, candidate_ids, main_cache, target_layer_ids)
    greedy = logits[:, :-1, :].argmax(dim=-1)
    matches = (greedy == proposed).to(torch.int32).cumprod(dim=1).sum().item()
    accepted_count = matches + 1  # The target's pending token is always accepted.
    if matches < proposed.shape[1]:
        restore_states = []
        for layer_type, state in zip(main_cache.layer_types, rollback_data):
            if layer_type == "full_attention":
                restore_states.append((state + accepted_count, None, None))
            else:
                restore_states.append(state[accepted_count - 1])
        main_cache.restore(restore_states)

    accepted_ids = candidate_ids[0, :accepted_count].tolist()
    eos_index = next((i for i, token_id in enumerate(accepted_ids) if token_id in eos_ids), None)
    if eos_index is not None:
        accepted_ids = accepted_ids[:eos_index]
    next_pending = logits[:, matches : matches + 1, :].argmax(dim=-1)
    return accepted_ids, next_pending, features[:, :accepted_count], eos_index is not None


def generate(target, draft_model, prompt_ids, max_new_tokens, block_size, eos_ids):
    generated = []
    accepted_drafts = 0
    pending, target_features, main_cache = prefill(target, prompt_ids, draft_model.target_layer_ids)
    draft_cache = DFlashContextCache(len(draft_model.layers))

    while len(generated) < max_new_tokens and pending.item() not in eos_ids:
        verify_size = min(block_size, max_new_tokens - len(generated))
        if verify_size == 1:
            generated.append(pending.item())
            break
        # print("target_features:", target_features.shape)  # 打开注释就能看到接受长度的变化，以及cache context的长度增加
        proposed = draft(draft_model, target, draft_cache, target_features, pending, verify_size)
        accepted_ids, pending, accepted_features, stopped = verify(
            target, pending, proposed, main_cache, draft_model.target_layer_ids, eos_ids
        )
        generated.extend(accepted_ids)
        accepted_drafts += max(0, len(accepted_ids) - 1)
        if stopped:
            break

        target_features = accepted_features
    return generated, accepted_drafts


def main():
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
        device = (
            f"cuda:{max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])}"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = str(torch.device(args.device))
    print(f"Running on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map={"": device}, local_files_only=True
    ).eval()
    draft = load_draft(draft_path, model)
    if not 2 <= args.block_size <= draft.block_size:
        parser.error(f"--block-size must be between 2 and {draft.block_size}")
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else {eos}
    with torch.inference_mode():
        generated, accepted_drafts = generate(
            model, draft, inputs["input_ids"], args.max_new_tokens, args.block_size, eos_ids
        )
    print(tokenizer.decode(generated, skip_special_tokens=True))
    print(f"DFlash accepted drafts: {accepted_drafts}")


if __name__ == "__main__":
    main()
