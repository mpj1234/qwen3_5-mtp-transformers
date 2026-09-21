"""Greedy Qwen3.5 text generation with an explicit one-layer MTP draft/verify loop."""

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


def prefill(model, inputs):
    cache = LayerwiseCache(model.config.text_config.layer_types)
    outputs = model(**inputs, past_key_values=cache, use_cache=True, logits_to_keep=1, output_hidden_states=True)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return next_token, outputs.hidden_states[-1], outputs.past_key_values


def prefill_mtp(mtp, prompt_ids, main_hidden, mtp_cache):
    # Populate the MTP cache through the final prompt token. The pending
    # main-model token is left for draft(), which makes the first proposal.
    shifted_ids = prompt_ids[:, 1:]
    if shifted_ids.shape[1] > 0:
        mtp(shifted_ids, main_hidden[:, :-1, :], mtp_cache)


def draft(mtp, input_ids, previous_hidden, mtp_cache, count, eos_ids):
    first_token, first_hidden = mtp(input_ids, previous_hidden, mtp_cache)

    drafted = [first_token]
    cache_states = [mtp_cache.snapshot()]  # cache through pending, then through each drafted input
    token, hidden = first_token, first_hidden
    while len(drafted) < count and token.item() not in eos_ids:
        # Only one MTP layer is present. Reuse its own last hidden state to
        # autoregressively propose more candidates; the main model verifies all.
        token, hidden = mtp(token, hidden, mtp_cache)
        drafted.append(token)
        cache_states.append(mtp_cache.snapshot())
    return torch.cat(drafted, dim=1), cache_states


def linear_prefix_states(attn, hidden, layer_cache):
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


def validation_forward(model, candidate_ids, main_cache):
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

    # Process the full candidate chunk once per decoder layer. Keep each linear
    # layer's input and pre-chunk state in case only a prefix is accepted.
    # BF16 chunk and single-token attention may round differently.
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


def verify(model, pending, drafted, main_cache, attention_mask):
    candidate_ids = torch.cat([pending, drafted], dim=1)
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
                restore_states.append((state + accepted_count, None, None))
            else:
                restore_states.append(state[accepted_count - 1])
        main_cache.restore(restore_states)
    accepted_mask = torch.cat([attention_mask, attention_mask.new_ones((1, accepted_count))], dim=1)
    next_token = logits[:, matched, :].argmax(dim=-1, keepdim=True)
    return next_token, hidden[:, :accepted_count, :], main_cache, accepted_mask, matched


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
            # MTP has processed pending and all but the final draft. Remove
            # states for drafts rejected by the main model, layer by layer.
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
                # The last accepted draft has not entered the MTP cache yet.
                mtp_next_ids = torch.cat([drafted[:, -1:], next_token], dim=1)
                mtp_previous_hidden = accepted_hidden[:, -2:, :]
            else:
                mtp_next_ids = next_token
                mtp_previous_hidden = accepted_hidden[:, -1:, :]
            pending = next_token

    print(tokenizer.decode(generated, skip_special_tokens=True))
    print(f"MTP accepted drafts: {accepted_drafts}")


if __name__ == "__main__":
    main()
