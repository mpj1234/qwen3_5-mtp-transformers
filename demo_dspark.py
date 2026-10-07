"""Single-request Qwen3-4B + DSpark speculative decoding, using local weights.

The draft model and its Markov/confidence heads are implemented below; this
example does not import DeepSpec or call Transformers ``generate()``.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python demo_dspark.py --trace

The target and draft checkpoints default to the two local paths in ``main``.
``--temperature 0`` uses greedy verification. Positive temperature uses the
draft/target acceptance ratio and residual distribution from DeepSpec.
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

from transformers import AutoTokenizer, DynamicCache, Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding, rotate_half  # noqa: E402


def apply_rope(states, cos, sin):
    return states * cos[:, None] + rotate_half(states) * sin[:, None]


def sample(logits, temperature):
    if temperature == 0:
        return logits.argmax(dim=-1)
    return torch.multinomial(torch.softmax(logits.float() / temperature, dim=-1), 1).squeeze(-1)


class DraftCache:
    """Only accepted target context enters this cache; the draft block is temporary."""

    def __init__(self, num_layers):
        self.length = 0
        self.keys = [None] * num_layers
        self.values = [None] * num_layers


class DSparkAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def cache_context(self, context, cos, sin, cache, layer_idx):
        """Context contributes K/V, not Q; its tokens were accepted by the target."""
        batch, length, _ = context.shape
        key = self.k_norm(self.k_proj(context).view(batch, length, self.num_kv_heads, self.head_dim))
        key = apply_rope(key.transpose(1, 2), cos, sin)
        value = self.v_proj(context).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if cache.keys[layer_idx] is not None:
            key = torch.cat([cache.keys[layer_idx], key], dim=-2)
            value = torch.cat([cache.values[layer_idx], value], dim=-2)
        cache.keys[layer_idx], cache.values[layer_idx] = key, value

    def forward(self, hidden, cos, sin, cache, layer_idx):
        batch, length, _ = hidden.shape
        query = self.q_norm(self.q_proj(hidden).view(batch, length, self.num_heads, self.head_dim))
        key = self.k_norm(self.k_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim))
        value = self.v_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim)
        query = apply_rope(query.transpose(1, 2), cos, sin)
        key = apply_rope(key.transpose(1, 2), cos, sin)
        value = value.transpose(1, 2)
        # DeepSpec's draft attention is noncausal within this block. The target
        # later verifies every chosen token with its own causal attention.
        key = torch.cat([cache.keys[layer_idx], key], dim=-2)
        value = torch.cat([cache.values[layer_idx], value], dim=-2)
        repeats = self.num_heads // self.num_kv_heads
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        return self.o_proj(output.transpose(1, 2).reshape(batch, length, -1))


class DSparkLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = DSparkAttention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden, cos, sin, cache, layer_idx):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), cos, sin, cache, layer_idx)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class DSparkDraft(nn.Module):
    """The local 5-layer Qwen3 DSpark checkpoint, including both small heads."""

    def __init__(self, config):
        super().__init__()
        self.target_layer_ids = config.target_layer_ids
        self.block_size = int(config.block_size)  # Seven proposal positions, anchor included in the first slot.
        self.mask_token_id = int(config.mask_token_id)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DSparkLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.markov_head = nn.Module()
        self.markov_head.markov_w1 = nn.Embedding(config.vocab_size, config.markov_rank)
        self.markov_head.markov_w2 = nn.Linear(config.markov_rank, config.vocab_size, bias=False)
        self.confidence_head = nn.Module()
        confidence_width = config.hidden_size + config.markov_rank if config.confidence_head_with_markov else config.hidden_size
        self.confidence_head.proj = nn.Linear(confidence_width, 1)
        self.confidence_head_with_markov = bool(config.confidence_head_with_markov)

    def forward(self, cache, target_features, draft_ids):
        # First round: all prompt features. Later: just the accepted prefix
        # from the preceding verification round.
        context = self.hidden_norm(self.fc(target_features))
        positions = torch.arange(cache.length, cache.length + context.shape[1], device=context.device)[None]
        cos, sin = self.rotary_emb(context, positions)
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn.cache_context(context, cos, sin, cache, layer_idx)
        cache.length += context.shape[1]

        hidden = self.embed_tokens(draft_ids)
        positions = torch.arange(cache.length, cache.length + draft_ids.shape[1], device=hidden.device)[None]
        cos, sin = self.rotary_emb(hidden, positions)
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer(hidden, cos, sin, cache, layer_idx)
        return self.norm(hidden)

    def choose_tokens(self, hidden, anchor, temperature, confidence_threshold):
        """One block backbone pass, then a cheap sequential Markov choice."""
        base_logits = self.lm_head(hidden)
        predecessor = anchor[:, 0]
        chosen, draft_probs, confidences = [], [], []
        for position in range(hidden.shape[1]):
            prev_embedding = self.markov_head.markov_w1(predecessor)
            logits = base_logits[:, position] + self.markov_head.markov_w2(prev_embedding)
            token = sample(logits, temperature)
            if self.confidence_head_with_markov:
                confidence_input = torch.cat([hidden[:, position], prev_embedding], dim=-1)
            else:
                confidence_input = hidden[:, position]
            confidence = self.confidence_head.proj(confidence_input).sigmoid()[:, 0]
            if confidence_threshold > 0 and confidence[0].item() < confidence_threshold:
                break
            chosen.append(token)
            confidences.append(confidence)
            if temperature > 0:
                draft_probs.append(torch.softmax(logits.float() / temperature, dim=-1))
            predecessor = token
        if not chosen:
            return anchor[:, :0], None, []
        return torch.stack(chosen, dim=1), torch.stack(draft_probs, dim=1) if draft_probs else None, confidences


def load_draft(path, target, device):
    config = Qwen3Config.from_pretrained(path, local_files_only=True)
    if config.num_target_layers != target.config.num_hidden_layers or config.vocab_size != target.config.vocab_size:
        raise ValueError("DSpark checkpoint does not match the target model")
    if config.markov_head_type != "vanilla" or not config.enable_confidence_head:
        raise ValueError("This demo implements the checkpoint's vanilla Markov and confidence heads")
    if max(config.target_layer_ids) >= target.config.num_hidden_layers - 1:
        raise ValueError("DSpark target_layer_ids must exclude the final normalized layer")
    draft = DSparkDraft(config)
    parameters = draft.state_dict()
    with safe_open(path / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        missing = set(parameters) - set(checkpoint.keys())
        unexpected = set(checkpoint.keys()) - set(parameters)
        if missing or unexpected:
            raise RuntimeError(f"DSpark weights mismatch: missing={missing}, unexpected={unexpected}")
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = checkpoint.get_tensor(name)
                if parameter.shape != value.shape:
                    raise RuntimeError(f"Wrong shape for {name}: {value.shape} != {parameter.shape}")
                parameter.copy_(value)
    return draft.to(device=device, dtype=target.dtype).eval()


def target_features(outputs, layer_ids):
    return torch.cat([outputs.hidden_states[index + 1] for index in layer_ids], dim=-1)


def prefill(target, prompt_ids, layer_ids, temperature):
    cache = DynamicCache()
    outputs = target(
        input_ids=prompt_ids,
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
    )
    pending = sample(outputs.logits[:, -1, :], temperature)[:, None]
    return pending, target_features(outputs, layer_ids), cache


def draft(draft_model, cache, features, pending, block_size, temperature, confidence_threshold):
    ids = pending.new_full((1, block_size), draft_model.mask_token_id)
    ids[:, 0] = pending[:, 0]
    hidden = draft_model(cache, features, ids)
    return draft_model.choose_tokens(hidden, pending, temperature, confidence_threshold)


def verify(target, cache, pending, proposed, draft_probs, layer_ids, temperature):
    candidate_ids = torch.cat([pending, proposed], dim=1)
    old_length = cache.get_seq_length()
    outputs = target(input_ids=candidate_ids, past_key_values=cache, use_cache=True, output_hidden_states=True)
    gamma = proposed.shape[1]
    if gamma == 0:
        matches = 0
        next_token = sample(outputs.logits[:, 0], temperature)[:, None]
    elif temperature == 0:
        greedy = outputs.logits[:, :gamma].argmax(dim=-1)
        matches = (greedy == proposed).long().cumprod(dim=1).sum().item()
        next_token = outputs.logits[:, matches].argmax(dim=-1)[:, None]
    else:
        target_probs = torch.softmax(outputs.logits.float() / temperature, dim=-1)
        p = target_probs[:, :gamma].gather(-1, proposed[..., None]).squeeze(-1)
        q = draft_probs.gather(-1, proposed[..., None]).squeeze(-1).clamp_min(1e-8)
        matches = (torch.rand_like(p) < (p / q).clamp(max=1)).long().cumprod(dim=1).sum().item()
        if matches < gamma:
            residual = (target_probs[:, matches] - draft_probs[:, matches]).clamp_min(0)
            total = residual.sum(dim=-1, keepdim=True)
            residual = torch.where(total > 1e-8, residual / total.clamp_min(1e-8), target_probs[:, matches])
            next_token = torch.multinomial(residual, 1)
        else:
            next_token = torch.multinomial(target_probs[:, -1], 1)

    accepted_count = matches + 1  # Pending was selected by the target before this round.
    remove_count = candidate_ids.shape[1] - accepted_count
    if remove_count:
        cache.crop(-remove_count)
    if cache.get_seq_length() != old_length + accepted_count:
        raise RuntimeError("Target cache rollback did not preserve the accepted prefix")
    features = target_features(outputs, layer_ids)[:, :accepted_count]
    return candidate_ids[:, :accepted_count], next_token, features, matches


def generate(target, draft_model, prompt_ids, max_new_tokens, block_size, temperature, confidence_threshold, eos_ids, trace):
    pending, features, target_cache = prefill(target, prompt_ids, draft_model.target_layer_ids, temperature)
    draft_cache = DraftCache(len(draft_model.layers))
    generated = []
    accepted_drafts = 0
    if trace:
        print(f"prefill: target_cache={target_cache.get_seq_length()}, pending={pending.item()}")

    round_index = 0
    while len(generated) < max_new_tokens and pending.item() not in eos_ids:
        round_index += 1
        remaining = max_new_tokens - len(generated)
        if remaining == 1:
            generated.append(pending.item())
            break
        proposal_size = min(block_size, remaining - 1)
        context_length = features.shape[1]
        proposed, draft_probs, confidence = draft(
            draft_model, draft_cache, features, pending, proposal_size, temperature, confidence_threshold
        )
        accepted, pending, features, matches = verify(
            target, target_cache, pending, proposed, draft_probs, draft_model.target_layer_ids, temperature
        )
        accepted_ids = accepted[0].tolist()
        if trace:
            print(
                f"round {round_index}: context=+{context_length}, draft_cache={draft_cache.length}, "
                f"proposed={proposed.shape[1]}, confidence={[round(x.item(), 3) for x in confidence]}, "
                f"accepted_drafts={matches}, target_cache={target_cache.get_seq_length()}"
            )
        if any(token in eos_ids for token in accepted_ids):
            accepted_ids = accepted_ids[: next(i for i, token in enumerate(accepted_ids) if token in eos_ids)]
            accepted_drafts += max(0, len(accepted_ids) - 1)
            generated.extend(accepted_ids)
            break
        accepted_drafts += matches
        generated.extend(accepted_ids)
    return generated, accepted_drafts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="讲述下存算一体")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=7, help="Number of DSpark proposal positions")
    parser.add_argument("--temperature", type=float, default=0.0, help="0: greedy; positive: rejection sampling")
    parser.add_argument("--confidence-threshold", type=float, default=0.0, help="0 disables confidence pruning")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.temperature < 0 or not 0 <= args.confidence_threshold <= 1:
        parser.error("max-new-tokens must be positive; temperature and confidence-threshold must be nonnegative")

    target_path = Path(os.environ.get("MODEL_PATH", "/data04/modelzoo/models/llm/Qwen3-4B"))
    draft_path = Path(os.environ.get("DSPARK_PATH", "/data04/modelzoo/models/llm/dspark_qwen3_4b_block7"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; choose --device cpu or run with GPU access")
    tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    target = Qwen3ForCausalLM.from_pretrained(target_path, dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    draft_model = load_draft(draft_path, target, device)
    if not 1 <= args.block_size <= draft_model.block_size:
        parser.error(f"--block-size must be between 1 and {draft_model.block_size}")
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )["input_ids"].to(device)
    eos = target.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else {eos}
    with torch.inference_mode():
        generated, accepted = generate(
            target, draft_model, prompt_ids, args.max_new_tokens, args.block_size,
            args.temperature, args.confidence_threshold, eos_ids, args.trace,
        )
    print(tokenizer.decode(generated, skip_special_tokens=True))
    print(f"生成 token 总数（不含 EOS）：{len(generated)}")
    print(f"Draft 命中 token 数：{accepted}")


if __name__ == "__main__":
    main()
