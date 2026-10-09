"""Two small, self-contained Chunk Prefill demonstrations.

The model is intentionally tiny.  The point is cache management and scheduling:

1. Chunk Prefill produces the same logits as one full Prefill.
2. Chunks from a long prompt can be interleaved with Decode from another request.

Run:
    python demo_chunk_prefill.py
"""

import torch
from torch import nn


class TinyCausalLM(nn.Module):
    """A one-layer causal attention model with an explicit K/V cache."""

    def __init__(self, vocab_size=64, hidden_size=32, num_heads=4):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

    def forward(self, input_ids, past_key=None, past_value=None):
        batch, q_len = input_ids.shape
        hidden = self.embed(input_ids)
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)

        def split_heads(x):
            return x.view(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        past_len = 0 if past_key is None else past_key.shape[-2]
        if past_key is not None:
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        # Query row i corresponds to absolute position past_len + i.
        key_positions = torch.arange(key.shape[-2], device=input_ids.device)
        query_positions = past_len + torch.arange(q_len, device=input_ids.device)
        causal = key_positions[None, :] <= query_positions[:, None]
        # print(causal)
        scores = query @ key.transpose(-1, -2) / self.head_dim**0.5
        scores = scores.masked_fill(~causal[None, None], torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        hidden = attention @ value
        hidden = hidden.transpose(1, 2).reshape(batch, q_len, -1)
        logits = self.lm_head(self.proj(hidden))
        return logits, key, value

    def forward_mixed(self, input_ids_list, past_list):
        """Process variable-length requests in one mixed forward call.

        Each item may be a Prefill chunk or one Decode token.  The requests
        share the embedding/QKV/LM-head work, but their attention and caches
        remain independent (equivalent to a varlen/block-diagonal batch).
        """
        lengths = [ids.shape[1] for ids in input_ids_list]
        hidden = self.embed(torch.cat(input_ids_list, dim=1))
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        total_len = hidden.shape[1]
        query = query.view(1, total_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(1, total_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(1, total_len, self.num_heads, self.head_dim).transpose(1, 2)

        logits_list, new_keys, new_values = [], [], []
        offset = 0
        for ids, past in zip(input_ids_list, past_list):
            q_len = ids.shape[1]
            q = query[:, :, offset : offset + q_len]
            k = key[:, :, offset : offset + q_len]
            v = value[:, :, offset : offset + q_len]
            past_key, past_value = past
            past_len = 0 if past_key is None else past_key.shape[-2]
            if past_key is not None:
                k = torch.cat((past_key, k), dim=-2)
                v = torch.cat((past_value, v), dim=-2)
            key_positions = torch.arange(k.shape[-2], device=ids.device)
            query_positions = past_len + torch.arange(q_len, device=ids.device)
            causal = key_positions[None, :] <= query_positions[:, None]
            scores = q @ k.transpose(-1, -2) / self.head_dim**0.5
            scores = scores.masked_fill(~causal[None, None], torch.finfo(scores.dtype).min)
            attention = scores.softmax(dim=-1)
            output = attention @ v
            output = output.transpose(1, 2).reshape(1, q_len, -1)
            logits_list.append(self.lm_head(self.proj(output)))
            new_keys.append(k)
            new_values.append(v)
            offset += q_len
        return logits_list, list(zip(new_keys, new_values))


def prefill_in_chunks(model, input_ids, chunk_size):
    """Run Prefill chunk by chunk and return logits plus the final cache."""
    key = value = None
    logits_parts = []
    for start in range(0, input_ids.shape[1], chunk_size):
        chunk = input_ids[:, start : start + chunk_size]
        logits, key, value = model(chunk, key, value)
        logits_parts.append(logits)
        print(f"  prefill chunk [{start}:{start + chunk.shape[1]}], cache length={key.shape[-2]}")
    return torch.cat(logits_parts, dim=1), key, value


def demo_only_chunk_prefill():
    print("\n=== Demo 1: full Prefill vs Chunk Prefill ===")
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    prompt = torch.randint(0, 64, (1, 12))

    with torch.no_grad():
        full_logits, full_key, full_value = model(prompt)
        chunk_logits, chunk_key, chunk_value = prefill_in_chunks(model, prompt, chunk_size=4)

    print("full cache:", full_key.shape, "chunk cache:", chunk_key.shape)
    print("max logits difference:", (full_logits - chunk_logits).abs().max().item())
    print("same final K:", torch.allclose(full_key, chunk_key), "same final V:", torch.allclose(full_value, chunk_value))
    assert torch.allclose(full_logits, chunk_logits, atol=1e-6)
    assert torch.allclose(full_key, chunk_key)
    assert torch.allclose(full_value, chunk_value)


def demo_mixed_prefill_decode():
    print("\n=== Demo 2: Chunk Prefill and Decode share the scheduler ===")
    torch.manual_seed(1)
    model = TinyCausalLM().eval()
    long_prompt = torch.randint(0, 64, (1, 12))
    short_prompt = torch.randint(0, 64, (1, 4))
    a_key = a_value = b_key = b_value = None
    b_input = torch.tensor([[7]], dtype=torch.long)

    # Request B is already decoding while request A is still in Prefill.
    with torch.no_grad():
        _, b_key, b_value = model(short_prompt)
        print("request B: initial prefill, cache length=", b_key.shape[-2])
        for start in range(0, long_prompt.shape[1], 4):
            a_chunk = long_prompt[:, start : start + 4]
            # One scheduler iteration: A's Prefill chunk and B's Decode token
            # are concatenated into one variable-length mixed batch.  This is
            # one model forward call, while the two KV caches stay separate.
            logits_list, caches = model.forward_mixed(
                [a_chunk, b_input], [(a_key, a_value), (b_key, b_value)]
            )
            (a_key, a_value), (b_key, b_value) = caches
            print(
                f"one mixed forward: A prefill [{start}:{start + a_chunk.shape[1]}] "
                f"+ B decode 1 token; A cache={a_key.shape[-2]}, B cache={b_key.shape[-2]}"
            )
            b_token = logits_list[1][:, -1].argmax(dim=-1, keepdim=True)
            b_input = b_token
            print("  B next token:", b_token.item())

    print("request A final cache:", a_key.shape, "request B final cache:", b_key.shape)
    print("A and B use independent caches: no K/V is mixed between requests.")


if __name__ == "__main__":
    demo_only_chunk_prefill()
    demo_mixed_prefill_decode()
