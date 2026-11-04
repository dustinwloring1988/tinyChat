"""Qwen-style dense causal language model used by tinyChat."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.flash_attention import flash_attn
from nanochat.optim import MuonAdamW


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 2
    n_embd: int = 768
    # Kept in checkpoint metadata for backwards compatibility. Qwen dense
    # attention uses the full available context in every layer.
    window_pattern: str = "L"
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    partial_rotary_factor: float = 1.0


class Linear(nn.Linear):
    """Keep fp32 master weights while matching weights to activation dtype."""

    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.square().mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)
        return (x_norm * self.weight.float()).to(input_dtype)


class CausalSelfAttention(nn.Module):
    """Qwen-style gated GQA with Q/K normalization and partial RoPE."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        self.window_size = (config.sequence_len, 0)
        assert config.n_embd % config.n_head == 0
        assert 0 < self.rotary_dim <= self.head_dim and self.rotary_dim % 2 == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.q_proj = Linear(self.n_embd, self.n_head * self.head_dim * 2, bias=False)
        self.k_proj = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = Linear(self.n_head * self.head_dim, self.n_embd, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    @staticmethod
    def _rotate_half(x):
        left, right = x.chunk(2, dim=-1)
        return torch.cat((-right, left), dim=-1)

    def _apply_rope(self, x, cos, sin):
        if self.rotary_dim == 0:
            return x
        x_rot, x_pass = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        cos = cos[..., :self.rotary_dim // 2]
        sin = sin[..., :self.rotary_dim // 2]
        x_rot = (x_rot * cos) + (self._rotate_half(x_rot) * sin)
        return torch.cat((x_rot, x_pass), dim=-1)

    def forward(self, x, cos_sin, kv_cache=None):
        B, T, _ = x.shape
        qg = self.q_proj(x).view(B, T, self.n_head, 2 * self.head_dim)
        q, gate = qg.chunk(2, dim=-1)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = cos_sin
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        if kv_cache is None:
            y = flash_attn.flash_attn_func(
                q, k, v, causal=True, window_size=self.window_size
            )
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True, window_size=self.window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, self.n_head * self.head_dim)
        y = y * torch.sigmoid(gate.reshape(B, T, -1))
        return self.o_proj(y)


class MLP(nn.Module):
    """Qwen's gated SiLU feed-forward block (SwiGLU)."""

    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.gate_proj = Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache=None):
        x = x + self.self_attn(self.input_layernorm(x), cos_sin, kv_cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            "ln_f": RMSNorm(config.n_embd, config.rms_norm_eps),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, config)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, config, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        rotary_dim = int((config.n_embd // config.n_head) * config.partial_rotary_factor)
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim))
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos, sin = freqs.cos().to(COMPUTE_DTYPE), freqs.sin().to(COMPUTE_DTYPE)
        return cos[None, :, None, :], sin[None, :, None, :]

    @torch.no_grad()
    def init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
        # Scale residual projections by depth to stabilize deep decoder stacks.
        residual_std = std / (2 * self.config.n_layer) ** 0.5
        for block in self.transformer.h:
            nn.init.normal_(block.self_attn.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=residual_std)
        for module in self.modules():
            if isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.config)
        self.cos, self.sin = cos, sin
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.transformer.wte.weight.device

    def num_matmul_params(self):
        return sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))

    def estimate_flops(self):
        # Matmul forward/backward plus quadratic full-attention work.
        head_dim = self.config.n_embd // self.config.n_head
        attention = 12 * self.config.n_layer * self.config.n_head * head_dim * self.config.sequence_len
        return 6 * self.num_matmul_params() + attention

    def estimate_decode_flops(self, context_len):
        head_dim = self.config.n_embd // self.config.n_head
        attention = 4 * self.config.n_layer * self.config.n_head * head_dim * min(context_len, self.config.sequence_len)
        return 2 * self.num_matmul_params() + attention

    def estimate_prefill_flops(self, num_tokens):
        head_dim = self.config.n_embd // self.config.n_head
        attention = 2 * self.config.n_layer * self.config.n_head * head_dim * num_tokens * (num_tokens + 1)
        return 2 * self.num_matmul_params() * num_tokens + attention

    def kv_bytes_per_token(self):
        head_dim = self.config.n_embd // self.config.n_head
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * COMPUTE_DTYPE.itemsize

    def kv_read_bytes(self, context_len):
        head_dim = self.config.n_embd // self.config.n_head
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * COMPUTE_DTYPE.itemsize * min(context_len, self.config.sequence_len)

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        norms = sum(p.numel() for p in self.transformer.ln_f.parameters())
        total = wte + lm_head + transformer_matrices + norms
        assert total == sum(p.numel() for p in self.parameters())
        return {"wte": wte, "lm_head": lm_head, "transformer_matrices": transformer_matrices, "scalars": norms, "total": total}

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        matrix_params = [p for name, p in self.named_parameters() if name.startswith("transformer.h.") and p.ndim >= 2]
        vector_params = [p for name, p in self.named_parameters() if p.ndim < 2 and not name.startswith("transformer.wte.")]
        owned = embedding_params + lm_head_params + matrix_params + vector_params
        assert len(owned) == len(list(self.parameters())) and len({id(p) for p in owned}) == len(owned)

        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({self.config.n_embd}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=vector_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            grouped = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind="muon", params=grouped, lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        B, T = idx.shape
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        assert T0 + T <= self.cos.size(1), f"Sequence length exceeds rotary cache: {T0 + T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device
        assert self.cos.dtype == COMPUTE_DTYPE
        cos_sin = self.cos[:, T0:T0 + T], self.sin[:, T0:T0 + T]

        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()

        if targets is not None:
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1, reduction=loss_reduction)
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)[:, -1, :]
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("Inf")
            if temperature > 0:
                next_ids = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1, generator=rng)
            else:
                next_ids = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()
