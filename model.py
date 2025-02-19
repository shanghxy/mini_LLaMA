import math

# from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class Config(PretrainedConfig):
    model_type: str = "mini_llama"

    def __init__(
        self,
        hidden_size: int = 512,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        flash_attn: bool = True,
        max_seq_len: int = 512,
        intermediate_size: int = 2048,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        vocab_size: int = 6400,
        n_layers: int = 8,
        dropout: float = 0.1,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.flash_attn = flash_attn
        self.max_seq_len = max_seq_len
        self.intermediate_size = intermediate_size
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        super().__init__(**kwargs)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, h):
        dtype = h.dtype
        h = h.to(torch.float32)
        var = h.pow(2).mean(dim=-1, keepdim=True)
        h = h * torch.rsqrt(var + self.eps)
        return h.to(dtype) * self.scale


#     def extra_repr(self):
#         return f"{tuple(self.scale.shape)}, eps = {self.eps}"


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_emb = q * cos + rotate_half(q) * sin
    k_emb = k * cos + rotate_half(k) * sin
    return q_emb, k_emb


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        pos_id = torch.arange(0, max_len).float()
        freqs = pos_id[:, None] * self.inv_freq[None, :]  # (max_len, dim // 2)
        freqs = torch.cat([freqs, freqs], dim=-1)  # (max_len, dim)

        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, q, k):
        n_seq = q.shape[1]
        cos = self.cos[None, :n_seq, :]  # (1, n_seq, dim)
        sin = self.sin[None, :n_seq, :]  # (1, n_seq, dim)
        return apply_rotary_pos_emb(q, k, cos, sin)


def repeat_kv(h, n_rep):
    bsz, n_seq, n_kv_head, dim_head = h.shape
    if n_rep == 1:
        return h
    h = h[:, :, :, None, :].expand(bsz, n_seq, n_kv_head, n_rep, dim_head)
    return h.reshape(bsz, n_seq, n_kv_head * n_rep, dim_head)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.max_seq_len = config.max_seq_len
        self.hidden_size = config.hidden_size
        self.n_head = config.num_attention_heads
        assert self.hidden_size % self.n_head == 0
        self.head_dim = self.hidden_size // self.n_head
        self.n_kv_head = config.num_key_value_heads
        assert self.n_head % self.n_kv_head == 0  # what is that?
        self.n_kv_groups = self.n_head // self.n_kv_head
        self.is_causal = True
        self.flash_attn = config.flash_attn
        self.bias = config.attention_bias

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.k_proj = nn.Linear(
            self.hidden_size, self.n_kv_head * self.head_dim, bias=self.bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.n_kv_head * self.head_dim, bias=self.bias
        )

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)

        self.res_dropout = nn.Dropout(self.dropout)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_len=self.max_seq_len)

        self.k_cache, self.v_cache = None, None

    def forward(self, h, use_kv_cache=False):
        bsz, n_seq, dim = h.shape
        if use_kv_cache and self.eval():
            if self.k_cache is None or self.k_cache.shape[1] < n_seq:
                q = self.q_proj(h)
                k = self.k_proj(h)
                v = self.v_proj(h)

            else:
                token = h[:, -1:, :]
                k = torch.cat([self.k_cache, self.k_proj(token)], dim=1)
                v = torch.cat([self.v_cache, self.v_proj(token)], dim=1)
                q = torch.cat(
                    [torch.zeros_like(h[:, :-1, :]).to(h), self.q_proj(token)], dim=1
                )

            self.k_cache = k
            self.v_cache = v

        else:
            q = self.q_proj(h)
            k = self.k_proj(h)
            v = self.v_proj(h)

        q = q.view(bsz, n_seq, self.n_head, self.head_dim)
        k = k.view(bsz, n_seq, self.n_kv_head, self.head_dim)
        v = v.view(bsz, n_seq, self.n_kv_head, self.head_dim)

        q, k = self.rotary_emb(q, k)

        k = repeat_kv(k, self.n_kv_groups)

        v = repeat_kv(v, self.n_kv_groups)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash_attn:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0,
                is_causal=self.is_causal,
            )

        else:
            mask = torch.full((1, 1, n_seq, n_seq), -float("inf")).to(h)
            mask = torch.triu(mask, diagonal=1)
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores += mask[:, :, :n_seq, :n_seq]
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            scores = self.attn_dropout(scores)
            out = torch.matmul(scores, v)

        out = out.transpose(1, 2).contiguous().view(bsz, n_seq, self.hidden_size)
        out = self.o_proj(out)
        out = self.res_dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.bias = config.mlp_bias
        self.g_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=self.bias
        )
        self.u_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=self.bias
        )
        self.d_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=self.bias
        )

    def forward(self, h):
        return self.d_proj(F.silu(self.g_proj(h)) * self.u_proj(h))


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.attn = Attention(config)
        self.mlp = MLP(config)

        self.pre_norm = RMSNorm(self.hidden_size)
        self.post_norm = RMSNorm(self.hidden_size)

    def forward(self, h, use_kv_cache=False):
        res = h
        h = self.pre_norm(h)
        h = self.attn(h, use_kv_cache)
        h = res + h
        res = h
        h = self.post_norm(h)
        h = self.mlp(h)
        h = res + h

        return h


class MiniLLAMA(PreTrainedModel):
    config_class = Config

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.n_layers)]
        )

        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight
        self.loss = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

    def free_cache(self):
        for layer in self.layers:
            layer.attn.k_cache = None
            layer.attn.v_cache = None

    def forward(self, inp_ids, labels, use_kv_cache=False):
        bsz, n_seq = inp_ids.shape
        h = self.token_emb(inp_ids)
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h, use_kv_cache)

        h = self.norm(h)

        if labels is not None:
            logits = self.lm_head(h)
            logits = logits.view(bsz * n_seq, self.vocab_size)
            labels = labels.view(-1)  # (bsz * n_seq,)
            self.loss = F.cross_entropy(logits, labels, ignore_index=0)

        else:
            logits = self.lm_head(h[:, [-1], :])
            self.loss = None

        return CausalLMOutputWithPast(
            loss=self.loss,
            logits=logits,
        )

    def generate(
        self,
        inp,
        eos,
        max_new_tokens=100,
        temp=0.7,
        top_k=None,
        stream=True,
        rep_pen=1,
        use_kv_cache=True,
    ):
        inp_ids = inp["input_ids"]
        labels = inp["labels"]
        bsz, n_seq = inp_ids.shape

        while inp_ids.shape[1] < max_new_tokens - 1:
            out = self.forward(inp_ids, labels, use_kv_cache)
            logits = out.logits[:, -1, :]

            if rep_pen is not None and rep_pen != 1:
                for token in set(inp_ids.flatten().tolist()):
                    logits[:, token] /= rep_pen

            if temp == 0:
                _, next_token = logits.topk(1, dim=-1)
            else:
                logits = logits / temp

                if top_k is not None:
                    thres, _ = logits.topk(min(top_k, logits.size(-1)), dim=-1)
                    logits[logits < thres[:, [-1]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            if next_token == eos:
                break

            inp_ids = torch.cat([inp_ids, next_token], dim=1)
            if stream:
                yield inp_ids[:, n_seq:]

        if not stream:
            yield inp_ids[:, n_seq:]
