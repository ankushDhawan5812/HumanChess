import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbeddings2D(nn.Module):
    def __init__(self, d_model, action_size):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.embedding = nn.Embedding(action_size, d_model)

    def forward(self, x):
        return self.embedding(x) * self.scale

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class FeedForwardBlock2D(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class RelativeMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        token_coords: torch.Tensor,
        max_distance: int = 8,
        dropout: float = 0.1,
        use_causal_mask: bool = False
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_causal_mask = use_causal_mask
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        num_rel = (2 * max_distance + 1) ** 2
        self.rel_bias = nn.Parameter(torch.randn(num_rel, num_heads) * (max_distance ** -0.5))
        idx = self._precompute_rel_idx(token_coords, max_distance)
        self.register_buffer('rel_idx', idx)

    def _precompute_rel_idx(self, coords: torch.Tensor, max_distance: int):
        seq_len = coords.size(0)
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  
        diff = diff.clamp(-max_distance, max_distance) + max_distance
        idx = diff[..., 0] * (2 * max_distance + 1) + diff[..., 1]
        return idx.long()  

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.d_k).transpose(1,2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.d_k).transpose(1,2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.d_k).transpose(1,2)
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_k)  
        bias = self.rel_bias[self.rel_idx]              
        bias = bias.permute(2,0,1).unsqueeze(0)         
        scores = scores + bias
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=~mask if mask is not None else None,
                dropout_p=self.dropout.p,
                is_causal=self.use_causal_mask
            )
        else:
            w = F.softmax(scores, dim=-1)
            w = self.dropout(w)
            out = w @ v
        out = out.transpose(1,2).reshape(B, S, self.d_model)
        return self.out_proj(out)

class TransformerDecoderBlock2D(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, token_coords, max_distance=8, use_causal_mask=False):
        super().__init__()
        self.attn = RelativeMultiHeadAttention(d_model, num_heads, token_coords, max_distance, dropout, use_causal_mask)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardBlock2D(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class TransformerDecoder2D(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        action_size: int,
        seq_len: int,
        output_size: int = None,
        max_distance: int = 8,
        use_causal_mask: bool = False
    ):
        super().__init__()
        # embeddings + coords
        self.pool = nn.Parameter(torch.zeros(1,1,d_model))
        self.input_emb = nn.Embedding(action_size, d_model)
        coords = [[-1,-1],[-1,-1]] + [[r,c] for r in range(8) for c in range(8)]
        coords += [[-1,-1]] * (seq_len + 1 - len(coords))
        token_coords = torch.tensor(coords, dtype=torch.long)
        self.register_buffer('token_coords', token_coords)
        # positional encoding
        self.pos_enc = PositionalEncoding2D(d_model, seq_len+1, dropout)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock2D(d_model, num_heads, d_ff, dropout, token_coords, max_distance, use_causal_mask)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, output_size or action_size)

    def forward(self, x, padding_mask=None):
        B, L = x.shape
        emb = self.input_emb(x)
        pool = self.pool.expand(B,-1,-1)
        x = torch.cat([pool, emb], dim=1)
        x = self.pos_enc(x)
        mask = None
        if padding_mask is not None:
            pad = padding_mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,L]
            mask = torch.cat([torch.ones(B,1,1,1,device=pad.device), pad], dim=-1)
        # apply layers
        for layer in self.layers:
            x = layer(x, mask)
        # predict from pool
        return self.out_proj(x[:,0])

if __name__ == '__main__':
    action_size, seq_len, d_model = 31, 77, 256
    num_layers, num_heads, d_ff, dropout = 4, 8, 256*4, 0.1
    model = TransformerDecoder2D(num_layers, d_model, num_heads, d_ff, dropout, action_size, seq_len)
    x = torch.randint(0, action_size, (2, seq_len))
    print(model(x).shape)
