import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# by hand conversion of deep mind transformer, might be wrong lowkey

class InputEmbeddings2D(nn.Module):
    def __init__(self, d_model, action_size):
        super().__init__()
        self.d_model = d_model
        self.action_size = action_size
        self.embedding = nn.Embedding(action_size, d_model)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape becomes: (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # easy to get afterward
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # this should not get updated w/ grads
        x = x + self.pe[:, :x.size(1), :] 
        return self.dropout(x)

class LayerNormalization2D(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (, d_model)
        #standard layer norm formula
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# basically just MLP block defined in the attention all you need paper
class FeedForwardBlock2D(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x)))) # changed to GELU here, might help the model

# multihead attention implemented as deepmind
class MultiHeadAttention2D(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, apply_qk_layernorm=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.apply_qk_layernorm = apply_qk_layernorm
        
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Modified approach: use a more standard attention but with a learned bias
        # This maintains the 2D spatial information while being much more efficient
        self.rel_bias = nn.Parameter(torch.zeros(1, num_heads, 78, 78))
        nn.init.xavier_uniform_(self.rel_bias)
    
    def forward(self, query, key, value, token_coords, mask=None):
        batch_size, seq_len, _ = query.size()
        
        # Project inputs    
        Q = self.linear_q(query)
        K = self.linear_k(key)
        V = self.linear_v(value)
        
        if self.apply_qk_layernorm:
            Q = self.layer_norm(Q)
            K = self.layer_norm(K)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Only use what we need for the current sequence length
        rel_bias = self.rel_bias[:, :, :seq_len, :seq_len]
        attn_scores = attn_scores + rel_bias
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.linear_out(context)

class TransformerDecoderBlock2D(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, use_causal_mask=True, apply_qk_layernorm=False):
        super().__init__()
        self.use_causal_mask = use_causal_mask
        self.self_attn = MultiHeadAttention2D(d_model, num_heads, dropout, apply_qk_layernorm)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardBlock2D(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, token_coords, mask=None):
        attn_output = self.self_attn(x, x, x, token_coords, mask)
        x = self.norm1(x + self.dropout(attn_output)) # added in the skip connection here, basically following attn all you need paper to the t
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output)) # another skip connect, tried switching order of dropout and norm to see if it makes a difference
        return x

class TransformerDecoder2D(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float, action_size: int,
                 seq_len: int, output_size: int = None, use_causal_mask: bool = True, apply_qk_layernorm: bool = False):
        super().__init__()
        self.pool_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.input_emb = InputEmbeddings2D(d_model, action_size)

        self.pos_enc = PositionalEncoding2D(d_model, seq_len + 1, dropout)

        coords = []
        coords.append([-1, -1])  # pool
        coords.append([-1, -1])  # side‐to‐move
        for r in range(8):
            for c in range(8):
                coords.append([r, c])
        coords += [[-1, -1]] * (seq_len + 1 - len(coords))
        self.register_buffer("token_coords", torch.tensor(coords, dtype=torch.long))

        self.layers = nn.ModuleList([
            TransformerDecoderBlock2D(d_model, num_heads, d_ff, dropout, use_causal_mask, apply_qk_layernorm)
            for _ in range(num_layers)
        ])
        self.output_size = output_size if output_size is not None else action_size
        self.linear_out = nn.Linear(d_model, self.output_size)
    
    def forward(self, x):
        # x: (batch_size, seq_len) tokens
        x_tok = self.input_emb(x) # (batch_size, seq_len, d_model)

        batch_size = x_tok.shape[0]
        # Expand the cls_token so that it is added to every input in the batch.
        pool_tokens = self.pool_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        
        # Concatenate the cls_token with the embedded tokens along sequence dimension.
        x = torch.cat((pool_tokens, x_tok), dim=1)
        x = self.pos_enc(x)
        
        mask = None
        """if self.layers[0].use_causal_mask:
            batch_size, seq_len, _ = x.size()
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)"""
        
        #run through the transformer blocks
        for layer in self.layers:
            x = layer(x, self.token_coords, mask)
        
        """logits = self.linear_out(x)  # (batch_size, seq_len, output_size)
        # logits = torch.log_softmax(logits, dim = -1)
        return logits """ # need to normalize predictions into probability classes

        pooled_out = x[:, 0, :]  # (batch_size, d_model)
        logits = self.linear_out(pooled_out)  # (batch_size, output_size)
        # slight error here where I was pre applying the softmax, which we shouldnt do for cross entropy loss
        return logits

if __name__ == "__main__":
    # Hyperparameters
    action_size = 31
    seq_len = 77
    d_model = 256
    num_layers = 4
    num_heads = 8
    d_ff = d_model * 4
    dropout = 0.1
    output_size = 1
    
    model = TransformerDecoder2D(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        action_size=action_size,
        seq_len=seq_len,
        output_size=output_size,
        use_causal_mask=False,        
        apply_qk_layernorm=False     
    )

    print("model built w/ params: ")
    
    print(f"""Action Size = {action_size}
              Sequence Length = {seq_len}
              Model Input Dim = {d_model}
              N Transformer Block Layers = {num_layers}
              Number of Heads = {num_heads}
              Hidden Dim = {d_ff}
              Dropout = {dropout}
              Model Output Dim = {output_size}
              Masking? = {False}
              Key Query Computer Layer Norm? = {False}""")
