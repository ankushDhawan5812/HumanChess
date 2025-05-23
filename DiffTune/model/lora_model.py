import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel
import math
from model.infra_2d_diff import TransformerDecoder2D

class LoraDiffusionModel(nn.Module):
    def __init__(self, base_model_path, elo_min, elo_max, bucket_size, betas, num_moves):
        super().__init__()

        self.piece_vocab = 31
        self.transformer = TransformerDecoder2D(
            num_layers=8,
            d_model=256,
            num_heads=8,
            d_ff=1024,
            dropout=0.1,
            action_size=31,
            seq_len=77,
            max_distance=8,
            use_causal_mask=False,
            output_size=num_moves,
        )

        self.d_model = 256
        self.embed_peices = nn.Embedding(self.piece_vocab, self.d_model)
        self.embed_moves = nn.Embedding(num_moves, self.d_model)

        pretrained_model = torch.load(base_model_path, map_location="cpu")
        filtered = {
            k: v for k, v in pretrained_model.items()
            if not k.startswith("out_proj")      
            and not k.startswith("input_emb")     
        }
        
        self.transformer.load_state_dict(filtered, strict=False)

        with torch.no_grad():
            self.embed_peices.weight.copy_(self.transformer.input_emb.weight)

        for p in self.transformer.parameters():
            p.requires_grad = False
        
        self.embed_peices.weight.requires_grad = False

        lora_cfg = LoraConfig(
            task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.1,
            target_modules=["q_proj","k_proj","v_proj","o_proj"]
        )
        self.lora_transformer = get_peft_model(self.transformer, lora_cfg)

        self.elo_min = elo_min
        self.bucket_size = bucket_size
        self.num_buckets = (elo_max-elo_min)//bucket_size + 1
        self.elo_buckets = nn.Embedding(self.num_buckets, self.d_model)
        self.move_head  = nn.Linear(self.d_model, num_moves) # mark this to look into, might need two output heads for states and actions

    def forward(self, s_tokens, x_t, elo_idx_float, t):
        batch_size, state_size = s_tokens.shape
        max_length = x_t.shape[1]
        d_model = self.d_model 
        s_emb = self.embed_peices(s_tokens) * math.sqrt(d_model)  
        pool  = self.lora_transformer.pool.expand(batch_size, -1, -1)                     
        x = torch.cat([pool, s_emb], dim=1)
        x = self.lora_transformer.pos_enc(x)                                                                

        idxf = elo_idx_float.clamp(0, self.num_buckets-1)
        lo, hi = idxf.floor().long(), idxf.ceil().long()                        
        w_hi = (idxf - lo.float()).unsqueeze(-1)                                 
        w_lo = 1.0 - w_hi                                                       
        emb_lo = self.elo_buckets(lo)                                            
        emb_hi = self.elo_buckets(hi)                                            
        elo_emb = emb_lo * w_lo + emb_hi * w_hi                                  
        elo_exp = elo_emb.unsqueeze(1).expand(-1, state_size+1, -1)                    

        x = x + elo_exp                                                
        f_emb = self.embed_moves(x_t) * math.sqrt(d_model)       
        x = torch.cat([x, f_emb], dim=1)     
 
        #NO MASKING TO BASICALLY LET IT SEE EVERYTHING AND REVERSE THE NOISE
        for layer in self.lora_transformer.layers:
            x = layer(x)                                                  

        future_hidden = x[:, 1+state_size :, :]                                           
        logits = self.move_head(future_hidden)                          
        return logits
