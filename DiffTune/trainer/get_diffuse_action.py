import torch
import sys
sys.path.append('..')
import chess
import numpy as np
from data.fen_conv_diff import convert_to_token, move_to_id, id_to_move, MOVE_TO_ID
from data.sv_move import return_next_move
from data.model_v2 import load_base_model
from collections import defaultdict
from model.lora_model import LoraDiffusionModel
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def mask_illegal_logits(logits_slice, board, move_offset=31):
    """
    logits_slice : Tensor [H, V] (single position)
    board        : python-chess Board
    Returns a masked-in-place tensor (no new allocation).
    """
    logits_slice[:, :move_offset] -= 1e9     

    for m in board.legal_moves:
        u = m.uci()
        if u in MOVE_TO_ID:                   
            gid = MOVE_TO_ID[u] + move_offset
            logits_slice[:, gid] += 1e9 

def multinomial_diffuse_generate(
        model,
        s_tokens,           
        elo_float,          
        horizon,            
        T=20,               
        vocab_size=2000,   
        move_offset=31,     
        device=None, 
        board=None):

    if device is None:
        device = s_tokens.device
    B = s_tokens.size(0)

    x_t = torch.randint(0, vocab_size, (B, horizon), device=device)

    frozen = torch.zeros_like(x_t, dtype=torch.bool)

    for t in range(T, 0, -1):                          
        t_tensor = torch.full((B,), t, dtype=torch.long, device=device)

        logits = model(s_tokens, x_t, elo_float, t_tensor)  
        # mask_illegal_logits(logits[0], board)
        probs  = torch.softmax(logits[0, 0], dim=-1)
        entropy = -(probs * probs.log()).sum().item()
        print(f"entropy on 1st future slot: {entropy:.2f}")   

        probs = F.softmax(logits, dim=-1)     
                   

        x_hat = torch.multinomial(
                    probs.view(-1, vocab_size), 1
                ).view(B, horizon)
        conf = torch.max(probs, dim=-1).values            

 
        keep_ratio = 0.5 * t / T
        for b in range(B):
           
            num_unfrozen   = (~frozen[b]).sum().item()
            num_to_freeze  = int(num_unfrozen * keep_ratio)

            if num_to_freeze == 0:
                continue

            idx_unfrozen = torch.where(~frozen[b])[0]
            conf_sorted  = conf[b, idx_unfrozen].argsort(descending=True)
            top_k = idx_unfrozen[conf_sorted[:num_to_freeze]]
            x_t[b, top_k]   = x_hat[b, top_k]
            frozen[b, top_k] = True

        if t > 1:  
            num_unfrozen_total = (~frozen).sum().item()
            if num_unfrozen_total > 0:
                x_t[~frozen] = torch.randint(0, vocab_size, (num_unfrozen_total,), device=device)

    first_move_global = x_t[:, 0]    
    return first_move_global

def main():
    device = None
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    total_t = 20
    betas = [(i + 1) / (total_t * 10) for i in range(total_t)]
    num_moves = 2000
    elo_min, elo_max, bucket_size = 1200, 1800, 100

    lora_model = LoraDiffusionModel(
        base_model_path="/home/ankush/repos/chess_train/HumanChess/DiffTune/trainer/model_epoch_7.pth",
        elo_min=elo_min, 
        elo_max=elo_max, 
        bucket_size=bucket_size,
        betas=betas, 
        num_moves=num_moves
    ).to(device)

    # Load the trained LoRA weights
    # lora_weights_path = "/home/ankush/repos/chess_train/HumanChess/DiffTune/trainer/lora_adapters.pt"
    # Alternative: use the specific epoch model if you prefer
    #lora_weights_path = "/home/ankush/repos/chess_train/HumanChess/DiffTune/trainer/old_models/1million_samples_1e5/model_lora_epoch_10.pt"
    lora_model.eval()                       
    ckpt = torch.load("full_model_epoch_10.pth", map_location=device)
    for k in list(ckpt.keys()):               # iterate over keys once
        if any(tag in k for tag in ["pos_enc.pe", "token_coords", "rel_idx"]):
            del ckpt[k]                       # drop the shape-dependent tensors
    lora_model.load_state_dict(ckpt, strict=False) 
    load_info = lora_model.load_state_dict(ckpt, strict=False)
    print("missing :", load_info.missing_keys)     # should be ONLY the buffers you deleted
    print("unexpected:", load_info.unexpected_keys)# should be []
    assert not [k for k in load_info.missing_keys if ".lora_" in k or "input_emb" in k], "critical trainable weights missing!"

    with torch.no_grad():
        W = lora_model.transformer.input_emb.weight
        print("avg ‖piece rows‖ :", W[:31].norm(dim=1).mean().item())
        print("avg ‖move  rows‖ :", W[31:].norm(dim=1).mean().item())
    fen_str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    #fen_str = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"
    board = chess.Board(fen_str) # set up current FEN

    first_state = convert_to_token(fen_str)
    first_state = torch.from_numpy(first_state).long().unsqueeze(0).to(device)
    print(len(first_state[0]))


    elo_embed_float = 3.0  # Example elo index, adjust as needed
    output = multinomial_diffuse_generate(model=lora_model, s_tokens=first_state, elo_float=torch.tensor([elo_embed_float]).to(device), horizon=312, T=20, vocab_size=2000, move_offset=31, device=device, board=board)
    print("Output tokens:", output)
    uci_move = id_to_move(output.item() - 31)
    print("DiffuSearch move:", uci_move)
if __name__ == "__main__":
    main()