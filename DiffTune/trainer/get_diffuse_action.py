import torch
import sys
sys.path.append('..')
import chess
import numpy as np
from data.fen_conv_diff import convert_to_token, move_to_id, id_to_move
from data.sv_move import return_next_move
from data.model_v2 import load_base_model
from collections import defaultdict
from model.lora_model import LoraDiffusionModel
import torch.nn.functional as F


transformer_model, device = load_base_model(model_path="/home/ankush/repos/chess_train/HumanChess/DiffTune/trainer/model_epoch_7.pth")
transformer_model.eval()
fen_str = "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1"
board = chess.Board(fen_str) # set up current FEN

first_state = convert_to_token(fen_str)
first_state = torch.from_numpy(first_state).long().unsqueeze(0).to(device)
print(len(first_state[0]))

if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

def denoise_input(fen_state, elo_idx):
    s_t = convert_to_token(fen_state)
    s_t = torch.from_numpy(s_t).long().unsqueeze(0).to(device)
    x_t = torch.randint(0, 1968, (1, 312), device=device)
    iterations = 20
    

    for t in range(iterations, 0, -1):
        with torch.no_grad():
            logits = transformer_model(s_t, x_t, elo_idx)
        
        logp = F.log_softmax(logits, dim=-1)
        current_confidence = logp.gather(dim=-1, index=x_t.unsqueeze(-1)).squeeze(1)

        thresh = 100 * (t - 1)/iterations
        
        update_mask = torch.ones_like(current_confidence, dtype=torch.bool)
        if thresh > 0:
            percentile_val = torch.quantile(current_confidence.flatten(), thresh / 100)
            update_mask = current_confidence <= percentile_val
        
        x_t = torch.where(update_mask, torch.argmax(logits, dim=-1), x_t)
        
        if t > 1 and update_mask.sum().item() == 0:
            break
    
    return x_t






        
