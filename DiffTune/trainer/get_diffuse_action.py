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

def denoise_input(fen_state, elo_idx, device=None, lora_model=None):
    # transformer_model = lora_model.lora_transformer
    s_t = convert_to_token(fen_state)
    s_t = torch.from_numpy(s_t).long().unsqueeze(0).to(device)
    x_t = torch.randint(0, 1968, (1, 312), device=device)
    iterations = 20
    
    for t in range(iterations, 0, -1):
        with torch.no_grad():
            logits = lora_model(s_t, x_t, elo_idx)
        
        logp = F.log_softmax(logits, dim=-1)
        current_confidence = logp.gather(dim=-1, index=x_t.unsqueeze(-1)).squeeze(1)

        thresh = 100 * (t - 1)/iterations
        
        update_mask = torch.ones_like(current_confidence, dtype=torch.bool)
        if thresh > 0:
            percentile_val = torch.quantile(current_confidence.flatten(), thresh / 100)
            update_mask = current_confidence <= percentile_val
        
        update_mask = update_mask.squeeze(-1)
        x_t = torch.where(update_mask, torch.argmax(logits, dim=-1), x_t)
        
        if t > 1 and update_mask.sum().item() == 0:
            break
    
    return x_t

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

    total_t = 10000
    betas = [(i + 1) / total_t for i in range(total_t)]
    num_moves = 1968
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
    lora_weights_path = "/home/ankush/repos/chess_train/HumanChess/DiffTune/trainer/old_models/1million_samples_1e5/model_lora_epoch_10.pt"

    try:
        lora_state_dict = torch.load(lora_weights_path, map_location=device)
        lora_model.lora_transformer.load_state_dict(lora_state_dict, strict=False)
        print(f"Successfully loaded LoRA weights from {lora_weights_path}")
    except FileNotFoundError:
        print(f"LoRA weights file not found at {lora_weights_path}")
        print("Using base model without LoRA fine-tuning")
    except Exception as e:
        print(f"Error loading LoRA weights: {e}")
        print("Using base model without LoRA fine-tuning")


    fen_str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = chess.Board(fen_str) # set up current FEN

    first_state = convert_to_token(fen_str)
    first_state = torch.from_numpy(first_state).long().unsqueeze(0).to(device)
    print(len(first_state[0]))

    elo_embed_float = 1.0  # Example elo index, adjust as needed
    output = denoise_input(fen_str, elo_idx=torch.tensor([elo_embed_float]).to(device), device=device, lora_model=lora_model)
    print("Output tokens:", output)
    print(id_to_move(output[0, 0].item()))
if __name__ == "__main__":
    main()