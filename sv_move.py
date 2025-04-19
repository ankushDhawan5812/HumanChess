import torch
from infra import TransformerDecoder   
from fen_conv import NUM_BUCKETS, BUCKET_MIDPOINTS, convert_to_token   

action_size = 31        
seq_len     = 77
d_model     = 256
num_layers  = 8
num_heads   = 8
d_ff        = d_model * 4
dropout     = 0.1
output_size = 128       

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

model = TransformerDecoder(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    dropout=dropout,
    action_size=action_size,
    seq_len=seq_len,
    output_size=output_size,
    use_causal_mask=False,
    apply_qk_layernorm=False,
).to(device)

# load cur params
state = torch.load("models/best_model.pth", map_location=device)
model.load_state_dict(state)      # strict=True by default

model.eval()
buckets = NUM_BUCKETS
midpoints = BUCKET_MIDPOINTS

def rank_moves(fen):
    board = chess.Board(fen) # set up current FEN

    for move in board.legal_moves:
        copy_board = board.copy()
        copy_board.push(move)
        new_fen = copy_board.fen()
        tokens = convert_to_token(new_fen)
        tokens = torch.from_numpy(tokens).long().unsqueeze(0).to(device)
    
        with torch.no_grad():
            logp = model(tokens).squeeze(0)
            p = logp.exp()
            win_p = float((p * BUCKET_MIDPOINTS.to(p)).sum())
        
        results.append((move.uci(), win_p))
    
    results.sort(key=lambda x: x[1]) 
    return results # these are ascending so we could essentially minimize parents value

