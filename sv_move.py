import torch
from infra import TransformerDecoder
from infra_2d import TransformerDecoder2D
from fen_conv import NUM_BUCKETS, BUCKET_MIDPOINTS, convert_to_token   
import chess

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

model = TransformerDecoder2D(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    dropout=dropout,
    action_size=action_size,            
    seq_len=seq_len,
    output_size=output_size,
    max_distance=8,         # how far apart (in grid steps) you want to model relative bias
    use_causal_mask=False    # keep as False unless you need an autoregressive mask
).to(device)

# load cur params
state = torch.load("models/model_epoch_2.pth", map_location=device)
model.eval()
model.load_state_dict(state)      # strict=True by default


buckets = NUM_BUCKETS
midpoints = BUCKET_MIDPOINTS

def return_next_move(fen):
    board = chess.Board(fen) # set up current FEN

    results = []
    for move in board.legal_moves:
        copy_board = board.copy()
        copy_board.push(move)
        new_fen = copy_board.fen()
        tokens = convert_to_token(new_fen) #this new fen will be the state of the opponent, so we want to choose the lowest score here
        tokens = torch.from_numpy(tokens).long().unsqueeze(0).to(device)
    
        with torch.no_grad():
            logits = model(tokens)
            probs = torch.softmax(logits, dim = -1)
            win_p = float((probs * torch.from_numpy(BUCKET_MIDPOINTS).to(device)).sum())
        
        results.append((move.uci(), win_p))
    
    results.sort(key=lambda x: x[1]) 
    return results
    

#next_move = return_next_move("6k1/8/5PK1/8/8/8/8/8 w - - 0 1")
#print(next_move)


 
