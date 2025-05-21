import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from infra_2d import TransformerDecoder2D

def load_base_model(model_path="/content/model_epoch_4.pth"):
    """
    Load the base TransformerDecoder2D model
    """
    # Define model architecture parameters
    action_size = 31        
    seq_len     = 77
    d_model     = 256
    num_layers  = 8
    num_heads   = 8
    d_ff        = d_model * 4
    dropout     = 0.1
    output_size = 128 
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create model
    model = TransformerDecoder2D(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        action_size=action_size,            
        seq_len=seq_len,
        output_size=output_size,
        max_distance=8,         
        use_causal_mask=False    
    ).to(device)
    
    # Load weights
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    print(f"Successfully loaded model from {model_path}")
    
    return model, device

if __name__ == "__main__":
    model, device = load_base_model()
    print("Model successfully loaded and ready for use")
