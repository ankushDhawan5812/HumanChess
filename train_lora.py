import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from fen_conv import convert_to_token
from model_v2 import load_base_model
from lora import LoRALayer
import re
from tqdm import tqdm


def add_lora_layers(model, rank, alpha):
    #use regex to store attention layer names
    target_patterns = [r'.*attn\.q_proj$', r'.*attn\.k_proj$', r'.*attn\.v_proj$', r'.*attn\.out_proj$', r'.*out_proj$']
    
    patterns = [re.compile(p) for p in target_patterns]
    
    lora_layers = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(p.match(name) for p in patterns):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            
            lora_module = LoRALayer(module, rank, alpha)
            setattr(parent, child_name, lora_module)
            lora_layers[name] = lora_module
            print(f"Added LoRA to {name}")
    
    return lora_layers

class ChessDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        fen = self.data.iloc[idx, 0]
        tokens = convert_to_token(fen)
        tokens = torch.from_numpy(tokens).long()
        action_dist = self.data.iloc[idx, 1:].values.astype(np.float32)
        action_dist = torch.tensor(action_dist, dtype=torch.float32)
        
        return tokens, action_dist

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

def train_lora(model, train_loader, val_loader, epochs, lr, device='cuda'):
    model = model.to(device)

    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend([module.lora_a, module.lora_b])
    
    optimizer = optim.AdamW(lora_params, lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in tqdm(train_loader, desc="Training"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            logits = model(batch_X)
            logp = F.log_softmax(logits, dim=1)
            loss = criterion(logp, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Train Loss: {avg_train_loss:.6f}")
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    logits = model(batch_X)
                    logp = F.log_softmax(logits, dim=1)
                    loss = criterion(logp, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.6f}")
    
    return model

def save_lora_weights(model, path):
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state_dict[f"{name}.lora_a"] = module.lora_a.data
            lora_state_dict[f"{name}.lora_b"] = module.lora_b.data
    
    torch.save(lora_state_dict, path)
    print(f"Saved LoRA weights to {path}")

def load_lora_weights(model, path):
    lora_state_dict = torch.load(path)
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            if f"{name}.lora_a" in lora_state_dict:
                module.lora_a.data.copy_(lora_state_dict[f"{name}.lora_a"])
            if f"{name}.lora_b" in lora_state_dict:
                module.lora_b.data.copy_(lora_state_dict[f"{name}.lora_b"])
    
    print(f"Loaded LoRA weights from {path}")

def main():
    model, device = load_base_model()
    add_lora_to_model(model, rank=8, alpha=32)

    input_dim = model.out_proj.in_features
    model.out_proj = nn.Linear(input_dim, 1968)  # 1968 possible chess moves
    nn.init.normal_(model.out_proj.weight, std=0.02)
    nn.init.zeros_(model.out_proj.bias)

    # need to figure out path
    # dataset = ChessDataset("noisy_fen_dataset.csv")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    model = train_lora(model, train_loader, val_loader, epochs=10, lr=1e-3, device=device)
    
    save_lora_weights(model, f"lora_weights_{elo_range}.pt")

if __name__ == "__main__":
    main()