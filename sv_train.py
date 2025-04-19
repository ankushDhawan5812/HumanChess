import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from fen_conv import convert_to_token, win_to_bucket
from infra import TransformerDecoder

def tensorize(data_df):
    tokens = [convert_to_token(fen) for fen in data_df['fen']]
    tokens = np.stack(tokens)
    winpct = data_df['win_percent'].to_numpy(dtype=np.float32)  # 0‑1
    bucket = win_to_bucket(winpct).astype(np.int64)
    # Convert tokens to torch.long type
    X = torch.from_numpy(tokens).long()
    y = torch.from_numpy(bucket)
    return X, y

class ChessDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.X, self.y = tensorize(df)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Split data into train and validation sets
dataset = ChessDataset('chessbench_state.csv')

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model hyperparameters
action_size = 31  
seq_len = 77      
d_model = 256
num_layers = 8
num_heads = 8
d_ff = d_model * 4
dropout = 0.1
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
    apply_qk_layernorm=False
).to(device)

# n_params to see
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Total params: {total_params:,}")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(reduction='mean') #need to add HL guass smoothing later
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# use this later
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in tqdm(train_loader, desc="Training"):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward pass
        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        
        # grad clipping 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_X, batch_y in tqdm(val_loader, desc="Validation"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            outputs = outputs.squeeze(-1)
            
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

# Training loop
num_epochs = 50
best_val_loss = float('inf')
train_losses = []
val_losses = []

# Create directory for saving models if it doesn't exist
os.makedirs('models', exist_ok=True)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Update learning rate
    # scheduler.step(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/best_model.pth')
        print(f"Model saved with validation loss: {val_loss:.4f}")

# Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.savefig('training_loss.png')
plt.close()

   

