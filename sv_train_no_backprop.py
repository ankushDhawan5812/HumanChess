import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import IterableDataset

from fen_conv import convert_to_token, win_to_bucket, hl_gauss
from infra import TransformerDecoder
from infra_2d import TransformerDecoder2D
import argparse 
import os

def tensorize(data_df):
    tokens = [convert_to_token(fen) for fen in data_df['fen']]
    tokens = np.stack(tokens)
    winpct = data_df['win_percent'].to_numpy(dtype=np.float32)  # 0‑1
    buckets = hl_gauss(winpct)
    # Convert tokens to torch.long type
    X = torch.from_numpy(tokens).long()
    y = torch.from_numpy(buckets)
    return X, y

class ChessDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.X, self.y = tensorize(df)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class ChessCSVDataset(IterableDataset):
    def __init__(self, path, skiprows=0, nrows=None, chunksize=50000):
        self.path = path
        self.skiprows = skiprows
        self.nrows = nrows
        self.chunksize = chunksize

    def __iter__(self):
        for df_chunk in pd.read_csv(
            self.path, 
            skiprows=1 + self.skiprows, 
            nrows=self.nrows, 
            chunksize=self.chunksize,
            names = ['fen', 'win_percent'],
            header=None
        ):
            X_chunk, y_chunk = tensorize(df_chunk)
            for x, y in zip(X_chunk, y_chunk):
                yield x, y
    
    def __len__(self):
        return self.nrows
         

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="select model"
    )
    parser.add_argument(
        "--kind",
        choices=["1", "2"],
        required=True,
        help="1 = normal token position encodings |  2 = 2d chess board positional encodings",
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    args = parser.parse_args()
    
    # Gradient accumulation steps
    accum_steps = args.accum_steps
    
    # Split data into train and validation sets
    n_total = 530310000 / 4

    train_size = int(0.95 * n_total)
    val_size = n_total - train_size
    train_dataset = ChessCSVDataset('chessbench_state_train.csv', skiprows=0, nrows=train_size)
    val_dataset = ChessCSVDataset('chessbench_state_train.csv', skiprows=train_size, nrows=val_size)

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

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
    )

    if args.kind == "2":
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
       )
        
    model.to(device)

    # Load a previously saved model
    epoch_start = 7
    checkpoint_path = f'/home/ankush/repos/chess_train/HumanChess/models/model_epoch_{epoch_start}.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")

    # n_params to see
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_params = count_parameters(model)
    print(f"Total params: {total_params:,}")
    print(f"Evaluating model (no gradient updates)")

    # Loss function and optimizer
    criterion = nn.KLDivLoss(reduction='batchmean')
    # optimizer and scheduler not needed for evaluation, but keeping for compatibility
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Evaluation function (modified train_epoch)
    def evaluate_epoch(model, train_loader, criterion, device):
        model.eval()  # Set to eval mode
        total_loss = 0
        num_batches = 0

        total_batches = len(train_loader.dataset) // batch_size
        print(f"Total batches in this epoch: {total_batches}")
        counter = 0
        with torch.no_grad():  # Disable gradient computation entirely
            for batch_idx, (batch_X, batch_y) in enumerate(tqdm(train_loader, desc="Evaluating", total=total_batches)):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                num_batches += 1
                
                # Forward pass
                logits = model(batch_X)
                logp = F.log_softmax(logits, dim=1)
                loss = criterion(logp, batch_y)

                # if counter % 100 == 0:
                #     print(loss.item())
                
                # Track the loss
                total_loss += loss.item()
                
                counter += 1
                # Save checkpoint every 1,000,000 batches (optional)
                if batch_idx > 0 and batch_idx % 1000000 == 0:
                    print(f"Checkpoint at batch {batch_idx}")
                    # Uncomment if you want to save checkpoints during evaluation
                    # torch.save(model.state_dict(), f'models/checkpoint_batch.pth')
        
        return total_loss / num_batches
    
    # Validation function
    def validate(model, val_loader, criterion, device):
        model.eval()
        total_loss = 0

        total_batches = len(val_loader.dataset) // 64
        print(f"Total batches in validation: {total_batches}")

        batch_count = 0

        print("Skipping validation")
        return 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_loader, desc="Validation", total=total_batches):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                batch_count += 1
                
                outputs = model(batch_X)
                logp = F.log_softmax(outputs, dim=1)
                
                loss = criterion(logp, batch_y)
                total_loss += loss.item()
        
        return total_loss / batch_count

    # Evaluation loop (modified training loop)
    num_epochs = 12
    train_losses = []

    # Create directory for saving models if it doesn't exist
    os.makedirs('models', exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Evaluate instead of train
        train_loss = evaluate_epoch(model, train_loader, criterion, device)
        
        # Save train loss to a file
        with open('eval_loss.txt', 'a') as f:
            f.write(f"Epoch {epoch_start+epoch+1}: {train_loss:.4f}\n")
        
        # Optional: Save model state (commented out since we're not training)
        # print("saving model")
        # torch.save(model.state_dict(), f'models/model_epoch_{epoch_start+epoch+1}.pth')
        # print("model saved")
        
        train_losses.append(train_loss)
        
        print(f"Evaluation Loss: {train_loss:.4f}")
        
        # Scheduler step commented out since we're not training
        # scheduler.step(train_loss)

    # Plot evaluation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Evaluation Losses')
    plt.legend()
    plt.savefig('evaluation_loss.png')
    plt.close()