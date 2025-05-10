import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# Import PEFT library
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    PeftConfig
)

class ChessMoveDataset(Dataset):
    """Dataset for chess games with player moves and ELO ratings"""
    
    def __init__(self, csv_path):
        """
        Args:
            csv_path: Path to CSV file with format: fen,move,player_elo
            where move is in UCI format (e.g., 'e2e4')
        """
        self.data = pd.read_csv(csv_path)
        print(f"Loaded dataset with {len(self.data)} positions")
        
        # Create move vocabulary (from UCI format to indices)
        self.move_to_idx = {}
        self.idx_to_move = {}
        
        # Create a vocabulary for all possible moves
        # UCI format moves:
        all_moves = []
        for r1 in range(8):
            for c1 in range(8):
                for r2 in range(8):
                    for c2 in range(8):
                        # Convert to UCI format (e.g., 'e2e4')
                        file1 = chr(ord('a') + c1)
                        rank1 = str(r1 + 1)
                        file2 = chr(ord('a') + c2)
                        rank2 = str(r2 + 1)
                        move = f"{file1}{rank1}{file2}{rank2}"
                        all_moves.append(move)
                        
                        # Add promotion variants
                        if (r1 == 6 and r2 == 7) or (r1 == 1 and r2 == 0):  # Pawn promotion
                            for piece in ['q', 'r', 'b', 'n']:  # Queen, Rook, Bishop, Knight
                                promotion_move = f"{move}{piece}"
                                all_moves.append(promotion_move)
        
        # Create mapping
        for i, move in enumerate(all_moves):
            self.move_to_idx[move] = i
            self.idx_to_move[i] = move
            
        print(f"Created move vocabulary with {len(self.move_to_idx)} moves")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            board_tokens: Tokenized board state
            move_idx: Index of the move played
            elo: Player ELO rating
        """
        item = self.data.iloc[idx]
        
        # Convert FEN to tokens
        board_tokens = self.tokenize_fen(item['fen'])
        
        # Convert move to index
        move = item['move']
        move_idx = self.move_to_idx.get(move, 0)  # Default to 0 if not found
        
        # Get ELO rating
        elo = item['player_elo']
        
        return torch.tensor(board_tokens, dtype=torch.long), move_idx, elo
    
    def tokenize_fen(self, fen_string, seq_len=77):
        """
        Convert FEN string to tokens using a more complete approach
        
        Args:
            fen_string: Standard FEN notation of a chess position
            seq_len: Expected length of the token sequence (default: 77)
            
        Returns:
            List of token indices of length seq_len
        """
        # Prepare character to index mapping
        idx = {
            'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,  # Black pieces
            'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12,  # White pieces
            'w': 13, 'K': 14, 'Q': 15, 'k': 16, 'q': 17,  # Side to move and castling rights
            'a': 18, 'b': 19, 'c': 20, 'd': 21, 'e': 22, 'f': 23, 'g': 24, 'h': 25,  # File names
            '1': 26, '2': 27, '3': 28, '4': 29, '5': 30, '6': 31, '7': 32, '8': 33,  # Rank names and counts
            '9': 34, '0': 35, # For move counts > 9
            '.': 0  # Empty square and padding
        }
        empty_set = {'1', '2', '3', '4', '5', '6', '7', '8'}
        
        # Split FEN components
        board, side, castling, en_passant, halfmoves_last, fullmoves = fen_string.split(' ')
        board = board.replace('/', '')
        board = side + board  # Prepend side to move
        
        token = []
        
        # Process board state and side to move
        for char in board:
            if char in empty_set:
                token.extend(int(char) * [idx['.']])
            else:
                token.append(idx[char])
        
        if castling == '-':
            token.extend(4 * [idx['.']])
        else:
            for char in castling:
                token.append(idx[char])
            if len(castling) < 4:
                token.extend((4 - len(castling)) * [idx['.']])
        
        if en_passant == '-':
            token.extend(2 * [idx['.']])
        else:
            for char in en_passant:
                token.append(idx[char])
        
        halfmoves_last += '.' * (3 - len(halfmoves_last))
        for char in halfmoves_last:
            if char == '.':
                token.append(idx['.'])
            else:
                token.append(idx[char])
        
        fullmoves += '.' * (3 - len(fullmoves))
        for char in fullmoves:
            if char == '.':
                token.append(idx['.'])
            else:
                token.append(idx[char])
        
        assert len(token) == seq_len, f"Token length {len(token)} doesn't match expected length {seq_len}"
        
        return token
    
    def filter_by_elo(self, min_elo, max_elo):
        """
        Returns indices of positions where the player's ELO is within the specified range
        """
        indices = self.data[(self.data['player_elo'] >= min_elo) & 
                           (self.data['player_elo'] < max_elo)].index.tolist()
        return indices


# Custom model head for move prediction
class MoveClassificationHead(nn.Module):
    """
    Classification head for predicting moves
    Replaces the win-percentage prediction head in your model
    """
    def __init__(self, input_dim, num_moves):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_moves)
        
    def forward(self, x):
        return self.linear(x)


def prepare_model_for_move_prediction(base_model, num_moves):
    """
    Adapt your TransformerDecoder2D model for move prediction
    
    Args:
        base_model: Your TransformerDecoder2D model trained for win prediction
        num_moves: Number of possible moves in your move vocabulary
        
    Returns:
        Modified model with a move prediction head
    """
    # Replace the output projection with a move prediction head
    d_model = base_model.layers[0].attn.d_model
    base_model.out_proj = MoveClassificationHead(d_model, num_moves)
    
    return base_model


def prepare_model_for_peft(model):
    """
    Prepare a PyTorch model for PEFT adaptation
    by marking certain layers for LoRA
    """
    # Freeze the base model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Define target modules by name - adjust based on your model's structure
    target_modules = []
    
    # Find all attention projection layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'out_proj']):
            target_modules.append(name)
    
    print(f"Found {len(target_modules)} attention layers to adapt with LoRA")
    return model, target_modules


def train_elo_move_adapter(
    base_model,
    dataset,
    min_elo,
    max_elo,
    output_dir,
    batch_size=64,
    num_epochs=5,
    learning_rate=1e-4,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.1,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train a LoRA adapter to mimic moves of players in a specific ELO range
    """
    print(f"Training move adapter for ELO range {min_elo}-{max_elo}")
    
    # 1. Filter dataset by ELO range
    indices = dataset.filter_by_elo(min_elo, max_elo)
    
    if len(indices) == 0:
        raise ValueError(f"No positions found in ELO range {min_elo}-{max_elo}")
    
    print(f"Found {len(indices)} positions in ELO range {min_elo}-{max_elo}")
    filtered_dataset = Subset(dataset, indices)
    
    dataloader = DataLoader(
        filtered_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 2. Prepare base model for move prediction
    num_moves = len(dataset.move_to_idx)
    model = prepare_model_for_move_prediction(base_model, num_moves)
    model, target_modules = prepare_model_for_peft(model)
    model.to(device)
    
    # 3. Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Closest to our move prediction task
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    
    # 4. Create PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 5. Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_moves = 0
        total_moves = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (board_states, move_indices, _) in enumerate(progress_bar):
            board_states = board_states.to(device)
            move_indices = move_indices.to(device)
            
            # Forward pass
            outputs = model(board_states)
            loss = criterion(outputs, move_indices)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Stats
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_moves += move_indices.size(0)
            correct_moves += (predicted == move_indices).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": running_loss / (batch_idx + 1),
                "acc": 100 * correct_moves / total_moves
            })
            
        # Print epoch stats
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct_moves / total_moves
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    # 6. Save the adapter
    adapter_name = f"elo_{min_elo}_{max_elo}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, adapter_name))
    
    return model


class ELOMoveAdapterManager:
    """
    Manages ELO-specific LoRA adapters for move prediction
    """
    
    def __init__(
        self,
        base_model,
        dataset,  # Need the dataset for the move vocabulary
        adapters_dir,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.move_to_idx = dataset.move_to_idx
        self.idx_to_move = dataset.idx_to_move
        
        # Prepare the base model for move prediction
        num_moves = len(dataset.move_to_idx)
        self.base_model = prepare_model_for_move_prediction(base_model, num_moves)
        _, self.target_modules = prepare_model_for_peft(self.base_model)
        self.base_model.to(device)
        
        # Load available adapters
        self.adapters_dir = adapters_dir
        self.adapters = {}
        self._scan_adapters()
        
        # Current active adapter
        self.active_adapter = None
        self.model = None
        
    def _scan_adapters(self):
        """Scan the adapters directory for available ELO adapters"""
        if not os.path.exists(self.adapters_dir):
            print(f"Warning: Adapters directory {self.adapters_dir} does not exist")
            return
            
        for adapter_name in os.listdir(self.adapters_dir):
            adapter_path = os.path.join(self.adapters_dir, adapter_name)
            
            if os.path.isdir(adapter_path) and adapter_name.startswith("elo_"):
                try:
                    # Parse ELO range from the adapter name
                    _, min_elo, max_elo = adapter_name.split("_")
                    elo_range = (int(min_elo), int(max_elo))
                    
                    # Store adapter info
                    self.adapters[elo_range] = adapter_path
                    print(f"Found adapter for ELO range {min_elo}-{max_elo}")
                except Exception as e:
                    print(f"Error parsing adapter {adapter_name}: {e}")
    
    def _find_adapter_for_elo(self, elo):
        """Find the most appropriate adapter for the given ELO rating"""
        best_range = None
        
        for elo_range in self.adapters:
            min_elo, max_elo = elo_range
            if min_elo <= elo < max_elo:
                return elo_range
                
        # If no direct match, find the closest
        if not best_range and self.adapters:
            best_range = min(
                self.adapters.keys(), 
                key=lambda x: min(abs(x[0] - elo), abs(x[1] - elo))
            )
            print(f"No exact adapter match for ELO {elo}, using {best_range}")
            
        return best_range
    
    def switch_adapter(self, elo):
        """
        Switch to the adapter most appropriate for the given ELO rating
        """
        elo_range = self._find_adapter_for_elo(elo)
        
        if elo_range is None:
            print("No adapters available")
            return
            
        if elo_range == self.active_adapter:
            print(f"Already using adapter for ELO range {elo_range}")
            return
            
        adapter_path = self.adapters[elo_range]
        
        # Load model with the new adapter
        # First load PEFT config to match the adapter
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=8,  # Use values from training or load from config
            lora_alpha=16,
            lora_dropout=0.0,  # Use 0 for inference
            target_modules=self.target_modules,
        )
        
        # Create a new PEFT model with this config
        model = get_peft_model(self.base_model, peft_config)
        
        # Load the saved adapter weights
        model.load_adapter(adapter_path)
        model.to(self.device)
        
        self.model = model
        self.active_adapter = elo_range
        print(f"Switched to adapter for ELO range {elo_range}")
    
    def predict_move(self, board_state):
        """
        Predicts the next move using the current adapter
        
        Args:
            board_state: Tokenized board state
            
        Returns:
            Most likely move in UCI format
        """
        if self.model is None:
            if not self.adapters:
                raise ValueError("No adapters available")
                
            # Default to the middle adapter
            middle_elo = sum(next(iter(self.adapters.keys()))) // 2
            self.switch_adapter(middle_elo)
            
        self.model.eval()
        with torch.no_grad():
            # Ensure input is properly shaped for the model
            if len(board_state.shape) == 1:
                board_state = board_state.unsqueeze(0)
            
            board_state = board_state.to(self.device)
            
            # Forward pass to get move logits
            logits = self.model(board_state)
            
            # Get the most likely move
            _, move_idx = torch.max(logits, dim=1)
            
            # Convert to UCI format
            move = self.idx_to_move[move_idx.item()]
            
            return move
    
    def predict_move_distribution(self, board_state):
        """
        Predicts the distribution over possible moves
        
        Args:
            board_state: Tokenized board state
            
        Returns:
            Probability distribution over moves
        """
        if self.model is None:
            if not self.adapters:
                raise ValueError("No adapters available")
                
            # Default to the middle adapter
            middle_elo = sum(next(iter(self.adapters.keys()))) // 2
            self.switch_adapter(middle_elo)
            
        self.model.eval()
        with torch.no_grad():
            # Ensure input is properly shaped for the model
            if len(board_state.shape) == 1:
                board_state = board_state.unsqueeze(0)
            
            board_state = board_state.to(self.device)
            
            # Forward pass to get move logits
            logits = self.model(board_state)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=1)
            
            return probs.squeeze(0).cpu().numpy()
