import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import your model and our behavioral cloning module
from model_v2 import load_base_model
from bc_lora import (
    ChessMoveDataset,
    train_elo_move_adapter,
    ELOMoveAdapterManager
)

# Configuration
DEFAULT_DATASET_PATH = # TODO: Create dataset
DEFAULT_ADAPTERS_DIR = "elo_move_adapters"
DEFAULT_ELO_RANGES = [
    (0, 600),
    (600, 1200),
    (1200, 1800),
    (1800, 2400),
    (2400, 3000)
]


def prepare_elo_move_dataset():
    """
    Prepare a dataset with chess moves and ELO ratings
    This function would need to be adapted to your specific data format
    """
    # Load dataset
    dataset = ChessMoveDataset(DEFAULT_DATASET_PATH)
    return dataset

def train_all_elo_adapters(base_model, dataset, adapters_dir, elo_ranges, device):
    """Train LoRA adapters for all ELO ranges"""
    for min_elo, max_elo in DEFAULT_ELO_RANGES:
        print(f"\n=== Training adapter for ELO range {min_elo}-{max_elo} ===")
        
        try:
            # Train adapter for this ELO range
            train_elo_move_adapter(
                base_model=base_model,
                dataset=dataset,
                min_elo=min_elo,
                max_elo=max_elo,
                output_dir=DEFAULT_ADAPTERS_DIR,
                batch_size=64,
                num_epochs=6, 
                learning_rate=1e-4,
                lora_rank=8,
                lora_alpha=16,
                lora_dropout=0.1,
                device=device
            )
            print(f"Completed training for ELO range {min_elo}-{max_elo}")
        except Exception as e:
            print(f"Error training adapter for ELO range {min_elo}-{max_elo}: {e}")

def display_chess_position(fen, move=None):
    """
    Create a simple ASCII representation of a chess position
    with the predicted move highlighted
    
    Args:
        fen: FEN string of the position
        move: UCI format move to highlight (e.g., 'e2e4')
    """
    # Extract the board part from FEN
    board_part = fen.split(' ')[0]
    rows = board_part.split('/')
    
    # Expand numbers to empty squares
    expanded_rows = []
    for row in rows:
        expanded = ""
        for char in row:
            if char.isdigit():
                expanded += '.' * int(char)
            else:
                expanded += char
        expanded_rows.append(expanded)
    
    # Prepare move coordinates if provided
    move_from = move_to = None
    if move:
        if len(move) >= 4:  # Basic sanity check
            file_from = ord(move[0]) - ord('a')
            rank_from = 8 - int(move[1])  # Convert to 0-7 index
            file_to = ord(move[2]) - ord('a')
            rank_to = 8 - int(move[3])  # Convert to 0-7 index
            
            move_from = (rank_from, file_from)
            move_to = (rank_to, file_to)
    
    # Print the board with row and column labels
    print("  a b c d e f g h")
    print(" +-----------------+")
    for i, row in enumerate(expanded_rows):
        rank = 8 - i
        line = f"{rank}| "
        for j, piece in enumerate(row):
            if move_from and (i, j) == move_from:
                line += f"[{piece}] "
            elif move_to and (i, j) == move_to:
                line += f"({piece}) "
            else:
                line += f"{piece} "
        print(line + f"|{rank}")
    print(" +-----------------+")
    print("  a b c d e f g h")
    
    if move:
        print(f"Move: {move}")


def test_adapters(base_model, dataset, adapters_dir, device):
    """Test switching between different ELO adapters"""
    # Create adapter manager
    adapter_manager = ELOMoveAdapterManager(
        base_model=base_model,
        dataset=dataset,
        adapters_dir=adapters_dir,
        device=device
    )
    
    # Test positions - we'll use a few common positions
    test_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # After 1.e4
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"  # Ruy Lopez setup
    ]
    
    # Test with different ELO levels
    test_elos = [400, 900, 1500, 2200]
    
    for pos_idx, fen in enumerate(test_positions):
        print(f"\n\n=== Position {pos_idx+1} ===")
        display_chess_position(fen)
        
        # Tokenize position
        test_position = dataset.tokenize_fen(fen)
        board_state = torch.tensor(test_position, dtype=torch.long)
        
        for elo in test_elos:
            print(f"\n=== Testing with ELO {elo} ===")
            
            # Switch to appropriate adapter
            adapter_manager.switch_adapter(elo)
            
            # Predict move
            move = adapter_manager.predict_move(board_state)
            print(f"Predicted move: {move}")
            
            # Display the position with the move
            display_chess_position(fen, move)
            
            # Get move distribution
            move_probs = adapter_manager.predict_move_distribution(board_state)
            
            # Get top 5 moves
            top_indices = np.argsort(move_probs)[-5:][::-1]
            print("Top 5 moves:")
            for i, idx in enumerate(top_indices):
                move = adapter_manager.idx_to_move[idx]
                prob = move_probs[idx] * 100
                print(f"  {i+1}. {move}: {prob:.2f}%")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train and test ELO-specific LoRA adapters")
    parser.add_argument("--model_path", type=str, default="models/model_epoch_2.pth",
                        help="Path to base model weights")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH,
                        help="Path to CSV with chess positions, moves, and ELO ratings")
    parser.add_argument("--adapters_dir", type=str, default=DEFAULT_ADAPTERS_DIR,
                        help="Directory to save adapters")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"],
                        default="both", help="Mode of operation")
    args = parser.parse_args()
    
    print("=== Loading Base Model ===")
    base_model, device = load_base_model(args.model_path)
    
    print("\n=== Preparing Dataset ===")
    dataset = prepare_elo_move_dataset(args.dataset)
    
    if args.mode in ["train", "both"]:
        print("\n=== Training Adapters ===")
        train_all_elo_adapters(base_model, dataset, args.adapters_dir, DEFAULT_ELO_RANGES, device)
    
    if args.mode in ["test", "both"]:
        print("\n=== Testing Adapters ===")
        test_adapters(base_model, dataset, args.adapters_dir, device)

if __name__ == "__main__":
    main()
