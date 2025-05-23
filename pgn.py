import os
import zstandard as zstd
import chess
import chess.pgn
import pandas as pd
import io
import numpy as np
from tqdm import tqdm
import argparse

def process_pgn_zst(pgn_zst_path, output_csv_path, max_games=None, min_elo=0, max_elo=3000):
    """
    Process a compressed .pgn.zst file into a CSV file with FEN positions, moves, and ELO ratings
    
    Args:
        pgn_zst_path: Path to the .pgn.zst file
        output_csv_path: Path to save the output CSV
        max_games: Maximum number of games to process (None for all games)
        min_elo: Minimum ELO rating to include
        max_elo: Maximum ELO rating to include
    """
    print(f"Processing {pgn_zst_path}...")
    
    # Initialize data lists
    positions = []
    moves = []
    elos = []
    
    # Check if output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Open the zstd compressed file
    with open(pgn_zst_path, 'rb') as compressed_file:
        # Create a decompressor
        dctx = zstd.ZstdDecompressor()
        
        # Create a reader that decompresses the data
        with dctx.stream_reader(compressed_file) as reader:
            # Text IO for chess.pgn to read
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            
            # Process games
            game_count = 0
            
            # Setup progress display
            pbar = tqdm(desc="Processing games", unit="game")
            
            while True:
                # Check if we've reached the maximum games
                if max_games is not None and game_count >= max_games:
                    break
                
                # Read the next game
                game = chess.pgn.read_game(text_stream)
                
                # Check if we've reached the end of the file
                if game is None:
                    break
                
                # Check if the game has ELO ratings
                white_elo = game.headers.get("WhiteElo", "?")
                black_elo = game.headers.get("BlackElo", "?")
                
                # Skip games with missing ELO
                if white_elo == "?" or black_elo == "?":
                    continue
                
                # Convert to integers
                white_elo = int(white_elo)
                black_elo = int(black_elo)
                
                # Skip games with ELO outside our range
                if (white_elo < min_elo or white_elo > max_elo or 
                    black_elo < min_elo or black_elo > max_elo):
                    continue
                
                # Initialize board
                board = game.board()
                
                # Process moves
                for move in game.mainline_moves():
                    # Current player's ELO
                    player_elo = white_elo if board.turn == chess.WHITE else black_elo
                    
                    # Store the current position before the move
                    positions.append(board.fen())
                    
                    # Store the move in UCI format
                    moves.append(move.uci())
                    
                    # Store the player ELO
                    elos.append(player_elo)
                    
                    # Make the move on the board
                    board.push(move)
                
                # Increment counter and update progress
                game_count += 1
                pbar.update(1)
                
                # Periodically save progress to avoid memory issues
                if game_count % 1000 == 0:
                    print(f"\nProcessed {game_count} games, {len(positions)} positions")
                    
                    # Create a temporary dataframe and save progress
                    temp_df = pd.DataFrame({
                        'fen': positions,
                        'move': moves,
                        'player_elo': elos
                    })
                    
                    # If file exists, append without header
                    if os.path.exists(output_csv_path):
                        temp_df.to_csv(output_csv_path, mode='a', header=False, index=False)
                    else:
                        # First write includes header
                        temp_df.to_csv(output_csv_path, index=False)
                    
                    # Clear the lists to free memory
                    positions.clear()
                    moves.clear()
                    elos.clear()
            
            pbar.close()
    
    # Save any remaining data
    if positions:
        temp_df = pd.DataFrame({
            'fen': positions,
            'move': moves,
            'player_elo': elos
        })
        
        if os.path.exists(output_csv_path):
            temp_df.to_csv(output_csv_path, mode='a', header=False, index=False)
        else:
            temp_df.to_csv(output_csv_path, index=False)
    
    print(f"Completed processing {game_count} games")
    print(f"Dataset saved to {output_csv_path}")

def sample_balanced_dataset(input_csv, output_csv, elo_ranges, samples_per_range=20000):
    """
    Create a balanced dataset with equal samples from each ELO range
    
    Args:
        input_csv: Path to the input CSV file
        output_csv: Path to save the balanced dataset
        elo_ranges: List of (min_elo, max_elo) tuples
        samples_per_range: Number of samples to include from each range
    """
    print(f"Creating balanced dataset from {input_csv}...")
    
    # Read the dataset in chunks to handle large files
    balanced_data = []
    
    for chunk in tqdm(pd.read_csv(input_csv, chunksize=100000), desc="Processing chunks"):
        for min_elo, max_elo in elo_ranges:
            # Filter positions in this ELO range
            range_data = chunk[(chunk['player_elo'] >= min_elo) & (chunk['player_elo'] < max_elo)]
            
            # Add to balanced data
            balanced_data.append(range_data)
    
    # Combine all chunks
    combined_data = pd.concat(balanced_data)
    
    # Shuffle the data
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # For each ELO range, select a balanced subset
    balanced_subsets = []
    
    for min_elo, max_elo in elo_ranges:
        # Filter positions in this ELO range
        range_data = combined_data[(combined_data['player_elo'] >= min_elo) & 
                                   (combined_data['player_elo'] < max_elo)]
        
        # Sample (or take all if fewer than requested)
        if len(range_data) > samples_per_range:
            range_sample = range_data.sample(samples_per_range, random_state=42)
        else:
            range_sample = range_data
            print(f"Warning: Only {len(range_data)} samples available for ELO range {min_elo}-{max_elo}")
        
        balanced_subsets.append(range_sample)
    
    # Combine all balanced subsets
    final_dataset = pd.concat(balanced_subsets)
    
    # Shuffle again
    final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    final_dataset.to_csv(output_csv, index=False)
    
    print(f"Balanced dataset created with {len(final_dataset)} total positions")
    print(f"Saved to {output_csv}")
    
    # Print distribution
    for min_elo, max_elo in elo_ranges:
        count = len(final_dataset[(final_dataset['player_elo'] >= min_elo) & 
                                 (final_dataset['player_elo'] < max_elo)])
        print(f"ELO range {min_elo}-{max_elo}: {count} positions")

def main():
    parser = argparse.ArgumentParser(description="Process PGN.ZST files for chess move prediction")
    parser.add_argument("--pgn_path", type=str, required=True, help="Path to the .pgn.zst file")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save processed data")
    parser.add_argument("--max_games", type=int, default=None, help="Maximum number of games to process")
    parser.add_argument("--balance", action="store_true", help="Create a balanced dataset across ELO ranges")
    
    args = parser.parse_args()
    
    # Define ELO ranges
    elo_ranges = [
        (0, 600),
        (600, 1200),
        (1200, 1800),
        (1800, 2400),
        (2400, 3000)
    ]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define output paths
    raw_csv_path = os.path.join(args.output_dir, "chess_moves_raw.csv")
    balanced_csv_path = os.path.join(args.output_dir, "chess_moves_with_elo.csv")
    
    # Process the PGN.ZST file
    process_pgn_zst(args.pgn_path, raw_csv_path, args.max_games)
    
    # Create balanced dataset if requested
    if args.balance:
        sample_balanced_dataset(raw_csv_path, balanced_csv_path, elo_ranges, samples_per_range=10000)

if __name__ == "__main__":
    main()