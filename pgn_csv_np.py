import chess.pgn
from fen_conv import move_to_id
import zstandard as zstd
import io
from tqdm import tqdm
import numpy as np
import os

def save_data_numpy(arrays_by_range, file_counters):
    """Save numpy arrays to npz files"""
    for label, arrays in arrays_by_range.items():
        if arrays['fens']:  # Check if there's data to save
            # Convert lists to numpy arrays
            fens_array = np.array(arrays['fens'], dtype='U200')  # Unicode string with max 200 chars
            actions_array = np.array(arrays['actions'], dtype='U10')  # UCI moves are typically short
            action_ids_array = np.array(arrays['action_ids'], dtype=np.int32)
            
            # Create filename with counter
            filename = f"chess_data_{label}_{file_counters[label]:03d}.npz"
            
            # Save to npz file
            np.savez_compressed(filename, 
                               fens=fens_array,
                               actions=actions_array, 
                               action_ids=action_ids_array)
            
            print(f"Saved {len(fens_array)} rows to {filename}")
            
            # Increment counter for next file
            file_counters[label] += 1
            
            # Clear the arrays for next batch
            arrays['fens'].clear()
            arrays['actions'].clear()
            arrays['action_ids'].clear()

def convert_all_ranges_numpy(pgn_path, ranges, batch_size=1000000):
    """Convert PGN to numpy arrays, saving every batch_size games"""
    # Initialize storage for each range
    arrays_by_range = {
        label: {
            'fens': [],
            'actions': [],
            'action_ids': []
        } for label in ranges
    }
    
    # Track file counters for each range
    file_counters = {label: 0 for label in ranges}
    game_counts = {label: 0 for label in ranges}
    total_games = 0
    
    with open(pgn_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(f)
        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
        
        pbar = tqdm(desc="Scanning games", unit="game")
        
        while True:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break
            
            # Filter by time control
            time_control = game.headers.get("TimeControl", "")
            if "+" in time_control:
                try:
                    initial_time = int(time_control.split("+")[0])
                    if initial_time < 300:
                        continue
                except ValueError:
                    continue
            else:
                continue
            
            total_games += 1
            pbar.update(1)
            
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
            
            # Check which range this game belongs to
            for label, (elo_min, elo_max) in ranges.items():
                if elo_min <= white_elo <= elo_max and elo_min <= black_elo <= elo_max:
                    game_counts[label] += 1
                    board = game.board()
                    
                    # Process each move in the game
                    for move in game.mainline_moves():
                        fen = board.fen()
                        action = move.uci()
                        action_id = move_to_id(action)
                        
                        # Add to arrays
                        arrays_by_range[label]['fens'].append(fen)
                        arrays_by_range[label]['actions'].append(action)
                        arrays_by_range[label]['action_ids'].append(action_id)
                        
                        board.push(move)
                    break  # Only count game once
            
            # Save progress every batch_size games
            if total_games % batch_size == 0:
                print(f"\n📝 Saving progress at {total_games} games...")
                save_data_numpy(arrays_by_range, file_counters)
        
        pbar.close()
        print(f"\n🔚 Finished after scanning {total_games} games.")
        
        # Save any remaining data
        save_data_numpy(arrays_by_range, file_counters)
    
    print("Final counts:", game_counts)
    print("Files created per range:", file_counters)

def load_chess_data_numpy(file_pattern):
    """
    Helper function to load and concatenate numpy arrays from multiple files
    Example usage: data = load_chess_data_numpy("chess_data_1200_1600_*.npz")
    """
    import glob
    
    all_fens = []
    all_actions = []
    all_action_ids = []
    
    files = sorted(glob.glob(file_pattern))
    print(f"Loading {len(files)} files...")
    
    for file in files:
        data = np.load(file)
        all_fens.append(data['fens'])
        all_actions.append(data['actions'])
        all_action_ids.append(data['action_ids'])
    
    if all_fens:
        combined_data = {
            'fens': np.concatenate(all_fens),
            'actions': np.concatenate(all_actions),
            'action_ids': np.concatenate(all_action_ids)
        }
        print(f"Loaded {len(combined_data['fens'])} total records")
        return combined_data
    else:
        print("No files found!")
        return None

if __name__ == "__main__":
    pgn_path = "lichess_db_standard_rated_2025-04.pgn.zst"
    
    ranges = {
        "800_1200": (800, 1200),
        "1200_1600": (1201, 1600),
        "1601_2000": (1601, 2000),
        "2001_2400": (2001, 2400)
    }
    
    # Process all ranges in one pass with numpy arrays
    convert_all_ranges_numpy(pgn_path, ranges, batch_size=1000000)
    
    # Example of how to load the data back:
    # data_1200_1600 = load_chess_data_numpy("chess_data_1200_1600_*.npz")