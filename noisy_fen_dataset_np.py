import pandas as pd
import numpy as np
import torch
import chess
from fen_conv import NUM_BUCKETS, BUCKET_MIDPOINTS, convert_to_token, id_to_move, move_to_id
from model_v2 import load_base_model
from sv_move import return_next_move
import glob

SCALE = 0.75
EPSILON = 0.00001

model, device = load_base_model()

def get_win_percentage(fen):
    tokens = convert_to_token(fen) #this new fen will be the state of the opponent, so we want to choose the lowest score here
    tokens = torch.from_numpy(tokens).long().unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tokens)
        probs = torch.softmax(logits, dim = -1)
        win_p = float((probs * torch.from_numpy(BUCKET_MIDPOINTS).to(device)).sum())

    return win_p

def calc_encoding(cur_id, top_6_winps, top_6_ids, target_winp):
    encoding = np.zeros(1968, dtype=float)

    deltas = [abs(target_winp - winp) for winp in top_6_winps]
    similarities = [1.0 / (delta + 0.001) for delta in deltas]
    total_similarity = sum(similarities)

    for i in range(len(top_6_ids)):
        encoding[top_6_ids[i]] = SCALE * similarities[i] / total_similarity  # invert win percentages

    encoding[encoding == 0] = EPSILON # add epsilon
    encoding[cur_id] = 0.0 # set current move to 0
    encoding = encoding / np.sum(encoding) # normalize to sum to 1

    encoding = encoding * (SCALE / (np.sum(encoding)))
    encoding[cur_id] = 1.0 - SCALE
    
    return encoding

def load_chess_data_from_npz(file_pattern):
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
    
    all_fens = []
    all_actions = []
    all_action_ids = []
    
    print(f"Loading {len(files)} NPZ files...")
    for file in files:
        print(f"Loading {file}...")
        data = np.load(file)
        all_fens.append(data['fens'])
        all_actions.append(data['actions'])
        all_action_ids.append(data['action_ids'])

    combined_fens = np.concatenate(all_fens)
    combined_actions = np.concatenate(all_actions)
    combined_action_ids = np.concatenate(all_action_ids)
    
    print(f"Loaded {len(combined_fens)} total records")
    return combined_fens, combined_actions, combined_action_ids

def save_noisy_data_npz(fens, encodings, output_filename):
    print(f"Saving {len(fens)} records to {output_filename}")
    np.savez_compressed(output_filename, 
                       fens=fens,
                       encodings=encodings)
    print(f"Saved to {output_filename}")

def process_in_batches(fens, actions, batch_size=10000, output_prefix="noisy_fen_dataset"):
    total_records = len(fens)
    all_processed_fens = []
    all_encodings = []
    
    for batch_start in range(0, total_records, batch_size):
        batch_end = min(batch_start + batch_size, total_records)
        batch_fens = fens[batch_start:batch_end]
        batch_actions = actions[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}: records {batch_start+1}-{batch_end}")
        
        batch_processed_fens = []
        batch_encodings = []
        
        for i, (fen, move) in enumerate(zip(batch_fens, batch_actions)):
            if (i + 1) % 1000 == 0:
                print(f"  Processing {i+1}/{len(batch_fens)} in current batch")
            
            try:
                board = chess.Board(fen)
                copy_board = board.copy()
                copy_board.push(chess.Move.from_uci(move))
                new_fen = copy_board.fen()
                new_fen_winp = get_win_percentage(new_fen)

                results = return_next_move(fen)
                sorted_results = sorted(results, key=lambda x: abs(x[1] - new_fen_winp))
                top_6_moves = sorted_results[1:7]  
                target_move = sorted_results[0]

                top_6_winps = [top_6_moves[j][1] for j in range(len(top_6_moves))]
                top_6_ids = [move_to_id(top_6_moves[j][0]) for j in range(len(top_6_moves))]

                encoding = calc_encoding(move_to_id(move), top_6_winps, top_6_ids, target_move[1])
                
                batch_processed_fens.append(fen)
                batch_encodings.append(encoding)
                
            except Exception as e:
                print(f"  Error processing record {i}: {e}")
                continue
        
        if batch_processed_fens:
            batch_processed_fens = np.array(batch_processed_fens, dtype='U200')
            batch_encodings = np.array(batch_encodings, dtype=np.float32)
            
            all_processed_fens.append(batch_processed_fens)
            all_encodings.append(batch_encodings)
            
    if all_processed_fens:
        final_fens = np.concatenate(all_processed_fens)
        final_encodings = np.concatenate(all_encodings)
        
        save_noisy_data_npz(final_fens, final_encodings, f"{output_prefix}.npz")
        return final_fens, final_encodings
    else:
        print("No data was successfully processed!")
        return None, None

def main():
    file_pattern = "chess_data_1200_1600_*.npz"  
    
    try:
        fens, actions, action_ids = load_chess_data_from_npz(file_pattern)
        processed_fens, encodings = process_in_batches(
            fens, actions, 
            batch_size=10000, 
            output_prefix="noisy_fen_dataset_1200_1600"
        )
        
        if processed_fens is not None:
            print(f"\nprocessed {len(processed_fens)} records")
    except Exception as e:
        print(f"Error: {e}")

def load_noisy_data(filename):
    data = np.load(filename)
    return data['fens'], data['encodings']

if __name__ == "__main__":
    main()
