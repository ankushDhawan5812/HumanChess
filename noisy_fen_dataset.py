'''
Read in fen string and corresponding action
Determine next state fen string
Pass fen string to the model to determine the win percentage
Let this win percentage be set to 50%

Find all possible next state fen strings for the given fen string
Determine the win percentages for each of the next state fen strings
Find the top 6 next state fen strings with the closest win percentages

Set the win percentage for these next 6 fen strings to be the remaining 50% for the win percentage.
Set all 0s to be epsilon

Update the fen string label to be this combination of the 5 fen strings plus the original fen string

'''

import pandas as pd
import numpy as np
import torch
import chess
from fen_conv import NUM_BUCKETS, BUCKET_MIDPOINTS, convert_to_token, id_to_move, move_to_id

from model_v2 import load_base_model

from sv_move import return_next_move

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
    encoding = encoding / np.sum(encoding) # normalize to sum to 1

    encoding = encoding * (SCALE / (np.sum(encoding)))

    encoding[cur_id] = 1 - SCALE
    return encoding

def main():
    # read in current fen string
    indexes = np.arange(0, 1968)
    str_indexes = [str(i) for i in indexes]
    new_fen_df = pd.DataFrame(columns=["FEN"] + str_indexes)

    fen_df = pd.read_csv("chess_data_1200_1800.csv")
    for i in range(10):
    # for i in range(len(fen_df))
        fen = fen_df.iloc[i, 0]
        move = fen_df.iloc[i, 1]
        print(f"Processing {i+1}/{len(fen_df)}: FEN: {fen}, Move: {move}")

        # get next fen string win percentage
        board = chess.Board(fen)
        copy_board = board.copy()
        copy_board.push(chess.Move.from_uci(move))
        new_fen = copy_board.fen()
        new_fen_winp = get_win_percentage(new_fen)

        # get next top 6 closest win percentages
        results = return_next_move(fen)
        sorted_results = sorted(results, key=lambda x: abs(x[1] - new_fen_winp)) # Sort results by the absolute difference in win percentage
        top_6_moves = sorted_results[1:7]  # Skip the first one as it is the current move, select the top 6 moves closest to the current win percentage, excluding the current move
        target_move = sorted_results[0]

        top_6_winps = [top_6_moves[i][1] for i in range(len(top_6_moves))] # sorted by the deltas
        top_6_ids = [move_to_id(top_6_moves[i][0]) for i in range(len(top_6_moves))]

        encoding = calc_encoding(move_to_id(move), top_6_winps, top_6_ids, target_move[1])

        data = {"FEN": [fen]}
        for i in range(len(encoding)):
            data[str_indexes[i]] = [encoding[i]]
        df_i = pd.DataFrame(data)
        new_fen_df = pd.concat([new_fen_df, df_i], ignore_index=True)

    new_fen_df.to_csv('noisy_fen_dataset.csv', index=False)

if __name__ == "__main__":
    main()





        



        

    