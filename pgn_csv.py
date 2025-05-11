import chess.pgn
import chess
import pandas as pd
from fen_conv import move_to_id

def convert_to_df(pgn_path, elo_range):
    df = pd.DataFrame(['fen', 'action'])

    with open(pgn_path) as f:
        while True:
            game_state = chess.pgn.read_game(f)
            if game_state is None:
                break
            
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
            board = game.board()

            if white_elo in range(elo_range) and black_elo in range(elo_range):
                for move in game.mainline_moves():
                    cur_fen = board.fen()
                    uci_move = move.uci() 
                    new_row = pd.DataFrame([{'fen': cur_fen, 'move': uci_move}])
                    df = pd.concat([df, new_row], ignore_index=True)
                    board.push(move)
            
    return df

def df_move_index(df):
    return df['action'].apply(move_to_id)