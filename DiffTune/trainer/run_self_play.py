from get_diffuse_action import denoise_input
from model.lora_model import LoraDiffusionModel
from data.fen_conv_diff import convert_to_token, move_to_id, id_to_move
import chess
import board
import torch

fen_str = "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1"
board = chess.Board(fen_str) # set up current FEN

def get_winner(board):
    if not board.is_game_over():
        return None
    result = board.result()
    if result == "1-0":
        return 0
    elif result == "0-1":
        return 1
    else:
        return -1

def run_sim(elo_one, elo_two, start_fen, board):
    elos = [elo_one, elo_two]
    cur_elo_idx = 0
    while True:
        copy_board = board.copy()
        with torch.no_grad():
            logits_player_1 = denoise_input(fen_str, elos[cur_elo_idx])
            logits_action = logits_player_1[0, 0, 0:1968]
            sorted_logits, sorted_indices = torch.sort(logits_action, descending=True)
            cur_index = 0
            action_id = sorted_indices[cur_index].item()
            while id_to_move(action_id) not in board.legal_moves:
                cur_index += 1
                action_id = sorted_indices[cur_index].item()

        action = id_to_move(action_id)
        copy_board.push(chess.Move.from_uci(action))
        fen_str = copy_board.fen()
        cur_elo_idx = 1 - cur_elo_idx
        if copy_board.is_game_over():
           return get_winner(copy_board)

  


