import chess.pgn
import pandas as pd
from fen_conv import move_to_id
import zstandard as zstd
import io
from tqdm import tqdm
import numpy as np
import os
from parallel_pandas import ParallelPandas


def df_move_index(df):
    df['action_id'] = df['action'].apply(move_to_id)
    return df


def save_data(rows_by_range):
    for label, rows in rows_by_range.items():
        if rows:
            df = pd.DataFrame(rows)
            df = df_move_index(df)
            filename = f"chess_data_{label}.csv"
            write_header = not os.path.exists(filename)
            df.to_csv(filename, index=False, mode='a', header=write_header)
            print(f"Appended {len(rows)} rows to {filename}")


def convert_all_ranges(pgn_path, ranges):
    rows_by_range = {label: [] for label in ranges}
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

            for label, (elo_min, elo_max) in ranges.items():
                if elo_min <= white_elo <= elo_max and elo_min <= black_elo <= elo_max:
                    game_counts[label] += 1
                    board = game.board()
                    for move in game.mainline_moves():
                        rows_by_range[label].append({
                            'fen': board.fen(),
                            'action': move.uci()
                        })
                        board.push(move)
                    break

            if total_games % 2000000 == 0:
                print(f"\n📝 Saving progress at {total_games} games...")
                save_data(rows_by_range)

        pbar.close()
        print(f"\n🔚 Finished after scanning {total_games} games.")
        save_data(rows_by_range)
    print("Final counts:", game_counts)


if __name__ == "__main__":
    pgn_path = "lichess_db_standard_rated_2025-04.pgn.zst"

    ranges = {
        "800_1200": (800, 1200),
        "1200_1600": (1201, 1600),
        "1601_2000": (1601, 2000),
        "2001_2400": (2001, 2400)
    }

    # Process all ranges in one pass
    convert_all_ranges(pgn_path, ranges)

    # for label, rows in rows_by_range.items():
    #     print(f"Saving {len(rows)} rows for {label}...")
    #     df = pd.DataFrame(rows)
    #     df = df_move_index(df)
    #     df.to_csv(f"chess_data_{label}.csv", index=False)
    #     print(f"Saved: chess_data_{label}.csv")
