import pandas as pd
from bagz import BagReader
from apache_beam.coders import StrUtf8Coder, FloatCoder, TupleCoder
import argparse 
import os

# 1) Beam coder exactly as the repo does
STATE_CODER = TupleCoder((
    StrUtf8Coder(),   # FEN string
    FloatCoder(),     # Stockfish win% float
))

ACTION_CODER = TupleCoder((
    StrUtf8Coder(), #FEN
    StrUtf8Coder(), #mov
    FloatCoder(), #win percent
))

def bag_to_sv_dataframe(path: str, limit: int | None) -> pd.DataFrame:
    rows, rdr = [], BagReader(path)
    for i, raw in enumerate(rdr):
        if limit is not None and i >= limit:
            break
        fen, win_p = STATE_CODER.decode(raw)
        rows.append({"fen": fen, "win_percent": win_p})
    return pd.DataFrame(rows)

def bag_to_av_dataframe(path: str, limit: int | None) -> pd.DataFrame:
    rows, rdr = [], BagReader(path)
    for i, raw in enumerate(rdr):
        if limit is not None and i >= limit:
            break
        fen, move, win_p = ACTION_CODER.decode(raw)
        rows.append({"fen": fen, "move": move, "win_percent": win_p})
    return pd.DataFrame(rows)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        description="Convert DeepMind .bag/.bagz files to a CSV."
    )
    parser.add_argument("path", help="Path to .bag or .bagz file")
    parser.add_argument(
        "--kind",
        choices=["sv", "av"],
        required=True,
        help="sv = state‑value  |  av = action‑value",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=None,
        help="Number of records to read (default: all)",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="CSV output path (default: <bagname>.csv)",
    )
    args = parser.parse_args()

    if args.kind == "sv":
        df = bag_to_sv_dataframe(args.path, args.num)
    else:
        df = bag_to_av_dataframe(args.path, args.num)

    out_path = args.out or f"{os.path.splitext(os.path.basename(args.path))[0]}.csv"
    df.to_csv(out_path, index=False)
    print(df.head())
    print(len(df))
