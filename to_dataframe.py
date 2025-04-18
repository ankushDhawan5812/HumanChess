import pandas as pd
from bagz import BagReader
from apache_beam.coders import StrUtf8Coder, FloatCoder, TupleCoder

# 1) Beam coder exactly as the repo does
ACTION_CODER = TupleCoder((
    StrUtf8Coder(),   # FEN string
    StrUtf8Coder(),   # UCI move string
    FloatCoder(),     # Stockfish win% float
))

def bag_to_dataframe(path: str, n: int = 1000) -> pd.DataFrame:
    """
    Reads up to n records from `path` (.bag or .bagz)
    and returns a DataFrame with columns ['fen','move','win_percent'].
    """
    rdr = BagReader(path)
    rows = []
    for i in range(min(len(rdr), n)):
        raw = rdr[i]
        fen, move, win_p = ACTION_CODER.decode(raw)
        rows.append({'fen': fen, 'move': move, 'win_percent': win_p})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = bag_to_dataframe("action_value_data.bag", n=2000000)
    print(df.head())
    print(len(df))
