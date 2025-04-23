import pandas as pd
from bagz import BagReader
from apache_beam.coders import StrUtf8Coder, FloatCoder, TupleCoder
import argparse
import os
import csv
from tqdm import tqdm

# 1) Beam coder exactly as the repo does
STATE_CODER = TupleCoder((
    StrUtf8Coder(),  # FEN string
    FloatCoder(),    # Stockfish win% float
))
ACTION_CODER = TupleCoder((
    StrUtf8Coder(),  # FEN
    StrUtf8Coder(),  # move
    FloatCoder(),    # win percent
))

def bag_to_sv_csv(path: str, out_path: str, limit: int | None, chunk_size: int = 100000):
    """Process state-value bag file directly to CSV in chunks without loading everything into memory"""
    rdr = BagReader(path)
    
    # Initialize CSV file
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "win_percent"])  # Write header
        
        # Process in chunks
        chunk = []
        total_processed = 0
        
        with tqdm(desc="Processing records") as pbar:
            for i, raw in enumerate(rdr):
                if limit is not None and i >= limit:
                    break
                
                fen, win_p = STATE_CODER.decode(raw)
                chunk.append([fen, win_p])
                
                # Write chunk when it reaches chunk_size
                if len(chunk) >= chunk_size:
                    writer.writerows(chunk)
                    total_processed += len(chunk)
                    pbar.update(len(chunk))
                    chunk = []  # Clear the chunk
            
            # Write any remaining records
            if chunk:
                writer.writerows(chunk)
                total_processed += len(chunk)
                pbar.update(len(chunk))
        
        print(f"Total records processed: {total_processed}")

def bag_to_av_csv(path: str, out_path: str, limit: int | None, chunk_size: int = 100000):
    """Process action-value bag file directly to CSV in chunks without loading everything into memory"""
    rdr = BagReader(path)
    
    # Initialize CSV file
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "move", "win_percent"])  # Write header
        
        # Process in chunks
        chunk = []
        total_processed = 0
        
        with tqdm(desc="Processing records") as pbar:
            for i, raw in enumerate(rdr):
                if limit is not None and i >= limit:
                    break
                
                fen, move, win_p = ACTION_CODER.decode(raw)
                chunk.append([fen, move, win_p])
                
                # Write chunk when it reaches chunk_size
                if len(chunk) >= chunk_size:
                    writer.writerows(chunk)
                    total_processed += len(chunk)
                    pbar.update(len(chunk))
                    chunk = []  # Clear the chunk
            
            # Write any remaining records
            if chunk:
                writer.writerows(chunk)
                total_processed += len(chunk)
                pbar.update(len(chunk))
        
        print(f"Total records processed: {total_processed}")

def sample_bag_to_dataframe(path: str, sample_size: int = 1000) -> pd.DataFrame:
    """Extract a small sample from the bag file to inspect"""
    rows, rdr = [], BagReader(path)
    for i, raw in enumerate(rdr):
        if i >= sample_size:
            break
        try:
            # Try both decoders
            try:
                fen, win_p = STATE_CODER.decode(raw)
                rows.append({"fen": fen, "win_percent": win_p})
            except:
                fen, move, win_p = ACTION_CODER.decode(raw)
                rows.append({"fen": fen, "move": move, "win_percent": win_p})
        except Exception as e:
            print(f"Error decoding record {i}: {e}")
            continue
    return pd.DataFrame(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DeepMind .bag/.bagz files to a CSV with memory-efficient processing"
    )
    parser.add_argument("path", help="Path to .bag or .bagz file")
    parser.add_argument(
        "--kind",
        choices=["sv", "av"],
        required=True,
        help="sv = state-value | av = action-value",
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
    parser.add_argument(
        "-c",
        "--chunk",
        type=int,
        default=100000,
        help="Chunk size for processing (default: 100000)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Only extract a small sample (1000 records) for inspection",
    )
    
    args = parser.parse_args()
    
    # Set output path if not specified
    out_path = args.out or f"{os.path.splitext(os.path.basename(args.path))[0]}.csv"
    
    if args.sample:
        print(f"Extracting sample of 1000 records from {args.path}...")
        df = sample_bag_to_dataframe(args.path)
        sample_path = f"{os.path.splitext(out_path)[0]}_sample.csv"
        df.to_csv(sample_path, index=False)
        print(f"Sample saved to {sample_path}")
        print(df.head())
        print(f"Sample size: {len(df)}")
    else:
        print(f"Processing {args.path} to {out_path} with chunk size {args.chunk}...")
        if args.kind == "sv":
            bag_to_sv_csv(args.path, out_path, args.num, args.chunk)
        else:
            bag_to_av_csv(args.path, out_path, args.num, args.chunk)
        
        # Display the first few rows of the output file
        try:
            sample = pd.read_csv(out_path, nrows=5)
            print("\nFirst 5 rows of output:")
            print(sample)
        except Exception as e:
            print(f"Error reading output file: {e}")