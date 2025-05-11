import pandas as pd
import numpy as np

# Load the dataset
file_path = '/Users/Ankush/Documents/Stanford/CS224R/HumanChess/noisy_fen_dataset.csv'
data = pd.read_csv(file_path)

for i in range(len(data)):
    # Get the FEN string - first column
    # fen = data.iloc[i, 0]
    
    # Get the encoding - all other columns
    encoding = data.iloc[i, 1:]
    encoding = encoding.to_numpy()
    encoding = encoding.astype(np.float32)
    print(np.sum(encoding))