import torch
from torch.utils.data import Dataset, DataLoader
from bagz import BagReader
from apache_beam.coders import StrUtf8Coder, FloatCoder, TupleCoder

from fen_conv import convert_to_token, move_to_id, bin_q

ACTION_CODER = TupleCoder((
    StrUtf8Coder(),   # fen string
    StrUtf8Coder(),   # uci move string
    FloatCoder()      # win percentage float
))

class ChessBenchDataset(Dataset):
    def __init__(self, bag_path: str, transform=None):
        self.reader    = BagReader(bag_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, idx: int):
        raw = self.reader[idx]
        fen, move, win_p = ACTION_CODER.decode(raw)
        board_ids  = convert_to_token(fen)       
        action_id  = move_to_id(move)            
        value_bin  = bin_q(win_p)                

        if self.transform:
            board_ids, action_id, value_bin = self.transform(
                board_ids, action_id, value_bin)

        return board_ids, action_id, value_bin


if __name__ == "__main__":
    ds = ChessBenchDataset("action_value_data.bag")
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)

    for batch in loader:
        board_ids, action_ids, value_bins = batch
        print("board_ids:", board_ids.shape)    # (8, 77)
        print("action_ids:", action_ids.shape)  # (8,)
        print("value_bins:", value_bins.shape)  # (8,)
        break
