from torch.utils.data import Dataset, DataLoader

class TakDataset(Dataset):
    def __init__(self, rows):               # rows: list of (tps, move_index, result)
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        return self.rows[index]                 # anything: tuple, dict, tensors