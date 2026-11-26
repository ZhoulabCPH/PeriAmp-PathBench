# data_loader.py
import os
import torch
import pandas as pd
import numpy as np
import h5py
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence


class H5Dataset(Dataset):
    def __init__(self, h5_dir, excel_file, class_names,transform=None):
        self.h5_dir = h5_dir
        self.excel_file = excel_file
        self.transform = transform

        # Read the label workbook and select available cases.
        self.class_names = class_names
        self.df = pd.read_excel(excel_file)
        self.df = self.df[self.df['available'] == 1]  # Keep rows marked as available.

        # Check the configured class columns.
        assert all([col in self.df.columns for col in class_names]), 'The label workbook is missing one or more class columns.'

        self.class_df = self.df[class_names]
        self.file_ids = self.df['ID'].astype(str).values
        self.labels = self.df[self.class_names].idxmax(axis=1)
        self.label_codes = pd.Categorical(
            self.labels,
            categories=class_names
        ).codes
        # print("First five labels:", self.labels[:5])
        # print("Class column order:", self.df.columns[1:].tolist())

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        try:
            label = self.labels.iloc[idx]
        except Exception as e:
            raise Exception(f"Could not process case {file_id}.") from e
        # Read the feature and coordinate arrays.
        h5_path = os.path.join(self.h5_dir, f"{file_id}.h5")
        with h5py.File(h5_path, 'r') as f:
            features = f['features'][:]
            coords = f['coords'][:]
        if self.transform:
            features, coords = self.transform(features, coords)

        return features, coords, self.label_codes[idx], file_id


def collate_fn(batch):
    """Collate variable-length bags into padded feature and coordinate tensors with labels, lengths, and a validity mask. Features and coordinates have shape (batch, max_n, feature_dim) and (batch, max_n, 2)."""
    # Unpack the samples in the batch.
    features = [torch.tensor(item[0], dtype=torch.float32) for item in batch]
    coords = [torch.tensor(item[1], dtype=torch.int32) for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    file_ids = [item[3] for item in batch]  # Case identifiers

    # Compute the valid length of each bag.
    lens = torch.tensor([len(item[0]) for item in batch], dtype=torch.long)

    # Pad features and coordinates to a common length.
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    padded_coords = pad_sequence(coords, batch_first=True, padding_value=0)

    # Mark valid rows in the padded batch.
    masks = [torch.ones(len(item[0]), dtype=torch.bool) for item in batch]
    padded_masks = pad_sequence(masks, batch_first=True, padding_value=False)

    return padded_features, padded_coords, labels, lens, padded_masks, file_ids  # Include the original case IDs.


# Split the dataset using the configured ratios.
def split_dataset(dataset, train_ratio=None, val_ratio=None, test_ratio=None):
    """Split a dataset into up to three subsets using the requested training, validation, and test ratios."""
    total_size = len(dataset)
    ratios = [r for r in (train_ratio, val_ratio, test_ratio) if r is not None]
    assert sum(ratios) == 1.0, 'The split ratios must sum to 1.'

    sizes = [int(total_size * r) for r in ratios]
    sizes[-1] = total_size - sum(sizes[:-1])  # Assign the remainder to the last split.

    subsets = random_split(dataset, sizes)
    return tuple(subsets)
