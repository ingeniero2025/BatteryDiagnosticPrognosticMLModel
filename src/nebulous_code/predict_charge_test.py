import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class BatteryDataset(Dataset):
    """
    Dataset class for battery time series data.
    """
    def __init__(self, data, sequence_length, scaler=None):
        if scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.normalized_data = self.scaler.fit_transform(data.reshape(-1, 1))
        else:
            self.scaler = scaler
            self.normalized_data = self.scaler.transform(data.reshape(-1, 1))
            
        self.sequence_length = sequence_length
        self.sequences, self.targets = self._create_sequences()
        
    def _create_sequences(self):
        """Create sequences for training from the normalized data."""
        sequences, targets = [], []
        
        for i in range(self.sequence_length, len(self.normalized_data)):
            sequences.append(self.normalized_data[i-self.sequence_length:i])
            targets.append(self.normalized_data[i])
            
        return (torch.FloatTensor(np.array(sequences)),
                torch.FloatTensor(np.array(targets)))
    
    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a sequence and its corresponding target."""
        return self.sequences[idx], self.targets[idx]
    
    def inverse_transform(self, data):
        """Convert normalized values back to original scale."""
        return self.scaler.inverse_transform(data.reshape(-1, 1))
    
def main():
    data = np.random.rand(100)
    sequence_length = 10
    dataset = BatteryDataset(data, sequence_length)

    print(f"Dataset length: {len(dataset)}")
    sequence, target = dataset[0]
    print(f"Sequence shape: {sequence.shape}")  # Should be (10, 1)
    print(f"Target shape: {target.shape}")      # Should be (1,)

if __name__ == "__main__":
    main()