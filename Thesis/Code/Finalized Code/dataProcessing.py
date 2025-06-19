from torch.utils.data import random_split



class SeqDataset(Dataset):
    def __init__(self, X_data, Y_data):
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        X = self.X_data[idx]
        Y = self.Y_data[idx]
        
        return X, Y
    
# Custom collate_fn to pad X and unpack Y sequences
# def collate_fn(batch):
#     # Sort by length in descending order (for packing)
#     batch.sort(key=lambda x: x[2], reverse=True)
    
#     X, Y, X_lengths, Y_lengths = zip(*batch)
    
#     # Pad X sequences
#     X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    
#     # Stack Y sequences (no padding here, since Y is unpadded)
#     Y_padded = pad_sequence(Y, batch_first=True, padding_value=0)
    
#     # Convert lists to tensors
#     X_lengths = torch.tensor(X_lengths)
#     Y_lengths = torch.tensor(Y_lengths)

#     return X_padded, Y_padded, X_lengths, Y_lengths


def createDataLoader(X_all_Train, Y_all_Train):
    training_torch_dataset = SeqDataset(X_all_Train, Y_all_Train)

    # Assuming you have a dataset called 'dataset'
    dataset_size = len(training_torch_dataset) 
    subset_size = int(0.2 * dataset_size)  # Taking 20% of the dataset
    subset, _ = random_split(training_torch_dataset, [subset_size, dataset_size - subset_size])
#     train_data_loader = DataLoader(training_torch_dataset, batch_size=256, shuffle=True, drop_last=True, collate_fn=lambda batch: collate_fn(batch))
    train_data_loader = DataLoader(subset, batch_size=256, shuffle=True, drop_last=True)
    return train_data_loader




