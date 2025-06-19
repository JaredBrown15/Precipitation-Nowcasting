import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from itertools import product
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from itertools import product  # Import product for Cartesian product
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 13   # Number of features in each frame
output_size = 13  # Number of features in each frame of the output sequence
learning_rate = 1e-3
input_lengths = [1, 7, 13, 25]
horizon_lengths = [1, 6, 12, 24]


def createXY(dataset, input_length, horizon_length):
    X = []
    Y = []
    for sample in dataset:
        for i in range(len(sample) - input_length - horizon_length + 1):
            x = torch.tensor(sample[i:i+input_length], dtype=torch.float32)
            y = torch.tensor(sample[i+input_length:i+input_length+horizon_length], dtype=torch.float32)
#             x_padded = pad_sequence(x, max_input_len, 0)
#             y_padded = pad_sequence(y, max_horizon_len, 0)
            X.append(x)
            Y.append(y)
    return X, Y


class AutoregressiveGRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(AutoregressiveGRUWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer for processing inputs
        self.lstm = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                             num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layer to project hidden states to output
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Attention Layer
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, lengths, output_length):
        # Pack the padded sequence (for training only)
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Pass the packed input through LSTM
        packed_output, hidden = self.lstm(packed_input)
        
        # Unpack the output (for training only, not used in autoregressive generation)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # Now use hidden state to start autoregressive generation
        next_input = output[:, -1, :].unsqueeze(1)  # Last hidden state as input
        # Initialize hidden and cell states
        next_input = self.fc(next_input.squeeze(1)).unsqueeze(1)  # Shape: [batch_size, 1, input_size]

        hidden = hidden  # [num_layers, batch_size, hidden_size]
        
        outputs = []
        past_predictions = []  # Store past predictions for attention
        
        for t in range(output_length):
            # LSTM generates a new output
#             print("LSTM INPUT SIZE: " + str(x.shape))
            output_lstm, hidden = self.lstm(next_input, hidden)
#             print("OUTPUT_LSTM SIZE: " + str(output_lstm.shape))

            # Use attention on past predictions
            if past_predictions:
                past_preds = torch.stack(past_predictions, dim=1)  # [batch_size, t, output_size]
                attn_weights = F.softmax(self.attn(past_preds), dim=1)  # Attention weights for past preds
                weighted_past_preds = torch.sum(attn_weights * past_preds, dim=1)  # Weighted sum of past predictions
#                 print("WEIGHTED PAST PREDS: " + str(weighted_past_preds.shape))
                output_lstm = output_lstm + weighted_past_preds.squeeze(1)  # Combine with new LSTM output
#                 print("OUTPUT_LSTM SIZE w attn: " + str(output_lstm.shape))
                
            # Fully connected layer for final prediction
            output = self.fc(output_lstm.squeeze(1))  # [batch_size, output_size]
            outputs.append(output)
            
            # Save the current prediction for future attention
            past_predictions.append(output_lstm.unsqueeze(1))  # Add to past predictions list
            
            # Use output as next input for autoregressive generation
            next_input = output.unsqueeze(1)  # This will be the sequence input for next timestep
#             print("OUTPUT SIZE: " + str(output.shape))

        return torch.stack(outputs, dim=1)  # Stack to form a sequence of shape [batch_size, output_length, output_size]


def trainGru(hidden_size, num_layers, num_epochs, train_data_loader):
    # Hyperparameters
    input_size = 13   # Number of features in each frame
    output_size = 13  # Number of features in each frame of the output sequence
    batch_size = 2
    learning_rate = 1e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = AutoregressiveGRUWithAttention(input_size, hidden_size, output_size, num_layers).to(device)

    # Loss function and optimizer
    criterion = nn.HuberLoss(reduction='mean', delta=2)
    criterion = nn.MSELoss()  # Assuming you are doing regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training model
        running_loss = 0.0
        for i, (X_batch, Y_batch, X_lengths, Y_lengths) in enumerate(train_data_loader):  # Assuming a data loader is used

            # Calculate percentage
            percent_done = (i + 1) / len(train_data_loader) * 100
            # Print progress
            print(f"Processing: {percent_done:.2f}% complete for epoch {epoch}", end='\r')


            # Move the batch to the same device as the model
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            X_lengths = X_lengths.to(device)
            Y_lengths = Y_lengths.to(device)

            # Forward pass
            Y_pred = model(X_batch, X_lengths, Y_lengths.max())

            # Compute loss on non-padded parts
            trimmed_y_pred = [arr[n-1] for arr, n in zip(Y_pred, Y_lengths)]
            trimmed_y_true = [arr[n-1] for arr, n in zip(Y_batch, Y_lengths)]
            trimmed_y_pred_tensor = torch.stack(trimmed_y_pred)
            trimmed_y_true_tensor = torch.stack(trimmed_y_true)

        #         print(trimmed_y_pred[0])
        #         print(trimmed_y_true[0])
            loss = criterion(trimmed_y_pred_tensor, trimmed_y_true_tensor)

            # Zero the gradients, perform a backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Print loss
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
#         torch.save(model, "autoregressive_lstm_with_attention.pth")
    return model


def r2ByColumn(y_true, y_pred):
    # Calculate the mean of y_true along each column
    y_mean = np.mean(y_true, axis=0)
    # Calculate SS_tot and SS_res for each column
    ss_tot = np.sum((y_true - y_mean) ** 2, axis=0)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    # Calculate R^2 for each column
    r2 = 1 - (ss_res / ss_tot)
    rounded_r2 = [np.round(x, 4) for x in r2]
    return rounded_r2

class SeqDataset(Dataset):
    def __init__(self, X_data, Y_data, X_lengths, Y_lengths):
        self.X_data = X_data
        self.Y_data = Y_data
        self.X_lengths = X_lengths
        self.Y_lengths = Y_lengths

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        X = self.X_data[idx]
        Y = self.Y_data[idx]
        X_len = self.X_lengths[idx]
        Y_len = self.Y_lengths[idx]
        
        return X, Y, X_len, Y_len
    
    
# Custom collate_fn to pad X and unpack Y sequences
def collate_fn(batch):
    # Sort by length in descending order (for packing)
    batch.sort(key=lambda x: x[2], reverse=True)
    
    X, Y, X_lengths, Y_lengths = zip(*batch)
    
    # Pad X sequences
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    
    # Stack Y sequences (no padding here, since Y is unpadded)
    Y_padded = pad_sequence(Y, batch_first=True, padding_value=0)
    
    # Convert lists to tensors
    X_lengths = torch.tensor(X_lengths)
    Y_lengths = torch.tensor(Y_lengths)

    return X_padded, Y_padded, X_lengths, Y_lengths


def globalR2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    r2_global = 1 - (ss_res / ss_tot)
    return r2_global

def makeTestSetDataloader(key):
    current_x_test = test_dataset[key]["X"][0]
    current_y_test = test_dataset[key]["Y"][0]
    current_x_test_lengths = [key[0]] * len(current_x_test)
    current_y_test_lengths = [key[1]] * len(current_y_test)
    current_test_dataset = SeqDataset(current_x_test, current_y_test, current_x_test_lengths, current_y_test_lengths)
    test_data_loader = DataLoader(current_test_dataset, batch_size=128, shuffle=True, drop_last=True, collate_fn=lambda batch: collate_fn(batch))
    return test_data_loader

test_dataset = {}
def evaluateGRU(model, norm_test):
    model.eval()
    for input_length in input_lengths:
        for horizon_length in horizon_lengths:
#             print("Crunching numbers for Input: " + str(input_length) + " Horizon: " + str(horizon_length))
            cur_x, cur_y = createXY(norm_test, input_length, horizon_length)
            key = (input_length, horizon_length)  # Create a key for the combination
            test_dataset[key] = {"X": [], "Y": []}
            test_dataset[key]["X"].append(cur_x)
            test_dataset[key]["Y"].append(cur_y)
            
    r2_dict = {}
    for key in test_dataset:
        test_data_loader = makeTestSetDataloader(key)
        predictions = []
        y_true = []
        for i, (X_batch, Y_batch, X_lengths, Y_lengths) in enumerate(test_data_loader):
            with torch.no_grad():
                batch_predictions = model(X_batch.to(device), X_lengths.to(device), Y_lengths.max().to(device))

            numpy_batch_predictions = batch_predictions.cpu().numpy()
    #         print(numpy_batch_predictions[0,:,:])
            numpy_batch_predictions = numpy_batch_predictions[:, -1, :] # Take only the final frame for computing r^2

    #         print(numpy_batch_predictions.shape)
            predictions.append(numpy_batch_predictions)
            batch_y_true = Y_batch[:, -1, :] # Take only the final frame for computing r^2
            y_true.append(batch_y_true.cpu().numpy())

        predictions_array = np.array(predictions)
    #     print("INPUT LENGTH: " + str(int(X_lengths[0])))
    #     print("HORIZON LENGTH: " + str(int(Y_lengths[0])))
    #     print("Predictions Length: " + str(predictions_array.shape))

    #     predictions_array = predictions_array.reshape(-1, Y_lengths[0], output_size)
        predictions_array = predictions_array.reshape(-1, 1, output_size)
        predictions_array = np.squeeze(predictions_array)
    #     print("Number of Samples: " + str(predictions_array.shape[0]) + "\n")

        y_true = np.array(y_true)
    #     y_true = y_true.reshape(-1, Y_lengths[0], output_size)
        y_true = y_true.reshape(-1, 1, output_size)
        y_true = np.squeeze(y_true)
    #     print("y_true Length: " + str(y_true.shape))

        r2_column = r2ByColumn(y_true, predictions_array)
    #     r2_global = globalR2(y_true[:,:-5], predictions_array[:,:-5])
        r2_global = globalR2(y_true, predictions_array)

        r2_dict[key] = {"by_column": r2_column, "global": r2_global}
    return r2_dict
        

        
        
        
        
        
        
        
        
        
        
        
        
        