# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from statistics import mean

# Matrix processing
matrix = pd.read_csv('final_matrix.csv')
matrix.set_index('Gene')
matrix2 = matrix.values
matrix3 = matrix2.T
final_matrix = pd.DataFrame(matrix3)

# Importing the dataset
X = final_matrix.iloc[1:, 0:26323].values
y = final_matrix.iloc[1:, 26323].values

# Convert the dataset to float32 dtype
X = X.astype(np.float32)
y = y.astype(np.float32)

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Create TensorDataset and DataLoader
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Building classifier (Fill with hyperparameters of choice)
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(len(X[0]), 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def build_classifier():
    classifier = Classifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters())
    return classifier, criterion, optimizer

classifier, criterion, optimizer = build_classifier()

# Check if CUDA is available and set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device:', device)

# Print the GPU name if CUDA is available
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(device))

# Move the model to the specified device
classifier = classifier.to(device)

# Training
for epoch in range(45):
    for batch_X, batch_y in data_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = classifier(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Cross entropy
def cross_entropy(pred, target, epsilon=1e-12):
    pred = np.clip(pred, epsilon, 1. - epsilon)
    N = pred.shape[0]
    ce = -np.sum(target*np.log(pred+1e-9))/N
    return ce

# Gene Importance
results = []

y_pred_base = classifier(X.to(device)).data.cpu().numpy()
y_true = y.data.cpu().numpy().tolist()
base_score = cross_entropy(y_pred_base, y_true)

for kinase_number in range(len(X[0])):
    error_increase_list = []
    X_copy = X.clone().detach().cpu().numpy()
    for shuffle_number in range(100):
        np.random.shuffle(X_copy[:, kinase_number])
        X_shuffle = torch.tensor(X_copy, dtype=torch.float32).to(device)
        y_pred_new = classifier(X_shuffle).data.cpu().numpy()
        new_score = cross_entropy(y_pred_new, y_true)
        error_increase = new_score - base_score
        error_increase_list.append(error_increase)
    mean_error = mean(error_increase_list)
    results.append(mean_error)
    print(kinase_number, '/', len(X[0]))

results_df = pd.DataFrame(results)
results_df['genes'] = matrix['Gene']

# Export to excel
results_df.to_excel('gene_importance.xlsx', sheet_name='1')