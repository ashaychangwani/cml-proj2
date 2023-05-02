# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

for epoch in range(45):
    optimizer.zero_grad()
    outputs = classifier(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Cross entropy
def cross_entropy(pred, target, epsilon=1e-12):
    pred = np.clip(pred, epsilon, 1. - epsilon)
    N = pred.shape[0]
    ce = -np.sum(target*np.log(pred+1e-9))/N
    return ce

#Gene Importance
results = []

y_pred_base = classifier(X).data.numpy()
y_true = y.data.numpy().tolist()
base_score = cross_entropy(y_pred_base, y_true)
for kinase_number in range(len(X[0])):
    error_increase_list = []
    X_copy = X.clone().detach().numpy()
    for shuffle_number in range (100):
        np.random.shuffle(X_copy[:,kinase_number])
        X_shuffle = torch.tensor(X_copy, dtype=torch.float32)
        y_pred_new = classifier(X_shuffle).data.numpy()
        new_score = cross_entropy(y_pred_new, y_true)
        error_increase = new_score - base_score
        error_increase_list.append(error_increase)
    mean_error = mean(error_increase_list)
    results.append(mean_error)
    print(kinase_number)

results_df = pd.DataFrame(results)
results_df["genes"] = matrix["Gene"]

#Export to excel
results_df.to_excel("gene_importance.xlsx", sheet_name='1')