# Neural-Network-Classificationex002
# EX:2 Developing a Neural Network Classification Model

## Name: CJ ROHIT
## Register Number: 212224243005
## Department: Artificial Intelligence and Machine Learning (AIML)
## College: Saveetha Engineering College

## AIM
## To develop a neural network classification model to predict
## customer segments A, B, C, and D.

## PROGRAM

## Method 1: Import Required Libraries

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


## Method 2: Load the Dataset

df = pd.read_csv(r"E:\deep\deeex\dataset\customers.csv")
df.head()


## Method 3: Remove the ID Column

df = df.drop(columns=["ID"])
df.head()


## Method 4: Display Column Names

print(df.columns)


## Method 5: Handle Missing Values

df.fillna({
    "Work_Experience": 0,
    "Family_Size": df["Family_Size"].median()
}, inplace=True)

cat_columns = [
    "Gender",
    "Ever_Married",
    "Graduated",
    "Profession",
    "Spending_Score",
    "Var_1"
]

for col in cat_columns:
    df[col] = df[col].fillna(df[col].mode()[0])


## Method 6: Encode Categorical Variables

encoders = {}

for col in cat_columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col].astype(str))

lbe = LabelEncoder()
df["Segmentation"] = lbe.fit_transform(df["Segmentation"].astype(str))

df.head()


## Method 7: Separate Features and Target

x = df.drop(columns=["Segmentation"])
y = df["Segmentation"].values


## Method 8: Split the Dataset

xt, xst, yt, yst = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


## Method 9: Normalize the Features

scaler = StandardScaler()

xt = scaler.fit_transform(xt)
xst = scaler.transform(xst)


## Method 10: Convert Data into PyTorch Tensors

xt = torch.FloatTensor(xt)
xst = torch.FloatTensor(xst)

yt = torch.LongTensor(yt)
yst = torch.LongTensor(yst)


## Method 11: Create DataLoaders

tr = TensorDataset(xt, yt)
tst = TensorDataset(xst, yst)

trl = DataLoader(tr, batch_size=16, shuffle=True)
tstl = DataLoader(tst, batch_size=16, shuffle=False)


## Method 12: Build the Neural Network Model

class classifier1(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.l1 = nn.Linear(input_size, 32)
        self.l2 = nn.Linear(32, 16)
        self.l3 = nn.Linear(16, 8)
        self.l4 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)

        return x


## Method 13: Initialize the Model

model = classifier1(input_size=xt.shape[1])

criterion = nn.CrossEntropyLoss()
op = optim.Adam(model.parameters(), lr=0.001)


## Method 14: Train the Model

epochs = 100

for i in range(epochs):
    model.train()
    total_loss = 0

    for a, b in trl:
        op.zero_grad()

        pred = model(a)
        loss = criterion(pred, b)

        loss.backward()
        op.step()

        total_loss += loss.item()

    if (i + 1) % 10 == 0:
        print(
            f"Epoch: {i + 1}/{epochs}, "
            f"Loss: {total_loss / len(trl):.4f}"
        )


## Method 15: Predict on Test Data

model.eval()

with torch.no_grad():
    output = model(xst)
    _, predicted = torch.max(output, 1)

pre = predicted.numpy()
act = yst.numpy()

print("Actual Labels:", act)
print("Predicted Labels:", pre)


## Method 16: Evaluate the Model

accuracy = accuracy_score(act, pre)

conf_matrix = confusion_matrix(
    act,
    pre,
    labels=[0, 1, 2, 3]
)

cl_report = classification_report(
    act,
    pre,
    labels=[0, 1, 2, 3],
    target_names=lbe.classes_,
    zero_division=0
)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", cl_report)


## Method 17: Plot the Confusion Matrix

xl = lbe.classes_

sns.heatmap(
    conf_matrix,
    annot=True,
    cmap="Blues",
    xticklabels=xl,
    yticklabels=xl,
    fmt="g"
)

plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


## Method 18: Result
## output
<img width="769" height="890" alt="image" src="https://github.com/user-attachments/assets/2b8e7b62-00da-4c18-a3ca-8ec20706af82" />
<img width="645" height="911" alt="image" src="https://github.com/user-attachments/assets/11cf0fc0-35ca-42f1-8bee-1d5ff38230ba" />
<img width="538" height="914" alt="image" src="https://github.com/user-attachments/assets/d982946c-e103-449c-8834-8c8cf0194d78" />
<img width="367" height="900" alt="image" src="https://github.com/user-attachments/assets/193cbf00-f9f7-4257-b212-d015ebf882c6" />
<img width="514" height="901" alt="image" src="https://github.com/user-attachments/assets/877d1846-be41-43a6-921b-fccfef0de329" />


print("Neural Network Classification Model Executed Successfully.")
