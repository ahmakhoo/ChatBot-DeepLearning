if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from Prototype import bag_of_words, tokenize, stem

from model import NeuralNet


with open('knowledge.json', 'r') as f:
    knowledge = json.load(f)

# print(knowledge)

all_words = []
tags = []
empty = []

for intent in knowledge['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        empty.append((w, tag))

ignore_words = ['?', '.', '!', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# print(all_words)
all_words = sorted(set(all_words))
tags = sorted(set(tags))
# print(tags)

X_train = []
y_train = []

for (pattern_sentence, tag) in empty:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)


X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples



dataset = ChatDataset()

hidden_size = 8
input_size = len(X_train[0])
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

# print(input_size, len(all_words))
# print(output_size, tags)


train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for(words, labels) in train_loader:
        words =  words.to(device)
        labels = labels.to(dtype = torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.3f}')  

print(f'final loss , loss={loss.item():.3f}')  


data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'all_words': all_words,
    'tags': tags
}

FILE = 'data.pth'
torch.save(data, FILE)

print(f'traning complete. file saved to {FILE}')