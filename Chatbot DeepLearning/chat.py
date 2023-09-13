import random
import json
import torch
from model import NeuralNet
from Prototype import bag_of_words, tokenize, stem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('knowledge.json', 'r') as f:
    knowledge = json.load(f)


FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = 'ChatDKY'

print("Let's chat! (type 'quit' to exit)")
while True:
    sentence = input('You: ')
    if sentence == 'quit':
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)

    _, predicted = torch.max(output, dim=1)
    # get the tag 
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.80:
        for intent in knowledge['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")

    else:
        print(f'{bot_name}: I do not understand what you are trying to say.......')