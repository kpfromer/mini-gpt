import torch
from model import BigramLanguageModel
from config import GPTConfig
from hyperparameters import *

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get all unique chars in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

config = GPTConfig(block_size=block_size,
                   vocab_size=vocab_size,
                   n_layers=n_layer,
                   n_heads=n_head,
                   n_embedding=n_embedding,
                   dropout=dropout)

model = BigramLanguageModel(config, device=device)

print("Loading Model")

model.load_state_dict(torch.load('gpt.pt'))
m = model.to(device)

m.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
text = decode(m.generate(context, max_new_tokens=10000)[0].tolist())

with open('gpt.txt', 'w') as f:
    f.write(text)