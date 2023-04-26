import torch
import torch.nn as nn
from torch.nn import functional as F
import json

# HYPERPARAMETERS
batch_size = 64  # how many independent sequences will we process in parallel
block_size = 256  # what is the maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
split = .9
n_embedding = 384
n_head = 6
n_layer = 6
dropout = .2
# -----------------------------------

# Create a file with all the reviews in it
# with open('yelp/reviews.txt', 'a') as reviews_file:
#     with open('yelp/yelp_academic_dataset_review.json', 'r') as f:
#         for line in f:
#             data = json.loads(line)
#             text = data['text']
#             reviews_file.write(f"{text}\n")

with open('yelp/reviews.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get all unique chars in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping to/from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(chars)
print(vocab_size)

# train and test split
# data = torch.tensor(encode(text), dtype=torch.long)
# n = int(split * len(data))
# train_data = data[:n]
# validation_data = data[n:]