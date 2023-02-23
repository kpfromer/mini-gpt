import torch
from hyperparameters import *
from model import BigramLanguageModel
from config import GPTConfig
import datetime
import os.path

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    # for line in f:
    #     chars.update(line)

# get all unique chars in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping to/from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(split * len(data))
train_data = data[:n]
validation_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# used for evaluating the performance of the model
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


config = GPTConfig(block_size=block_size,
                   vocab_size=vocab_size,
                   n_layers=n_layer,
                   n_heads=n_head,
                   n_embedding=n_embedding,
                   dropout=dropout)

# Load checkpoint if it exists
if os.path.isfile('checkpoint.pt'):
    print("Loading checkpoint")
    checkpoint = torch.load('checkpoint.pt')
    config = checkpoint['config']
    model = BigramLanguageModel(config, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iter = checkpoint['iter']
    print(f"Starting from iteration {iter}")
else:
    print("Creating new model")
    model = BigramLanguageModel(config, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    iter = 0

m = model.to(device)
m.train()

print("Training Model")

print(f"Using device={device}")
start_time = datetime.datetime.now()
while iter < max_iters:
    # every once in a while evalute the loss on the train and validation sets
    if iter % eval_interval == 0:
        losses = estimate_loss(m)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        torch.save(
            {
                'iter': iter,
                'model_state_dict': m.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, 'checkpoint.pt')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    iter += 1

end_time = datetime.datetime.now()

print(
    f"Training complete - time: {(end_time - start_time).total_seconds()} seconds"
)

torch.save(m.state_dict(), 'gpt.pt')