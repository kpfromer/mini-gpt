import torch

# batch_size = 64  # how many independent sequences will we process in parallel
# block_size = 256  # what is the maximum context length for predictions
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# split = .9
# n_embedding = 384
# n_head = 6
# n_layer = 6
# dropout = .2

batch_size = 8  # how many independent sequences will we process in parallel
block_size = 32  # what is the maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
split = .9
n_embedding = 32
n_head = 4
n_layer = 3
dropout = .2