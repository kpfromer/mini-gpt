import torch
import torch.nn as nn
from torch.nn import functional as F
from config import GPTConfig


class Head(nn.Module):
    """ one head of self-attenion """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.head_size = config.head_size
        self.key = nn.Linear(config.n_embedding, self.head_size, bias=False)
        self.query = nn.Linear(config.n_embedding, self.head_size, bias=False)
        self.value = nn.Linear(config.n_embedding, self.head_size, bias=False)
        self.register_buffer(
            'tril', torch.tril(torch.ones(config.block_size,
                                          config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attenion scores ("affinities")
        wei = q @ k.transpose(
            -2,
            -1) * self.head_size**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # perform the weighted aggreation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(config) for _ in range(config.n_heads)])
        self.projection = nn.Linear(config.n_embedding, config.n_embedding)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # the project is a linear transformation of the outcome of the multi-head attentions
        out = self.dropout(self.projection(out))
        return out


"""
fast with linear = 
fast but no linear * 3 = 76 seconds
slow = 135 seconds
"""


class MultiHeadAttentionEfficient(nn.Module):
    """ multiple heads of self-attention in parallel, but more efficient """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embedding % config.n_heads == 0, "n_embedding must be divisible by n_heads"

        self.n_heads = config.n_heads

        self.key = nn.Linear(config.n_embedding,
                             config.n_embedding,
                             bias=False)
        self.query = nn.Linear(config.n_embedding,
                               config.n_embedding,
                               bias=False)
        self.value = nn.Linear(config.n_embedding,
                               config.n_embedding,
                               bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size))

        self.projection = nn.Linear(config.n_embedding, config.n_embedding)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape  # where C = n_embedding

        # print(f"B={B}, T={T}, C={C}, n_heads={self.n_heads}")
        # print(f"weight shape: {self.key.weight.shape}")
        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(
            1, 2
        )  # (B,T,C) -> (B,n_heads,T,head_size) where head_size = C // n_heads
        q = self.query(x).view(
            B, T, self.n_heads, C // self.n_heads
        ).transpose(
            1, 2
        )  # (B,T,C) -> (B,n_heads,T,head_size) where head_size = C // n_heads
        v = self.value(x).view(
            B, T, self.n_heads, C // self.n_heads
        ).transpose(
            1, 2
        )  # (B,T,C) -> (B,n_heads,T,head_size) where head_size = C // n_heads

        # compute attention scores ("affinities")
        attention = q @ k.transpose(-2, -1) * (
            C // self.n_heads
        )**-0.5  # (B,n_heads,T,head_size) @ (B,n_heads,head_size,T) -> (B,n_heads,T,T)
        attention = attention.masked_fill(self.tril[:, :, :T, :T] == 0,
                                          float('-inf'))
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        # perform the weighted aggreation of the values
        attention = attention @ v  # (B,n_heads,T,T) @ (B,n_heads,T,head_size) -> (B,n_heads,T,head_size)
        out = attention.transpose(1, 2).contiguous().view(
            B, T, C)  # (B,n_heads,T,head_size) -> (B,T,C)

        return self.dropout(self.projection(out))


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embedding, 4 * config.n_embedding),
            nn.ReLU(),
            nn.Linear(4 * config.n_embedding,
                      config.n_embedding),  # projection layer
            nn.Dropout(config.dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        # self.sa = MultiHeadAttentionEfficient(config)
        self.ffwd = FeedForward(config)
        self.linear_norm_1 = nn.LayerNorm(config.n_embedding)
        self.linear_norm_2 = nn.LayerNorm(config.n_embedding)

    def forward(self, x):
        # the x + is the residual connection
        x = x + self.sa(self.linear_norm_1(x))
        x = x + self.ffwd(self.linear_norm_2(x))
        return x


class BigramLanguageModel(nn.Module):
    """
    Notes:
    B = the batch size
    T = the target size
    C = the number of channels (n_embedding)
    """

    def __init__(self, config: GPTConfig, device):
        super().__init__()

        self.device = device

        # each token directly reads off the logits for the next token from a lookup table
        self.block_size = config.block_size
        self.token_embedding_table = nn.Embedding(config.vocab_size,
                                                  config.n_embedding)
        self.position_embedding_table = nn.Embedding(config.block_size,
                                                     config.n_embedding)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layers)])
        self.layer_norm = nn.LayerNorm(config.n_embedding)
        # language modeling head
        self.lm_head = nn.Linear(config.n_embedding, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_embeddings = self.token_embedding_table(idx)  # (B,T,C)
        position_embeddings = self.position_embedding_table(
            torch.arange(T, device=self.device))  # (T,C)
        x = token_embeddings + position_embeddings  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.layer_norm(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last config.block_size, tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilites
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx