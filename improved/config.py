from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 64
    vocab_size: int = 256
    n_layers: int = 6
    n_heads: int = 6
    n_embedding: int = 384
    dropout: float = 0.2

    @property
    def head_size(self):
        return self.n_embedding // self.n_heads