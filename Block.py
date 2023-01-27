import torch.nn as nn
from MultiHeadAttention import *
from FeedForward import *
from HyperParameters import *
class Block(nn.Module):
   def __init__(self, params, n_head=4):
      super().__init__()
      self.params = params

      head_size = self.params.n_embd // n_head
      self.sa = MultiHeadAttention(n_head, head_size, self.params.n_embd, self.params.block_size )
      self.ffwd = FeedForward(self.params.n_embd)
      self.ln1 = nn.LayerNorm(self.params.n_embd)
      self.ln2 = nn.LayerNorm(self.params.n_embd)
   def forward(self, x):
      x = x + self.sa(self.ln1(x))
      x = x + self.ffwd(self.ln2(x))

      return x