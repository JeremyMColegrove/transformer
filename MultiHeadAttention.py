import torch.nn as nn
from Head import *
class MultiHeadAttention(nn.Module):
   def __init__(self, num_heads, head_size, n_embd, block_size):
      super().__init__()
      self.num_heads = num_heads
      self.head_size = head_size
      self.n_embd = n_embd
      self.block_size = block_size
      self.heads = nn.ModuleList([Head(self.head_size, self.n_embd, self.block_size) for _ in range(self.num_heads)])
      self.proj = nn.Linear(self.n_embd, self.n_embd)

   def forward(self, x):
      out = torch.cat([h(x) for h in self.heads], dim=-1)
      out = self.proj(out)
      return out

