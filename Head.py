import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
   def __init__(self, head_size, n_emb, block_size):
      super().__init__()
      self.head_size = head_size
      self.n_emb = n_emb
      self.block_size = block_size
      self.key = nn.Linear(self.n_emb, self.head_size, bias=False)
      self.query = nn.Linear(self.n_emb, self.head_size, bias=False)
      self.value = nn.Linear(self.n_emb, self.head_size, bias=False)
      self.register_buffer("tril", torch.tril(torch.ones(self.block_size, self.block_size)))
   def forward(self, x):
      B,T,C = x.shape
      k = self.key(x)
      q = self.query(x)
      wei = q @ k.transpose(-2,-1) * C**-0.5
      wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
      wei = F.softmax(wei, dim=1)
      v = self.value(x)
      out = wei @ v
      return out

