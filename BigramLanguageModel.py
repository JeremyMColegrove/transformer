import torch
import torch.nn as nn
from torch.nn import functional as F
from Head import *
from FeedForward import *
from MultiHeadAttention import *
from Block import *
from HyperParameters import *
class BigramLanguageModel(nn.Module):
   def __init__(self, params):
      super().__init__()
      self.params = params


      self.to(self.params.device)
      self.token_embedding_table = nn.Embedding(self.params.vocab_size, self.params.n_embd)
      self.position_embedding_table = nn.Embedding(self.params.block_size, self.params.n_embd)
      self.ffwd = FeedForward(self.params.n_embd)
      self.sa_heads = MultiHeadAttention(4, self.params.n_embd//4, self.params.n_embd, self.params.block_size)
      self.blocks = nn.Sequential(
         Block(self.params, n_head=4),
         Block(self.params, n_head=4),
         Block(self.params, n_head=4),
         nn.LayerNorm(self.params.n_embd)
      )
      self.lm_head = nn.Linear(self.params.n_embd, self.params.vocab_size)

   def forward(self, idx, targets=None):

      assert idx is not None and targets is not None
      B, T = idx.shape
      tok_emb = self.token_embedding_table(idx)
      pos_emb = self.position_embedding_table(torch.arange(T, device=self.params.device))
      x = tok_emb + pos_emb
      x = self.blocks(x)
      logits = self.lm_head(x)

      if targets is None:
         loss=None
      else:
         B, T, C = logits.shape
         logits = logits.view(B * T, C)
         targets = targets.view(B * T)
         loss = F.cross_entropy(logits, targets)

      return logits, loss

   def generate(self, idx, max_new_tokens):
      for tok_iter in range(max_new_tokens):
         idx_cond = idx[:, -self.params.block_size:]

         logits, loss = self(idx_cond)

         logits = logits[:, -1, :] # becomes (B, C)

         probs = F.softmax(logits, dim=1)

         idx_next = torch.multinomial(probs, num_samples=1)

         idx = torch.cat((idx, idx_next), dim=1)
      return idx