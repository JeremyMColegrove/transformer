import torch.nn as nn
class FeedForward(nn.Module):
   def __init__(self, n_embd):
      super().__init__()
      self.n_embd = n_embd
      self.net = nn.Sequential(
         nn.Linear(self.n_embd, 4 * self.n_embd),
         nn.ReLU(),
         nn.Linear(4 * self.n_embd, self.n_embd),

      )
   def forward(self, x):
      return self.net(x)
