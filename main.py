import tiktoken
import torch
import torch.nn as nn
from BigramLanguageModel import *
from HyperParameters import *

params = HyperParameters()

device = torch.device("cpu")
if torch.backends.mps.is_available():
   if torch.backends.mps.is_built():
      if params.mps_is_available:
         device = torch.device("mps")
   else:
      print("Torch backend is not built for mps")
else:
   print("mps is not available on this device")
params.device = device
print(f"Running on {device}...")

print("Reading train.txt...")
with open("train.txt", "r", encoding="utf-8") as f:
   text = f.read()
   f.close()


# define encode and decode functions
chars = sorted(list(set(text)))
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
if params.use_tiktoken == True:
   print("Getting gpt2 tokens...")
   enc = tiktoken.get_encoding("gpt2")
   params.vocab_size = enc.n_vocab
   encode = lambda s: enc.encode(s)
   decode = lambda l: enc.decode(l)
else:
   params.vocab_size = len(chars)
   encode = lambda s: [stoi[c] for c in s]
   decode = lambda l: ''.join([itos[i] for i in l])

print(f"vocab size = {params.vocab_size}")
# encode the training text
print("Encoding data...")
enc_data = torch.tensor(encode(text), dtype=torch.long)

# get test and train data
n = int(params.training_split*len(enc_data))
train_data = enc_data[:n]
test_data = enc_data[n:]


def get_batch(data):
   ix = torch.randint(len(data)-params.block_size, (params.batch_size,))
   x = torch.stack([data[i:i+params.block_size] for i in ix])
   y = torch.stack(([data[i+1:i+params.block_size+1] for i in ix]))
   x, y = x.to(device), y.to(device)
   return x, y


# creating model
print("Creating language model...")
model = BigramLanguageModel(params)

# train the model
optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)

# training
print("Training model...")
for iter in range(params.max_iterations):

   # sample
   xb, yb = get_batch(train_data)

   assert len(xb) == len(yb) and xb is not None and yb is not None
   
   #get loss
   logits, loss = model(xb, yb)
   optimizer.zero_grad(set_to_none=True)
   loss.backward()
   optimizer.step()

   if iter % params.eval_interval == 0:
      print(f"{iter}: loss: {loss.item()}")

# generate data
idx = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(idx, max_new_tokens=100)[0].tolist()))
