# Databricks notebook source
import torch
import torch.nn.functional as F

# COMMAND ----------

with open("names.txt", "r") as f:
    all_names = f.read().splitlines()

# COMMAND ----------

#names 
print(all_names[:8])
print(f"Total names: {len(all_names)}")

# COMMAND ----------

all_chars = sorted(list(set("".join(all_names))))

stoi = {ch:i+1 for i, ch in enumerate(all_chars)}
stoi["."] = 0

itos = {value: key for key, value in stoi.items()}

# COMMAND ----------

print(stoi)
print(itos)

# COMMAND ----------

CONTEXT_WINDOW = 3

CH_EMB_DIM = 10

VOCAB = 27

X = []
Y = []

# COMMAND ----------

for name in all_names:
    # print(f"name: {name}")
    extended_name  = "..." + name + "."
    # print(f"extended name: {extended_name}")
    for idx in range(len(extended_name)-CONTEXT_WINDOW):
        in_char = extended_name[idx:idx+CONTEXT_WINDOW]
        out_char = extended_name[idx+CONTEXT_WINDOW]
        # print(f"{in_char} ---> {out_char}")
        X.append([stoi[x] for x in list(in_char)])
        Y.append(stoi[out_char])
    # print("_"*30)

# COMMAND ----------

X = torch.tensor(X)
Y = torch.tensor(Y)

n_batch = X.shape[0]

# COMMAND ----------

X.shape, X.dtype, Y.shape, Y.dtype

# COMMAND ----------

for idx in range(5):
    print(f"x: {X[idx]} --> y: {Y[idx]}")

# COMMAND ----------

# MAGIC %md
# MAGIC # MLP Network 

# COMMAND ----------

input_size_1 = CONTEXT_WINDOW * CH_EMB_DIM
out_size_1 = 200

input_size_2 = 200
out_size_2 = VOCAB

lrs = [0.1, 0.01, 0.001] 
epochs = 56000
minibatch = 32

# COMMAND ----------

# MAGIC %md
# MAGIC # Initalize Weights and biases

# COMMAND ----------

gen = torch.Generator().manual_seed(2147483647)

C = torch.randn((VOCAB, CH_EMB_DIM), requires_grad=True, generator=gen)

w1 = torch.randn((input_size_1, out_size_1), requires_grad=True, generator=gen)
b1 = torch.randn(out_size_1, requires_grad=True, generator=gen)

w2 = torch.randn((input_size_2, out_size_2), requires_grad=True, generator=gen)
b2 = torch.randn(out_size_2, requires_grad=True, generator=gen)

parameters = [C, w1, b1, w2, b2]

# COMMAND ----------

C.shape

# COMMAND ----------

sum(p.nelement() for p in parameters)

# COMMAND ----------

# MAGIC %md
# MAGIC ### forward pass

# COMMAND ----------

for epoch in range(epochs):

    # Minibatch
    sample = torch.randint(0, X.shape[0], (minibatch,))
        
    # Forward Pass
    emb = torch.reshape(C[X[sample]], (minibatch, (CONTEXT_WINDOW * CH_EMB_DIM)))
    out = torch.tanh(emb @ w1 + b1)
    logits = out @ w2 + b2

    for p in parameters:
        p.grad = None
    # Loss
    loss = F.cross_entropy(logits, Y[sample])
    loss.backward()

    if epoch < 50000:
        lr = lrs[0]
    elif epoch < 10000:
        lr = lrs[1]
    else:
        lr = lrs[2]

    # update
    for p in parameters:
        p.data += - lr * p.grad
    
    if epoch % 5000 == 0:
        print(f"{epoch=} loss: {loss.item():.4f}")

# COMMAND ----------


emb = torch.reshape(C[X], (n_batch, (CONTEXT_WINDOW * CH_EMB_DIM)))
out = torch.tanh(emb @ w1 + b1)
logits = out @ w2 + b2

for p in parameters:
    p.grad = None
# Loss
loss = F.cross_entropy(logits, Y)
print(loss)

# COMMAND ----------

# sample from the model
g = torch.Generator().manual_seed(2147483647)

for _ in range(200):
    
    out = []
    context = [0] * CONTEXT_WINDOW # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(1, -1) @ w1 + b1)
      logits = h @ w2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      if ix == 0:
        break
      out.append(ix)
    
    print(''.join(itos[i] for i in out))

# COMMAND ----------

import matplotlib.pyplot as plt

# visualize dimensions 0 and 1 of the embedding matrix C for all characters
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
plt.grid('minor')