# Databricks notebook source
import random
import torch
import torch.nn.functional as F

random.seed(42)

# COMMAND ----------

with open("/Workspace/Users/hitesh.meena@renew.com/RNN/mlp/names.txt", "r") as f:
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

def build_dataset(names):
    X, Y = [], []

    for name in names:
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

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    print("X,Y shapes: ", X.shape, Y.shape)

    return X, Y

random.shuffle(all_names)
n1 = int(len(all_names)*0.8)
n2 = int(len(all_names)*0.9)

Xtrain, Ytrain = build_dataset(all_names[:n1])
Xval, Yval = build_dataset(all_names[n1:n2])
Xtest, Ytest = build_dataset(all_names[n2:])

# COMMAND ----------

n_embed = 10
n_hidden = 200
VOCAB = 27

# COMMAND ----------

for idx in range(5):
    print(f"x: {Xtrain[idx]} --> y: {Ytrain[idx]}")

# COMMAND ----------

# MAGIC %md
# MAGIC # MLP Network 

# COMMAND ----------

input_size_1 = CONTEXT_WINDOW * n_embed

lrs = [0.1, 0.01, 0.001]

epochs = 200000
batch_size = 32

# COMMAND ----------

# MAGIC %md
# MAGIC # Initalize Weights and biases

# COMMAND ----------

gen = torch.Generator().manual_seed(2147483647)

C = torch.randn((VOCAB, n_embed),          generator=gen)

w1 = torch.randn((input_size_1, n_hidden), generator=gen) * 0.2
b1 = torch.randn(n_hidden,                 generator=gen) * 0.01

w2 = torch.randn((n_hidden, VOCAB),        generator=gen) * 0.01
b2 = torch.randn(VOCAB,                    generator=gen) * 0

parameters = [C, w1, b1, w2, b2]

for p in parameters:
    p.requires_grad=True

# COMMAND ----------

sum(p.nelement() for p in parameters)

# COMMAND ----------

# MAGIC %md
# MAGIC ### forward pass

# COMMAND ----------

for epoch in range(epochs):

    # Minibatch
    sample = torch.randint(0, Xtrain.shape[0], (batch_size,))
        
    # Forward Pass
    emb = torch.reshape(C[Xtrain[sample]], (batch_size, (CONTEXT_WINDOW * n_embed)))
    hpreac = emb @ w1 + b1
    out = torch.tanh(hpreac)
    logits = out @ w2 + b2

    for p in parameters:
        p.grad = None
    # Loss
    loss = F.cross_entropy(logits, Ytrain[sample])
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

split_data = {
    "train": (Xtrain, Ytrain),
    "test": (Xtest, Ytest),
    "valid": (Xval, Yval)
}

@torch.no_grad()
def split_loss(Xdata, Ydata, split):

    emb = torch.reshape(C[Xdata], (Xdata.shape[0], (CONTEXT_WINDOW * n_embed)))
    out = torch.tanh(emb @ w1 + b1)
    logits = out @ w2 + b2

    for p in parameters:
        p.grad = None
    # Loss
    loss = F.cross_entropy(logits, Ydata)
    print(f"split: {split} loss: {loss.item():.4f}")

split_loss(*split_data["train"], "train")
split_loss(*split_data["test"], "test")
split_loss(*split_data["valid"], "valid")

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

# COMMAND ----------



# COMMAND ----------

logits = torch.tensor([1/27]*27)

# COMMAND ----------

F.softmax(logits)

# COMMAND ----------

plt.hist(hpreac.view(-1).tolist(),bins=50)
plt.show()

# COMMAND ----------

plt.figure(figsize=(20,10))
plt.imshow(out.abs() > 0.99, cmap="gray", interpolation="nearest" )