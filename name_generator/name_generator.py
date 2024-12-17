# Databricks notebook source
import torch
from collections import defaultdict

import pandas as pd

# COMMAND ----------

with open("names.txt", "r") as f:
    all_names = f.read().splitlines()

# COMMAND ----------

#names 
print(all_names[:5])
print(f"Total names: {len(all_names)}")

# COMMAND ----------

all_chars = sorted(list(set("".join(all_names))))

stoi = {ch:i+1 for i, ch in enumerate(all_chars)}
stoi["."] = 0

itos = {value: key for key, value in stoi.items()}

# COMMAND ----------

N = torch.zeros((27,27), dtype=torch.int32)

# COMMAND ----------

# bigram table
bigram_table = defaultdict(int)

for name in all_names:
    
    name = "." + name + "."
        
    for index in range(len(name)-1):
        bigram = (name[index], name[index+1])
        bigram_table[bigram] += 1

# COMMAND ----------

bigram_statistics = torch.zeros((27,27))

for key, value in bigram_table.items():
    i, j = stoi[key[0]], stoi[key[1]]
    bigram_statistics[ i, j ] = value

bigram_statistics = bigram_statistics/bigram_statistics.sum(dim=1, keepdim=True)

# COMMAND ----------

N = pd.DataFrame(0,columns=["."]+all_chars,index=["."]+all_chars)

# Loop through the bigram table and assign the values to the DataFrame
for key, value in bigram_table.items():
    
    # Assign the value in the DataFrame at the position of (key[0], key[1])
    N.loc[key[0], key[1]] = value

# COMMAND ----------

N

# COMMAND ----------

N = N.div(N.sum(axis=1), axis=0)

# COMMAND ----------

N

# COMMAND ----------

generator_cpu = torch.Generator().manual_seed(2147483647)

# COMMAND ----------

names_req = 100

for idx in range(names_req):
    ch = "."
    predicted_name = ""

    while True:
        p = torch.Tensor(list(N.loc[ch,:]))
        ch = torch.multinomial(p, 1, replacement=True, generator=generator_cpu).item()
        ch = N.columns[ch]
        
        if ch == ".":
            break
        predicted_name += ch
        
    print(predicted_name)

# COMMAND ----------

 # negative log likelihood function log(p(a) * p(b) * p(c))
log_loss = 0
n=0

for name in all_names[:3]:
    
    name = "." + name + "."
        
    for index in range(len(name)-1):
        ch1, ch2 = (name[index], name[index+1])
        p = torch.tensor(N.loc[ch1,ch2])
        print( f"{ch1}{ch2}: prob: {p:.4f}" )
        log_loss += torch.log( p )
        n+=1

neg_log_loss = -log_loss

print(f"{log_loss=}")
print(f"{neg_log_loss=}")
print(f"{neg_log_loss/n:.4f}")

# COMMAND ----------

# negative log likelihood function log(p(a) * p(b) * p(c))
X, ytarget = [], [] 

for name in all_names:
    
    name = "." + name + "."
        
    for index in range(len(name)-1):
        ch1, ch2 = (name[index], name[index+1])
        X.append(stoi[ch1])
        ytarget.append(stoi[ch2])

X = torch.tensor(X)
ytarget = torch.tensor(ytarget)

# COMMAND ----------

print(f"{X = }")
print(f"{ytarget = }")

# COMMAND ----------

ytarget.shape[0]

# COMMAND ----------

W = torch.randn((27,27), generator=generator_cpu, requires_grad=True)
X_one_hot = torch.nn.functional.one_hot(X, 27).float()

# COMMAND ----------

from tqdm import tqdm

# COMMAND ----------

epochs = 1000

num = ytarget.shape[0]
losses = []

for epoch in tqdm(range(epochs), "Progress"):
    logits = X_one_hot @ W
    counts = logits.exp()
    probs = counts/ counts.sum(dim=1, keepdim=True)
    loss = -probs[torch.arange(num), ytarget].log().mean()

    W.grad = None
    loss.backward()
    W.data += - 0.1 * W.grad

    losses.append(loss.item())

# COMMAND ----------

import matplotlib.pyplot as plt

# COMMAND ----------

plt.plot(losses)

# COMMAND ----------

# MAGIC %md
# MAGIC ################################

# COMMAND ----------

print(f"X one hot shape: {X_one_hot.shape}")
print(f"W shape: {W.shape}")

# COMMAND ----------

epochs = 1

nlls = torch.zeros(5) 
for epoch in range(epochs):

    for idx in range(5):
        x, y = X[idx].item(), ytarget[idx].item()
        print("_"*20)
        print(f"bigram example: {itos[x]}{itos[y]} (indexes: {x}, {y})")
        print(f"input to the neural net: {x}")
        print(f"output probs from the neural net: {probs[idx]}")
        p = probs[idx, y]
        print(f"probabilty assigned by NN to correct character: {p}")
        logp = torch.log(p)
        print(f"log likelihood: {logp}") 
        nll = -logp
        print(f"negative lof likelihood: {nll}")
        nlls[idx] = nll

print("#"*30)
print(f"LOSS: {nlls.mean()}")

# COMMAND ----------

