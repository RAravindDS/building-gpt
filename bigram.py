import torch 
import torch.nn as nn 
from tqdm.auto import tqdm 
from torch.nn import functional as F 


batch_size = 32 
chunk_size = 8 
max_iters = 1000
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu" 
eval_iters = 200 
eval_interval = 300 


torch.manual_seed(1337) 

with open("input.txt", 'r') as file: 
    data = file.read()

chars = sorted(list(set(data)))
vocab_size = len(chars) 
stoi = { ch:i for i,ch in enumerate(chars)} 
itos = { i:ch for i, ch in enumerate(chars) }

encode = lambda s:  [stoi[c] for c in s]   # take a string output integers
decode = lambda s:  "".join([itos[c] for c in s])  # take a number and output strings


data = torch.tensor(encode(data), dtype = torch.long) 
ninety_percent_data = int((90 / 100) * len(data))  # separating 90% of the data for training 

train_data = data[:ninety_percent_data]
val_data = data[ninety_percent_data:]


def get_batch(split): 
    data = train_data if split == "train" else val_data
    dix = torch.randint( len(data) - chunk_size, (batch_size, ))
    x =  [ data[i: i+chunk_size] for i in dix]
    y = [ data[i+1:i+chunk_size+1] for i in dix]

    return torch.stack(x), torch.stack(y)



class BigramLanguageModel(nn.Module): 
    
    def __init__(self, vocab_size): 
        super().__init__() 
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) 
        self.optimizer = torch.optim.AdamW(self.token_embedding_table.parameters(), lr = 1e-3)

    def forward(self, idx, targets=None): 
        logits = self.token_embedding_table(idx)  
        if targets == None: loss = None 
        else: 
            B, T, C = logits.shape 
            logits = logits.view(B*T, C)  # Before [8, 10, 65] after transforming: [80,65]
            targets = targets.view(-1) # Before [8, 10] after view: [80]
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    

    def generate(self, idx, nr_of_new_tokens): 

        for _ in range(nr_of_new_tokens): 
            logits, loss = self(idx)
            logits = logits[:, -1] # Taking only lenght and channels 
            probs = F.softmax(logits, dim = -1) 
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat( (idx, idx_next), dim = 1) # (B, T + 1)

        return idx 
    



m = BigramLanguageModel(vocab_size=vocab_size)
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-5)  # IN simple words we are going to train the embedding layer 

for steps in tqdm(range(max_iters)): 


    xb, yb = get_batch("train") 
    logits, loss = m(xb, yb) 
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step() 

print(loss.item())
print(decode(m.generate(idx = torch.zeros((1,1), dtype = torch.long), nr_of_new_tokens=800)[0].tolist())) 