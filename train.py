import sys
from dataclasses import dataclass

import torch
import tiktoken
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_


class DataloaderLite:
    def __init__(self, b, t):
        self.b = b
        self.t = t
        
        with open("input.txt", "r") as f:
            text =  f.read()
        enc =  tiktoken.get_encoding("gpt2")
        self.tokens =  torch.tensor(enc.encode(text))
        self.current_position = 0
        print(f"Loaded data = {self.tokens}")
        print(f"One Epoch = {len(self.tokens) // (b * t)} batches")
    
    def next_batch(self):
        buff = self.tokens[self.current_position : self.current_position + (self.b * self.t) + 1]
        x = buff[:-1].view(self.b, self.t)
        y = buff[1:].view(self.b, self.t) 
        self.current_position += (self.b * self.t)
        
        # if loading the next batch is out of bounds, reset
        if self.current_position + ((self.b * self.t) + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
    
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        
        # key, query, value projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANO_GPT_SCALE_UNIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        b,t,c = x.size() # batch_size, sequence length, embedding dimensionality (n_embed)
        # calculate query, key, values for all heads in batch and move head forward to be batch
        # nh is 'number of heads', hs is 'head size' and C = number of channels = nh * hs
        # e.g in GPT-2 (124M), head=12, hs=64, so nh*hs=C=768 channels in transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(b, t,  self.n_head, c // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        v = v.view(b, t,  self.n_head, c // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        q = q.view(b, t,  self.n_head, c // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        
        # self attention implementation with flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # re-assemble all head outputs side by side
        # output projection
        y = y.transpose(1, 2).contiguous().view(b, t, c) 
        y = self.c_proj(y)
        return y
        
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANO_GPT_SCALE_UNIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
         
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)   
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        
        # weight shairing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        # init all weights
        self.apply(self._init_weights)
    
    def forward(self, idx, targets=None):
        _ , t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embed)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embed)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANO_GPT_SCALE_UNIT"):
                std *= (2 * self.config.n_layer) ** - 0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
        
        
        
# auto detect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    pass

print(f"Using device - {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

torch.set_float32_matmul_precision('high')
train_loader = DataloaderLite(16, 1024)

# get logits
model = GPT(GPTConfig())
model.to(device)
torch.compile(model)

#optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16): 
        logits, loss = model(x,y)
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print(f"Step {i} \t Loss = {loss.item()}")



sys.exit(0)