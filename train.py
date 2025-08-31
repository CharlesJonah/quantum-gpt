# Trained in Fine-Web Edu 10 billlion tokens
import sys
import os
import math
import inspect
import time
from dataclasses import dataclass

import torch
import tiktoken
from datasets import load_dataset
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP


# Setup DDP(Distributed Data Parallel)      
# torchrun command sets env vars RANK, LOCAL_RANK and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run
if ddp:
    assert torch.cuda.is_available(), "CUDA is needed for ddp"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f"cuda: {ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    
    # auto detect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        pass

print(f"Using device - {device}")

TOTAL_BATCH_SIZE = 524288 # 2**19  ~0.5M, to get a nice number that is a power of 2
MAX_STEPS = 19073 # 10b tokens divided by 0.5M batch size
MAX_LR = 6e-4 # as per gpt3 paper
MIN_LR = MAX_LR * 0.1 # as per gpt3 paper
WARM_STEPS = 715 # as per gpt3 paper
MICRO_BATCH_SIZE = 16
SEQUENCE_LENGTH = 1024
GRAD_ACCUM_STEPS = TOTAL_BATCH_SIZE // (MICRO_BATCH_SIZE * SEQUENCE_LENGTH * ddp_world_size)

class GPTTrainingUtilities:
    
    @staticmethod
    def get_lr(step):
        # Linear warmup for the warmup steps
        if step < WARM_STEPS:
            return MAX_LR * (step + 1) / WARM_STEPS
        
        # If step > lr_decay_steps, then return min_lr
        if step > MAX_STEPS:
            return MIN_LR
        
        # in between, use cosine decay down to min learning rate
        decay_ratio = (step - WARM_STEPS) / (MAX_STEPS - WARM_STEPS)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts from 1 and goes to 0
        return MIN_LR + coeff * (MAX_LR - MIN_LR)
    
    @staticmethod
    def load_tokens(filename):
        npt = np.load(filename)
        npt = npt.astype(np.int32) # added after video
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt
    
    @staticmethod
    def download_data():
        local_dir = "edu-fineweb10B"
        remote_name = "sample=10BT"
        shard_size = int(1e8) # 100M tokens per shard, total 100 shards
        
        DATA_CACHE_DIR = os.path.join(os.path.dirname(__name__), local_dir)
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        
        fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split='train')
    
    @staticmethod
    def tokenize():
        # init the tokenizer
        enc = tiktoken.get_encoding('gpt2')
        eot = enc._special_tokens['<|endoftext|>'] # end of text token
        
        
        
        
        
class DataloaderLite:
    def __init__(self, b, t, process_rank, num_of_processes, split):
        self.b = b
        self.t = t
        self.process_rank = process_rank
        self.num_of_processes = num_of_processes
        assert split in {'train', 'val'}
        
        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
    
    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = GPTTrainingUtilities.load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
        
    
    def next_batch(self):
        buff = self.tokens[self.current_position : self.current_position + (self.b * self.t) + 1]
        x = buff[:-1].view(self.b, self.t)
        y = buff[1:].view(self.b, self.t) 
        self.current_position += (self.b * self.t * self.num_of_processes)
        
        # if loading the next batch is out of bounds, reset
        if self.current_position + ((self.b * self.t * self.num_of_processes) + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = GPTTrainingUtilities.load_tokens(self.shards[self.current_shard])
            self.current_position = self.b * self.t * self.process_rank
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
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
    
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
        
# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

assert TOTAL_BATCH_SIZE % (MICRO_BATCH_SIZE * SEQUENCE_LENGTH * ddp_world_size) == 0, "Make sure total batch size is divisible by MICRO_BATCH_SIZE * SEQUENCE_LENGTH * ddp_world_size"
torch.set_float32_matmul_precision('high')
train_loader = DataloaderLite(16, 1024, ddp_rank, ddp_world_size, 'train')

# get logits
model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model  = model.module if ddp else model

#optimize
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(MAX_STEPS): 
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    # Use gradient accumulation since a batch size of 0.5M is too big to fit in our GPUs
    for micro_step in range(GRAD_ACCUM_STEPS):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16): 
            logits, loss = model(x,y) 
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / GRAD_ACCUM_STEPS      
        loss_accum += loss.detach()
        if ddp:
            # this is not the official but it's more sleek, if pytorch ddp api on this paramter changes, this might need a change too
            model.require_backward_grad_sync = (micro_step == GRAD_ACCUM_STEPS - 1)
        loss.backward()
    if ddp:
        all_reduce(loss_accum, op = ReduceOp.AVG)
    norm = clip_grad_norm_(model.parameters(), 1.0)
    
    # determine and set the learning rate for this step
    lr = GPTTrainingUtilities.get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * GRAD_ACCUM_STEPS * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()



sys.exit(0)