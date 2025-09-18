import os

import torch
import torch.nn.functional as F
import tiktoken
from collections import OrderedDict


from train import GPT, GPTConfig


if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if device.startswith("cuda") else 'cpu' # for torch.autocast

    model = GPT(GPTConfig())
    model.to(device)

    checkpoint = torch.load(os.path.join('log', f"base_model_{76291:05d}_v1.pt"), map_location=device, weights_only=False)
    state_dict = checkpoint['model']

    # remove the "_orig_mod." prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")  # strip prefix
        new_state_dict[new_key] = v

    # load cleaned weights
    model.load_state_dict(new_state_dict)

    enc = tiktoken.get_encoding('gpt2')
    model.eval()
    num_return_sequences = 1
    max_length = 1024
    tokens = enc.encode("Roses are red, violets are blue,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    inf_rng = torch.Generator(device=device).manual_seed(42)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=inf_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"{decoded}")