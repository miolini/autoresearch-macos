"""
Chat-style text generation from the model saved by train.py.
Usage: uv run generate.py
"""
import os
import torch
import torch.nn.functional as F

# Reuse the model definition from train.py without running training
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")).read()
exec(src.split("# Hyperparameters")[0])

device = torch.device("mps")
ckpt = torch.load(os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "model.pt"), map_location=device)
config = GPTConfig(**ckpt["config"])
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
model.load_state_dict(ckpt["state_dict"])
model.eval()
tokenizer = Tokenizer.from_directory()

@torch.no_grad()
def generate(prompt, max_new_tokens=150, temperature=0.8, top_k=50):
    ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
    idx = torch.tensor([ids], device=device)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -config.sequence_len:])[:, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
    return tokenizer.decode(idx[0, len(ids):].tolist())

print("Tiny 11.5M-param model trained for 5 minutes. It continues text; it is not a chatbot.")
print("Type the start of a sentence and press Enter (Ctrl+C to quit).\n")
while True:
    try:
        prompt = input("> ")
    except (EOFError, KeyboardInterrupt):
        break
    print(prompt + generate(prompt) + "\n")
