"""
TUI Chat interface for testing the trained model.
Usage: uv run chat.py [--checkpoint PATH]

This script loads the model and tokenizer, then provides a simple chat interface.
"""

import os
import sys
import argparse
import pickle

# Verify macOS environment
def verify_macos_env():
    if sys.platform != "darwin":
        raise RuntimeError(f"This script requires macOS with Metal. Detected platform: {sys.platform}")
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Metal Performance Shaders) is not available. Ensure you are running on Apple Silicon.")
    print("Environment verified: macOS with MPS available.")

verify_macos_env()

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# Model classes (copied from train.py to avoid importing training code)
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Expand heads for KV based on GQA
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        
        # Transpose to [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply mask for sliding window
        window = window_size[0]
        if window > 0 and window < T:
            mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
            mask = mask.triu(diagonal=1 - window)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Placeholder - not needed for inference
        pass

    @torch.no_grad()
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits


# ---------------------------------------------------------------------------
# Tokenizer wrapper
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper."""
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token("<|reserved_0|>")

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


# ---------------------------------------------------------------------------
# Chat functions
# ---------------------------------------------------------------------------

def find_latest_checkpoint(checkpoints_dir="checkpoints"):
    """Find the most recent checkpoint in the directory."""
    if not os.path.exists(checkpoints_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")]
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoints_dir, f)))
    return os.path.join(checkpoints_dir, checkpoints[-1])


def generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_p=0.9):
    """Generate a response from the model."""
    device = next(model.parameters()).device
    
    # Encode the prompt
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            logits = model(idx)
            logits = logits[:, -1, :] / temperature
            
            # Simple top-k sampling for reliability
            top_k = min(50, logits.size(-1))
            top_probs, top_indices = torch.topk(logits, top_k)
            probs = torch.zeros_like(logits).scatter_(-1, top_indices, torch.softmax(top_probs, dim=-1))
            
            # Sample from the filtered distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat([idx, next_token], dim=1)
            
            # Stop if we hit EOS (or a reserved token)
            if next_token.item() >= tokenizer.get_vocab_size() - 4:
                break
    
    # Decode the response (skip the prompt)
    response_tokens = idx[0][len(tokens):].tolist()
    return tokenizer.decode(response_tokens)


def chat_loop(model, tokenizer, args):
    """Main chat loop with simple TUI."""
    print("\n" + "="*50)
    print("  AutoResearch Chat Interface")
    print("  Type 'quit' or 'exit' to stop")
    print("  Type 'reset' to clear conversation")
    print(f"  Generation: max_tokens={args.max_new_tokens}, temp={args.temperature}, top_p={args.top_p}")
    print("="*50 + "\n")
    
    conversation = []
    
    while True:
        try:
            user_input = input("You> ").strip()
        except EOFError:
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'reset':
            conversation = []
            print("[Conversation reset]")
            continue
        
        # Add to conversation
        conversation.append(f"User: {user_input}")
        
        # Build context from conversation (last few turns)
        context = "\n".join(conversation[-4:])  # Last 4 messages
        prompt = context + "\nAssistant:"
        
        # Generate response
        print("Assistant> ", end="", flush=True)
        response = generate_response(
            model, tokenizer, prompt, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(response)
        
        conversation.append(f"Assistant: {response}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Chat with the trained model")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint file (default: latest in checkpoints/)")
    parser.add_argument("--max-new-tokens", type=int, default=150,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling threshold")
    args = parser.parse_args()
    
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("Error: No checkpoint found!")
            print("Run training first to create checkpoints, or specify --checkpoint PATH")
            sys.exit(1)
    
    # Detect device
    device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_type)
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint['config']
    
    # Build model config
    config = GPTConfig(
        sequence_len=config_dict['sequence_len'],
        vocab_size=config_dict['vocab_size'],
        n_layer=config_dict['n_layer'],
        n_head=config_dict['n_head'],
        n_kv_head=config_dict['n_kv_head'],
        n_embd=config_dict['n_embd'],
        window_pattern=config_dict['window_pattern'],
    )
    
    # Create model and load weights
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {config.n_layer} layers, {config.n_embd} embd dim")
    print(f"Trained for {checkpoint.get('total_training_time', 0)/3600:.1f} hours")
    if 'val_bpb' in checkpoint:
        print(f"Val bpb: {checkpoint['val_bpb']:.4f}")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_directory()
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    
    # Start chat
    chat_loop(model, tokenizer, args)


if __name__ == "__main__":
    main()
