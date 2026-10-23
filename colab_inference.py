"""
Densy-330M-Chat • Interactive Inference & Google Colab Quickstart
Run in Google Colab (Free T4 GPU or CPU):

!pip install torch huggingface_hub xerv-crayon
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

REPO_ID = "Phase-Technologies/Densy-330M-Chat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

print(f"🚀 Initializing Densy-330M-Chat on {DEVICE} ({DTYPE})...")

try:
    from tokenizer.crayon_wrapper import ForgeTokenizer
    tokenizer = ForgeTokenizer()
except Exception:
    import sys
    sys.path.insert(0, ".")
    from tokenizer.crayon_wrapper import ForgeTokenizer
    tokenizer = ForgeTokenizer()

from config import DensyConfig
from model import build_densy_330m

cfg_path = hf_hub_download(repo_id=REPO_ID, filename="model_config.json")
cfg = DensyConfig.from_json(cfg_path)
model = build_densy_330m(cfg)

try:
    from safetensors.torch import load_file
    st_path = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors")
    sd = load_file(st_path)
except Exception:
    bin_path = hf_hub_download(repo_id=REPO_ID, filename="pytorch_model.bin")
    sd = torch.load(bin_path, map_location="cpu")

sd = {k.replace("_orig_mod.", ""): v.to(DTYPE) for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model = model.to(DEVICE).to(DTYPE).eval()
print("✅ Densy-330M-Chat loaded cleanly and ready to chat!\n")

def chat(prompt: str, max_new_tokens: int = 128, temperature: float = 0.7, repetition_penalty: float = 1.15):
    formatted = f"<|bos|><|usr|>{prompt.strip()}<|asst|>"
    input_ids = torch.tensor([tokenizer.encode(formatted, add_bos=False, add_eos=False)], device=DEVICE)
    
    with torch.inference_mode():
        if DEVICE == "cuda":
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                out = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, repetition_penalty=repetition_penalty)
        else:
            out = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, repetition_penalty=repetition_penalty)
            
    res = tokenizer.decode(out[0].tolist(), skip_special_tokens=False)
    if "<|asst|>" in res:
        res = res.split("<|asst|>")[-1].replace("<|eos|>", "").strip()
    return res

if __name__ == "__main__":
    print("=" * 60)
    print("Type your message below (or type 'quit' to exit):")
    print("=" * 60)
    while True:
        try:
            usr = input("\nUser > ")
            if usr.strip().lower() in ("quit", "exit", "q"):
                break
            ans = chat(usr)
            print(f"Densy > {ans}")
        except KeyboardInterrupt:
            break
