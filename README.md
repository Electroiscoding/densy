# Densy

**Densy** is a pure dense (non-LLaMA, Anthropic/GPT-style) causal language model architecture built from first principles for maximum hardware FLOPs utilization (MFU).

---

## Architecture Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | **514M** (`0.514B`) | 100% dense (every parameter active on every token) |
| **Layers** | **24** | Full pre-LN Transformer blocks |
| **Hidden Dimension ($d_{\text{model}}$)** | **1024** | Model width |
| **Attention Heads** | **16** | Head dimension 64 |
| **Attention Backend** | **SDPA** | PyTorch native fused FlashAttention-2 / FlashAttention-3 |
| **Positional Encoding** | **RoPE** | Rotary position embeddings with cached frequencies |
| **FFN / MLP** | **Dense GeLU** | 2 matmuls, $4\times$ expansion ($1024 \to 4096 \to 1024$) |
| **Normalization** | **LayerNorm** | Pre-LN architecture (unconditionally stable convergence) |
| **Vocabulary Size** | **206,464** | Multi-lingual & code vocabulary (`crayon_vocab`) |
| **Weight Tying** | **Enabled** | LM head shares weights with token embeddings |
| **Max Sequence Length** | **4,096 tokens** | Pretraining context window |

---

## Why Pure Dense Over Hybrid / MoE?

1. **Maximum Hardware Saturation:** 99%+ of compute operations are standard GEMM (General Matrix Multiply), saturating NVIDIA Tensor Cores at peak theoretical limits.
2. **High MFU:** Achieves 55–65%+ Model FLOPs Utilization (MFU) on A100/H100 compared to 25–30% on memory-bandwidth-bound recurrent/MoE models.
3. **Zero Kernel Overhead:** Zero custom CUDA/Triton compilation stalls, zero state scanning bottlenecks, and 100% full-graph `torch.compile(mode="max-autotune")` compatibility.
4. **Instant Dataset Compatibility:** Uses the exact 206,464 vocabulary and `.npy` pre-tokenized shards directly without re-tokenizing.

---

## Pretraining Launch Commands

### 8× A100 / 8× H100 (Multi-GPU DDP)
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=8 run_pretrain.py \
  --data_dir /path/to/pretrain_data \
  --output_dir ./checkpoints/densy_500m \
  --total_tokens 15000000000 \
  --seq_len 4096 \
  --global_batch_tokens 4194304 \
  --micro_batch_per_gpu 8 \
  --lr_max 3e-4 \
  --lr_min 3e-5 \
  --warmup_ratio 0.02 \
  --grad_clip 1.0 \
  --bf16 \
  --save_every_steps 500
```

### Single GPU (Local / Test Run)
```bash
python run_pretrain.py \
  --data_dir /path/to/pretrain_data \
  --output_dir ./checkpoints/densy_500m \
  --total_tokens 100000000 \
  --seq_len 2048 \
  --global_batch_tokens 131072 \
  --micro_batch_per_gpu 2 \
  --bf16 \
  --save_every_steps 100
```

---

## Inference

### 1-Click CLI
```bash
python run_inference.py \
  --model_path ./checkpoints/densy_500m/final \
  --prompt "<|bos|><|usr|>What is machine learning?<|asst|>" \
  --max_new_tokens 128 \
  --temperature 0.8 \
  --top_p 0.9
```

### Python API
```python
import sys, torch
from pathlib import Path

repo_dir = Path("./")
sys.path.insert(0, str(repo_dir))

from config import DensyConfig
from model import build_densy_500m
from tokenizer.crayon_wrapper import ForgeTokenizer

tokenizer = ForgeTokenizer.from_pretrained(str(repo_dir / "tokenizer"))
cfg = DensyConfig.from_json(str(repo_dir / "model_config.json"))
model = build_densy_500m(cfg)

weights = torch.load(repo_dir / "pytorch_model.bin", map_location="cpu")
model.load_state_dict({k: v.to(torch.bfloat16) for k, v in weights.items()})
model = model.to("cuda" if torch.cuda.is_available() else "cpu").to(torch.bfloat16).eval()

prompt = "<|bos|><|usr|>Explain how neural networks learn.<|asst|>"
input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=False, add_eos=False)], device=model.device)

with torch.inference_mode():
    output_ids = model.generate(input_ids=input_ids, max_new_tokens=100, temperature=0.7)

print(tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False))
```
