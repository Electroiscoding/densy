# Densy (330M)

**Densy** is a pure dense (non-LLaMA, Anthropic/GPT-style) causal language model architecture built from first principles for maximum hardware FLOPs utilization (MFU).

---

## Architecture Specifications (Combo 3: 330M)

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | **328.5M** (`0.329B`) | 100% dense (every parameter active on every token) |
| **Reasoning Parameters** | **170.0M** (**51.7%**) | Active transformer blocks (MHA + GeLU MLP) |
| **Embedding Parameters** | **158.6M** (**48.3%**) | Weight-tied to output LM head |
| **Layers** | **24** | Full pre-LN Transformer blocks |
| **Hidden Dimension ($d_{\text{model}}$)** | **768** | Model width |
| **Attention Heads** | **12** | Head dimension 64 |
| **Attention Backend** | **SDPA** | PyTorch native fused FlashAttention-2 / FlashAttention-3 |
| **Positional Encoding** | **RoPE** | Rotary position embeddings with cached frequencies |
| **FFN / MLP** | **Dense GeLU** | 2 matmuls, $4\times$ expansion ($768 \to 3072 \to 768$) |
| **Normalization** | **LayerNorm** | Pre-LN architecture (unconditionally stable convergence) |
| **Vocabulary Size** | **206,464** | Multi-lingual & code vocabulary (`crayon_vocab`) |
| **Weight Tying** | **Enabled** | LM head shares weights with token embeddings |
| **Max Sequence Length** | **4,096 tokens** | Pretraining context window |

---

## Why Combo 3 (330M)?

1. **Reasoning Exceeds Dictionary:** Unlike sub-200M models where 70%+ is just embeddings, Densy-330M dedicates **51.7% of its capacity directly to transformer thinking layers**.
2. **Speed & Efficiency:** Reaches **~700,000 tokens/sec** on 4× H100 SXM, chewing through all **15 Billion tokens in ~5.5 hours**.
3. **Budget Friendly:** Full 15B pretraining run costs **~$88 – $95 total** on 4× H100 SXM.
4. **Zero Overhead:** 100% standard GEMM operations, FlashAttention-2/3 fused, and full `torch.compile(mode="max-autotune")` support.
5. **Direct Data Compatibility:** Works directly with your existing 206,464 vocabulary and `.npy` pre-tokenized shards — **0 seconds wasted re-tokenizing**.

---

## 1. Pretraining (to 15B Tokens)

### 4× H100 SXM (Recommended: ~5.5 hours, ~$88)
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=4 run_pretrain.py \
  --data_dir /path/to/pretrain_data \
  --output_dir ./checkpoints/densy_330m \
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

### 2× H100 SXM (~11 hours, ~$88)
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=2 run_pretrain.py \
  --data_dir /path/to/pretrain_data \
  --output_dir ./checkpoints/densy_330m \
  --total_tokens 15000000000 \
  --seq_len 4096 \
  --global_batch_tokens 2097152 \
  --micro_batch_per_gpu 8 \
  --lr_max 3e-4 \
  --lr_min 3e-5 \
  --warmup_ratio 0.02 \
  --grad_clip 1.0 \
  --bf16 \
  --save_every_steps 500
```

---

## 2. Supervised Fine-Tuning (SFT)

```bash
torchrun --nproc_per_node=4 run_sft.py \
  --base_model ./checkpoints/densy_330m/final \
  --data_dir /path/to/sft_data \
  --output_dir ./checkpoints/densy_sft \
  --total_tokens 20000000 \
  --seq_len 4096 \
  --global_batch_tokens 131072 \
  --micro_batch_per_gpu 2 \
  --lr_max 2e-5 \
  --lr_min 2e-6 \
  --bf16 \
  --save_every_steps 50
```

---

## 3. Direct Preference Optimization (DPO)

```bash
torchrun --nproc_per_node=4 run_dpo.py \
  --base_model ./checkpoints/densy_sft/final \
  --data_dir /path/to/dpo_data \
  --output_dir ./checkpoints/densy_dpo \
  --beta 0.1 \
  --batch_pairs 1 \
  --ga_steps 4 \
  --seq_len 2048 \
  --max_steps 40 \
  --max_pairs 1000 \
  --lr 5e-7 \
  --bf16 \
  --save_every_steps 20
```

---

## 4. Inference

### CLI
```bash
python run_inference.py \
  --model_path ./checkpoints/densy_330m/final \
  --prompt "<|bos|><|usr|>What is artificial intelligence?<|asst|>" \
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
from model import build_densy_330m
from tokenizer.crayon_wrapper import ForgeTokenizer

tokenizer = ForgeTokenizer.from_pretrained(str(repo_dir / "tokenizer"))
cfg = DensyConfig.from_json(str(repo_dir / "model_config.json"))
model = build_densy_330m(cfg)

weights = torch.load(repo_dir / "pytorch_model.bin", map_location="cpu")
model.load_state_dict({k: v.to(torch.bfloat16) for k, v in weights.items()})
model = model.to("cuda" if torch.cuda.is_available() else "cpu").to(torch.bfloat16).eval()

prompt = "<|bos|><|usr|>Explain how neural networks learn.<|asst|>"
input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=False, add_eos=False)], device=model.device)

with torch.inference_mode():
    output_ids = model.generate(input_ids=input_ids, max_new_tokens=100, temperature=0.7)

print(tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False))
```
