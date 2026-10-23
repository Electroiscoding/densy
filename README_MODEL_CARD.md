---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- causal-lm
- dense-transformer
- gpt-style
- anthropic-style
- sft
- dpo
- h100
- runpod
- chat
base_model: Phase-Technologies/densy-330m-pretrain
---

# Densy-330M-Chat: Pure Dense Causal Transformer Architecture

<div align="center">

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Parameters](https://img.shields.io/badge/Parameters-328.5M%20Total%20%7C%20Dense-green.svg)]()
[![Hardware](https://img.shields.io/badge/Trained%20On-4%C3%97%20NVIDIA%20H100%2080GB%20SXM5-76B900.svg?logo=nvidia)]()
[![Tokens](https://img.shields.io/badge/Pretrained%20On-15.0%20Billion%20Tokens-purple.svg)]()
[![Alignment](https://img.shields.io/badge/Alignment-SFT%20%2B%20DPO-orange.svg)]()

</div>

---

## ⚡ Executive Summary

**Densy-330M-Chat** is an open-source, pure dense causal transformer language model designed and trained from scratch by **Phase Technologies**. Unlike recent hybrid or mixture-of-experts architectures, Densy is built on the classic, battle-tested dense Anthropic/GPT foundation:

* **Zero MoE routing overhead**: 100% of the parameters participate in every forward and backward pass.
* **Rotary Position Embeddings (RoPE)** with FlashAttention / Scaled Dot-Product Attention (SDPA).
* **Weight Tying**: Shared embeddings between input token representations and the output projection head, saving memory and accelerating vocabulary learning across a massive **206,464-token** vocabulary.
* **End-to-End Pipeline**: Pretrained on **15.00 Billion tokens** from raw scratch, fine-tuned with **Supervised Fine-Tuning (SFT)** on dialogue traces with loss masking, and polished using **Direct Preference Optimization (DPO)**.

---

## 📐 Model Architecture Specifications

| Specification | Value | Technical Detail |
| :--- | :--- | :--- |
| **Total Parameters** | **328,508,928 (328.5M)** | Exact parameter count (Storage: ~657 MB in BF16) |
| **Active Parameters** | **328.5M (100%)** | Fully dense architecture (no sparsity, no MoE routing) |
| **Hidden Dimension ($d_{\text{model}}$)** | **768** | Intermediate hidden state representation |
| **Transformer Layers ($n_{\text{layers}}$)** | **24** | 24 identical Pre-LayerNorm Transformer blocks |
| **Attention Heads ($n_{\text{heads}}$)** | **12** | 12 heads, 64-dim per head ($d_{\text{head}} = 64$) |
| **MLP Expansion Dimension** | **3,072** | $4 \times d_{\text{model}}$ with GeLU (tanh approximation) |
| **Context Window ($seq\_len$)** | **4,096 tokens** | Scalable rotary position context |
| **Vocabulary Size** | **206,464** | CRAYON Turbo BPE vocabulary |
| **Weight Tying** | **Enabled** | Embedding matrix tied to LM Head ($206{,}464 \times 768 = 158.6\text{M}$ weights) |

```
                       Input Tokens (B, T)
                                │
                                ▼
               [Token Embedding (206,464 × 768, Tied)]
                                │
          ┌─────────────────────┴─────────────────────┐
          │  Block 0 .. 23 (24 Identical Layers):     │
          │   ├─ Pre-LayerNorm (d=768)                │
          │   ├─ Multi-Head Attention (12 Heads, RoPE)│
          │   │   └─ FlashAttention-2 / SDPA          │
          │   ├─ Residual Add                         │
          │   ├─ Pre-LayerNorm (d=768)                │
          │   ├─ Dense MLP (768 → 3072 → 768, GeLU)   │
          │   └─ Residual Add                         │
          └─────────────────────┬─────────────────────┘
                                ▼
                       [Final LayerNorm (768)]
                                │
                                ▼
                 [Tied LM Projection Head (206,464)]
                                │
                                ▼
                          Output Logits
```

---

## 🛠️ Complete Training Chronology

The model was developed, pretrained, and aligned directly on high-performance cloud hardware:

### 1. Pretraining (15.00 Billion Tokens)
* **Compute Cluster**: 4× NVIDIA H100 80GB HBM3 (SXM5) connected via NVLink on RunPod.
* **Global Batch Size**: 4,194,304 tokens per step (Sequence length: 4,096; Micro-batch per GPU: 8; Gradient accumulation: 32).
* **Hardware Throughput**: **~582,000 tokens/second** sustained across the cluster.
* **Total Steps**: 3,576 optimizer steps.
* **Optimizer**: Fused AdamW ($\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}$, weight decay 0.1) with Cosine Learning Rate Decay ($3\times 10^{-4} \to 3\times 10^{-5}$).
* **Tokens Seen**: Exactly **15,000,000,000 tokens** from pretokenized web, code, and academic text shards (`Phase-Technologies/forge-3b-pretrain-data`).

### 2. Supervised Fine-Tuning (SFT)
* **Dataset**: `Phase-Technologies/forge-3b-sft-data` (Multi-turn conversational dialogue with user/assistant role separation).
* **Objective**: Masked cross-entropy (loss computed strictly on assistant response tokens, prompt tokens masked with `-100`).
* **Volume**: 20,000,000 tokens (76 optimizer steps on 4× H100, completed in ~45 seconds).

### 3. Direct Preference Optimization (DPO)
* **Dataset**: `Phase-Technologies/forge-3b-dpo-data` (1,000 high-margin chosen/rejected pairs).
* **Objective**: Reference-anchored implicit reward optimization ($\beta=0.1$) aligning helpfulness, reasoning clarity, and harmlessness.

---

## 💻 Google Colab & Local Quickstart

You can run Densy-330M-Chat on any Google Colab instance (even free T4 GPU or CPU) with zero external setup:

```python
# Install lightweight dependencies
!pip install -q torch huggingface_hub xerv-crayon safetensors

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# 1. Download model files
repo_id = "Phase-Technologies/Densy-330M-Chat"
model_file = hf_hub_download(repo_id=repo_id, filename="model.py")
config_file = hf_hub_download(repo_id=repo_id, filename="config.py")
tok_file = hf_hub_download(repo_id=repo_id, filename="tokenizer/crayon_wrapper.py")
weights_file = hf_hub_download(repo_id=repo_id, filename="model.safetensors")

# 2. Import architecture
import sys
sys.path.insert(0, ".")
from config import DensyConfig
from model import build_densy_330m
from tokenizer.crayon_wrapper import ForgeTokenizer

# 3. Instantiate and load
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

tokenizer = ForgeTokenizer()
cfg = DensyConfig()
model = build_densy_330m(cfg)
sd = load_file(weights_file)
model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
model = model.to(device).to(dtype).eval()

# 4. Generate Chat Response
prompt = "<|bos|><|usr|>What are three foundational principles of physics?<|asst|>"
input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=False, add_eos=False)], device=device)

with torch.inference_mode():
    output_ids = model.generate(input_ids, max_new_tokens=64, temperature=0.7)

response = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False)
print(response)
```

---

## 🏷️ Chat Prompt Template & Special Tokens

| Token | Representation | Purpose |
| :--- | :--- | :--- |
| `<|bos|>` | `Token 1` | Beginning of sequence |
| `<|eos|>` | `Token 2` | End of sequence / completion termination |
| `<|usr|>` | `Token 4` | User conversational turn delimiter |
| `<|asst|>` | `Token 5` | Assistant response turn delimiter |
| `<|pad|>` | `Token 0` | Sequence padding |

Format:
```text
<|bos|><|usr|>Your question here<|asst|>Assistant response...<|eos|>
```

---

## a note from the creator.....

honestly when we set out to build densy we wanted to prove that pure, classic dense transformer architecture still has unmatched elegance and power when pushed to its limits..... after wrestling with complex mixture-of-experts routing and hybrid state space layers in our previous models, densy was our return to first principles: no gate networks, no expert balancing losses, no quadratic memory tricks—just 24 deep, pure dense layers weight-tied to a massive 206k vocabulary running wide open on 4x nvidia h100 sxm5 gpus at over 580,000 tokens every single second.....

watching the loss drop from 12 down across 15 billion tokens in runpod's datacenter was an unbelievable feeling..... training independent frontier-style models with limited personal budgets is terrifying and exhilarating all at once..... every single dollar counted, every step mattered, and when little bugs cropped up in our loss shifts or dataloaders we had to diagnose them in real time on live silicon with the clock ticking.....

huge love and endless gratitude to the runpod team for building the infrastructure that lets independent developers spin up multi-h100 nodes with a single click and actually train real models from scratch..... this journey has been exhausting, beautiful, and deeply humbling..... to everyone following along, experimenting with our models, or building their own architectures in bedrooms and dorms: never let compute monopolies convince you that you can't train from scratch..... i hope you enjoy talking with densy..... thank you for believing in this crazy dream.....

— soham pal (phase technologies)
