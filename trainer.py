import os
import sys
import time
import math
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from config import DensyConfig
from model import DensyModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("densy.trainer")


class CosineScheduler:
    def __init__(self, optimizer, lr_max, lr_min, warmup_tokens, total_tokens):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_tokens = warmup_tokens
        self.total_tokens = total_tokens
        self._seen = 0

    def step(self, n_tokens: int) -> float:
        self._seen += n_tokens
        t = self._seen
        if t < self.warmup_tokens:
            lr = self.lr_max * t / max(1, self.warmup_tokens)
        else:
            p = (t - self.warmup_tokens) / max(1, self.total_tokens - self.warmup_tokens)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * min(p, 1.0)))
        for g in self.optimizer.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)
        return lr


def build_optimizer(model: nn.Module, lr_max: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "layernorm" in name.lower() or "ln" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay,    "weight_decay": weight_decay, "lr_mult": 1.0},
        {"params": no_decay, "weight_decay": 0.0,          "lr_mult": 1.0},
    ]
    try:
        opt = torch.optim.AdamW(groups, lr=lr_max, betas=(0.9, 0.95), eps=1e-8, fused=True)
        logger.info("AdamW fused=True")
    except Exception:
        opt = torch.optim.AdamW(groups, lr=lr_max, betas=(0.9, 0.95), eps=1e-8)
        logger.info("AdamW standard")
    return opt


class Trainer:
    def __init__(
        self,
        model: DensyModel,
        cfg: DensyConfig,
        rank: int,
        world_size: int,
        local_rank: int,
        output_dir: str,
        total_tokens: int,
        global_batch_tokens: int,
        micro_batch_per_gpu: int,
        seq_len: int,
        lr_max: float = 3e-4,
        lr_min: float = 3e-5,
        warmup_ratio: float = 0.02,
        grad_clip: float = 1.0,
        weight_decay: float = 0.1,
        bf16: bool = True,
        use_compile: bool = True,
        save_every_steps: int = 500,
        resume_from: Optional[str] = None,
        hourly_cost: float = 12.0,
    ):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main = rank == 0
        self.device = torch.device(f"cuda:{local_rank}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.total_tokens = total_tokens
        self.global_batch_tokens = global_batch_tokens
        self.micro_batch_per_gpu = micro_batch_per_gpu
        self.seq_len = seq_len
        self.grad_clip = grad_clip
        self.bf16 = bf16
        self.save_every_steps = save_every_steps
        self.hourly_cost = hourly_cost
        self.global_step = 0
        self.tokens_seen = 0
        self._start_time = time.time()

        self.tokens_per_step = global_batch_tokens
        self.ga_steps = global_batch_tokens // (micro_batch_per_gpu * seq_len * world_size)
        assert self.ga_steps >= 1, "global_batch_tokens too small for given micro_batch/seq_len/world_size"

        model = model.to(self.device)
        if bf16:
            model = model.to(torch.bfloat16)

        if use_compile:
            logger.info("Compiling model with torch.compile max-autotune...")
            model = torch.compile(model, mode="max-autotune")

        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

        self.model = model
        self.raw_model = model.module if hasattr(model, "module") else model

        self.optimizer = build_optimizer(self.raw_model, lr_max, weight_decay)
        warmup_tokens = int(total_tokens * warmup_ratio)
        self.scheduler = CosineScheduler(self.optimizer, lr_max, lr_min, warmup_tokens, total_tokens)

        if resume_from:
            self._load_checkpoint(resume_from)

        total_params = sum(p.numel() for p in self.raw_model.parameters())
        if self.is_main:
            logger.info(f"Densy Trainer | {total_params/1e6:.1f}M params | {world_size} GPUs | ga_steps={self.ga_steps}")
            logger.info(f"Target: {total_tokens/1e9:.1f}B tokens | {total_tokens // global_batch_tokens} optimizer steps")

    def _autocast(self):
        if self.bf16:
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return torch.amp.autocast("cuda", enabled=False)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.raw_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt.get("global_step", 0)
        self.tokens_seen = ckpt.get("tokens_seen", 0)
        self.scheduler._seen = self.tokens_seen
        if self.is_main:
            logger.info(f"Resumed from {path} at step {self.global_step} ({self.tokens_seen/1e9:.2f}B tokens)")

    def _save_checkpoint(self, tag: str):
        if not self.is_main:
            return
        ckpt_dir = self.output_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        raw = self.raw_model
        torch.save({
            "model": raw.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
        }, ckpt_dir / "checkpoint.pt")
        bf16_sd = {k: v.to(torch.bfloat16) for k, v in raw.state_dict().items()}
        torch.save(bf16_sd, ckpt_dir / "pytorch_model.bin")
        raw.cfg.to_json(str(ckpt_dir / "model_config.json"))
        logger.info(f"Checkpoint saved: {ckpt_dir}")

    def train(self, dataloader: DataLoader):
        self.model.train()
        total_steps = self.total_tokens // self.global_batch_tokens
        step_tokens = self.ga_steps * self.micro_batch_per_gpu * self.seq_len * self.world_size

        step_start = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        accum_loss = 0.0

        for batch in dataloader:
            if self.tokens_seen >= self.total_tokens:
                break

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            labels = torch.where(labels == self.raw_model.cfg.pad_token_id, torch.full_like(labels, -100), labels)

            is_last_micro = (micro_step == self.ga_steps - 1)
            ctx = self.model.no_sync() if (self.world_size > 1 and not is_last_micro) else torch.contextlib.nullcontext() if hasattr(torch, "contextlib") else __import__("contextlib").nullcontext()

            with ctx:
                with self._autocast():
                    out = self.model(input_ids=input_ids, labels=labels)
                    loss = out["loss"] / self.ga_steps
                loss.backward()

            accum_loss += loss.item()
            micro_step += 1

            if micro_step == self.ga_steps:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.grad_clip)
                lr = self.scheduler.step(step_tokens)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1
                self.tokens_seen += step_tokens

                if self.is_main and self.global_step % 10 == 0:
                    elapsed = time.perf_counter() - step_start
                    tok_per_sec = step_tokens * 10 / elapsed
                    cost = (time.time() - self._start_time) / 3600 * self.hourly_cost
                    eta_h = (self.total_tokens - self.tokens_seen) / max(tok_per_sec, 1) / 3600
                    logger.info(
                        f"Step {self.global_step}/{total_steps} | "
                        f"Loss {accum_loss*self.ga_steps:.4f} | "
                        f"LR {lr:.2e} | GradNorm {grad_norm:.3f} | "
                        f"tok/s {tok_per_sec:,.0f} | "
                        f"Seen {self.tokens_seen/1e9:.2f}B/{self.total_tokens/1e9:.0f}B | "
                        f"ETA {eta_h:.1f}h | Cost ${cost:.2f}"
                    )
                    step_start = time.perf_counter()

                if self.global_step % self.save_every_steps == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

                accum_loss = 0.0
                micro_step = 0

        if self.world_size > 1:
            dist.barrier()
        self._save_checkpoint("final")
        if self.is_main:
            logger.info(f"Training complete. {self.tokens_seen/1e9:.2f}B tokens processed.")
