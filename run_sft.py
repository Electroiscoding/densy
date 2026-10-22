#!/usr/bin/env python3
import os
import sys
import time
import math
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from config import DensyConfig
from model import build_densy_500m
from trainer import build_optimizer, CosineScheduler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_sft")


class PackedSFTDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int, max_samples: Optional[int] = None, seed: int = 42):
        self.seq_len = seq_len
        npz_files = sorted(Path(data_dir).glob("**/*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")
        all_ids, all_masks = [], []
        for f in npz_files:
            try:
                data = np.load(str(f))
                ids = data["input_ids"]
                mask = data["loss_mask"]
                if ids.shape[1] == seq_len:
                    all_ids.append(ids)
                    all_masks.append(mask)
            except Exception as e:
                logger.warning(f"Error loading {f}: {e}")
        self.input_ids = np.concatenate(all_ids, axis=0)
        self.loss_mask = np.concatenate(all_masks, axis=0)
        if max_samples and max_samples < len(self.input_ids):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(self.input_ids), max_samples, replace=False)
            self.input_ids = self.input_ids[idx]
            self.loss_mask = self.loss_mask[idx]
        logger.info(f"PackedSFTDataset: {len(self.input_ids):,} samples, loss coverage: {self.loss_mask.mean():.1%}")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.input_ids[idx].astype(np.int64)
        mask = self.loss_mask[idx]
        input_ids = torch.from_numpy(ids)
        labels = input_ids.clone()
        labels[mask == 0] = -100
        return {"input_ids": input_ids, "labels": labels}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",          required=True)
    p.add_argument("--data_dir",            required=True)
    p.add_argument("--output_dir",          default="./checkpoints/densy_sft")
    p.add_argument("--total_tokens",        type=int,   default=20_000_000)
    p.add_argument("--seq_len",             type=int,   default=4096)
    p.add_argument("--global_batch_tokens", type=int,   default=131072)
    p.add_argument("--micro_batch_per_gpu", type=int,   default=2)
    p.add_argument("--lr_max",              type=float, default=2e-5)
    p.add_argument("--lr_min",              type=float, default=2e-6)
    p.add_argument("--warmup_ratio",        type=float, default=0.03)
    p.add_argument("--grad_clip",           type=float, default=0.5)
    p.add_argument("--weight_decay",        type=float, default=0.01)
    p.add_argument("--bf16",                action="store_true", default=True)
    p.add_argument("--no_compile",          action="store_true")
    p.add_argument("--save_every_steps",    type=int,   default=50)
    p.add_argument("--num_workers",         type=int,   default=4)
    return p.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0
    base_dir = Path(args.base_model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = DensyConfig.from_json(str(base_dir / "model_config.json")) if (base_dir / "model_config.json").exists() else DensyConfig()
    cfg.seq_len = args.seq_len
    model = build_densy_500m(cfg)

    for fn in ("pytorch_model.bin", "model_bf16.pt", "model.pt", "checkpoint.pt"):
        wp = base_dir / fn
        if wp.exists():
            sd = torch.load(str(wp), map_location="cpu")
            if "model" in sd:
                sd = sd["model"]
            model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
            if is_main:
                logger.info(f"Loaded weights from {wp}")
            break

    model = model.to(device)
    if args.bf16:
        model = model.to(torch.bfloat16)
    if not args.no_compile:
        model = torch.compile(model, mode="max-autotune")
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    raw_model = model.module if hasattr(model, "module") else model
    dataset = PackedSFTDataset(args.data_dir, seq_len=args.seq_len)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_per_gpu,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    ga_steps = max(1, args.global_batch_tokens // (args.micro_batch_per_gpu * args.seq_len * world_size))
    step_tokens = ga_steps * args.micro_batch_per_gpu * args.seq_len * world_size
    total_steps = args.total_tokens // step_tokens

    optimizer = build_optimizer(raw_model, args.lr_max, args.weight_decay)
    warmup_tokens = int(args.total_tokens * args.warmup_ratio)
    scheduler = CosineScheduler(optimizer, args.lr_max, args.lr_min, warmup_tokens, args.total_tokens)

    if is_main:
        logger.info(f"Densy SFT | {world_size} GPUs | ga_steps={ga_steps} | total_steps={total_steps}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    global_step = 0
    tokens_seen = 0
    accum_loss = 0.0
    start_t = time.perf_counter()

    for batch in loader:
        if tokens_seen >= args.total_tokens:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        is_last = (micro_step == ga_steps - 1)
        ctx = model.no_sync() if (world_size > 1 and not is_last) else torch.nullcontext() if hasattr(torch, "nullcontext") else __import__("contextlib").nullcontext()

        with ctx:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                out = model(input_ids, labels=labels)
                loss = out["loss"] / ga_steps
            loss.backward()

        accum_loss += loss.item()
        micro_step += 1

        if micro_step == ga_steps:
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            lr = scheduler.step(step_tokens)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            tokens_seen += step_tokens

            if is_main and global_step % 10 == 0:
                elapsed = time.perf_counter() - start_t
                tok_s = step_tokens * 10 / max(elapsed, 1e-5)
                logger.info(f"SFT Step {global_step}/{total_steps} | Loss {accum_loss*ga_steps:.4f} | LR {lr:.2e} | Norm {grad_norm:.3f} | tok/s {tok_s:,.0f}")
                start_t = time.perf_counter()

            if global_step % args.save_every_steps == 0 and is_main:
                ckpt_dir = output_dir / f"sft_step{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                raw = raw_model
                bf16_sd = {k: v.to(torch.bfloat16) for k, v in raw.state_dict().items()}
                torch.save(bf16_sd, ckpt_dir / "pytorch_model.bin")
                raw.cfg.to_json(str(ckpt_dir / "model_config.json"))
                logger.info(f"Saved SFT checkpoint: {ckpt_dir}")

            accum_loss = 0.0
            micro_step = 0

    if is_main:
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        raw = raw_model
        bf16_sd = {k: v.to(torch.bfloat16) for k, v in raw.state_dict().items()}
        torch.save(bf16_sd, final_dir / "pytorch_model.bin")
        torch.save(bf16_sd, final_dir / "model_bf16.pt")
        raw.cfg.to_json(str(final_dir / "model_config.json"))
        logger.info(f"SFT complete. Model exported to {final_dir}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
