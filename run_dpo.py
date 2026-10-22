#!/usr/bin/env python3
import os
import sys
import time
import math
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from config import DensyConfig
from model import build_densy_500m
from trainer import build_optimizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_dpo")


class PreTokenizedPreferenceDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int, max_samples: Optional[int] = None):
        self.seq_len = seq_len
        npz_files = sorted(Path(data_dir).glob("**/*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")
        self.samples = []
        for f in npz_files:
            try:
                data = np.load(str(f), allow_pickle=True)
                cf, rf = data["chosen_full_ids"], data["rejected_full_ids"]
                cm, rm = data["chosen_loss_mask"], data["rejected_loss_mask"]
                for i in range(len(cf)):
                    self.samples.append((cf[i], rf[i], cm[i], rm[i]))
                    if max_samples and len(self.samples) >= max_samples:
                        break
                if max_samples and len(self.samples) >= max_samples:
                    break
            except Exception as e:
                logger.warning(f"Error loading {f}: {e}")
        logger.info(f"PreTokenizedPreferenceDataset: {len(self.samples):,} pairs loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def _pad_trunc(self, arr, pad_val=0):
        t = torch.tensor(arr[:self.seq_len], dtype=torch.long)
        if len(t) < self.seq_len:
            pad = torch.full((self.seq_len - len(t),), pad_val, dtype=torch.long)
            t = torch.cat([t, pad])
        return t

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cf, rf, cm, rm = self.samples[idx]
        ci = self._pad_trunc(cf, 0)
        ri = self._pad_trunc(rf, 0)
        cl = ci.clone()
        rl = ri.clone()
        c_mask = self._pad_trunc(cm, 0)
        r_mask = self._pad_trunc(rm, 0)
        cl[c_mask == 0] = -100
        rl[r_mask == 0] = -100
        return {"chosen_input_ids": ci, "chosen_labels": cl, "rejected_input_ids": ri, "rejected_labels": rl}


def get_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    B, T_m1, V = shift_logits.shape
    nll = F.cross_entropy(shift_logits.view(-1, V).float(), shift_labels.view(-1), reduction="none", ignore_index=-100).view(B, T_m1)
    mask = (shift_labels != -100)
    n_tokens = mask.float().sum(-1).clamp(min=1.0)
    return (-nll.sum(-1)) / n_tokens


def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta: float = 0.1):
    pi_ratio = pi_c - pi_r
    ref_ratio = ref_c - ref_r
    logits = pi_ratio - ref_ratio
    loss = -F.logsigmoid(beta * logits).mean()
    c_rewards = beta * (pi_c - ref_c).detach()
    r_rewards = beta * (pi_r - ref_r).detach()
    return loss, c_rewards, r_rewards


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",       required=True)
    p.add_argument("--data_dir",         required=True)
    p.add_argument("--output_dir",       default="./checkpoints/densy_dpo")
    p.add_argument("--beta",             type=float, default=0.1)
    p.add_argument("--batch_pairs",      type=int,   default=1)
    p.add_argument("--ga_steps",         type=int,   default=4)
    p.add_argument("--seq_len",          type=int,   default=2048)
    p.add_argument("--max_steps",        type=int,   default=40)
    p.add_argument("--max_pairs",        type=int,   default=1000)
    p.add_argument("--lr",               type=float, default=5e-7)
    p.add_argument("--grad_clip",        type=float, default=0.3)
    p.add_argument("--bf16",             action="store_true", default=True)
    p.add_argument("--no_compile",       action="store_true")
    p.add_argument("--save_every_steps", type=int,   default=20)
    p.add_argument("--num_workers",      type=int,   default=2)
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

    policy = build_densy_500m(cfg)
    ref = build_densy_500m(cfg)

    for fn in ("pytorch_model.bin", "model_bf16.pt", "model.pt"):
        wp = base_dir / fn
        if wp.exists():
            sd = torch.load(str(wp), map_location="cpu")
            if "model" in sd:
                sd = sd["model"]
            sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
            policy.load_state_dict(sd, strict=False)
            ref.load_state_dict(sd, strict=False)
            if is_main:
                logger.info(f"Loaded base weights from {wp}")
            break

    policy = policy.to(device)
    ref = ref.to(device)
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()

    if args.bf16:
        policy = policy.to(torch.bfloat16)
        ref = ref.to(torch.bfloat16)
    if not args.no_compile:
        policy = torch.compile(policy, mode="max-autotune")
        ref = torch.compile(ref, mode="max-autotune")
    if world_size > 1:
        policy = DDP(policy, device_ids=[local_rank], find_unused_parameters=False)

    raw_policy = policy.module if hasattr(policy, "module") else policy
    optimizer = build_optimizer(raw_policy, args.lr, weight_decay=0.0)

    dataset = PreTokenizedPreferenceDataset(args.data_dir, seq_len=args.seq_len, max_samples=args.max_pairs)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_pairs,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if is_main:
        logger.info(f"Densy DPO | beta={args.beta} | {world_size} GPUs | max_steps={args.max_steps}")

    policy.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    global_step = 0
    accum_loss = 0.0
    all_acc = []

    for batch in loader:
        if global_step >= args.max_steps:
            break

        ci = batch["chosen_input_ids"].to(device, non_blocking=True)
        cl = batch["chosen_labels"].to(device, non_blocking=True)
        ri = batch["rejected_input_ids"].to(device, non_blocking=True)
        rl = batch["rejected_labels"].to(device, non_blocking=True)

        combined_in = torch.cat([ci, ri], dim=0)
        B = ci.shape[0]

        is_last = (micro_step == args.ga_steps - 1)
        ctx = policy.no_sync() if (world_size > 1 and not is_last) else torch.nullcontext() if hasattr(torch, "nullcontext") else __import__("contextlib").nullcontext()

        with ctx:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                p_out = policy(combined_in)
                p_logits = p_out["logits"]
                pi_c = get_log_probs(p_logits[:B], cl)
                pi_r = get_log_probs(p_logits[B:], rl)

                with torch.no_grad():
                    r_out = ref(combined_in)
                    r_logits = r_out["logits"]
                    ref_c = get_log_probs(r_logits[:B], cl)
                    ref_r = get_log_probs(r_logits[B:], rl)

                loss, c_rew, r_rew = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=args.beta)
                loss = loss / args.ga_steps

            loss.backward()

        accum_loss += loss.item()
        acc = (c_rew > r_rew).float().mean().item()
        all_acc.append(acc)
        micro_step += 1

        if micro_step == args.ga_steps:
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_policy.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if is_main and global_step % 5 == 0:
                avg_acc = sum(all_acc[-20:]) / max(1, len(all_acc[-20:]))
                logger.info(f"DPO Step {global_step}/{args.max_steps} | Loss {accum_loss*args.ga_steps:.4f} | Acc {avg_acc:.3f} | Norm {grad_norm:.3f}")

            if global_step % args.save_every_steps == 0 and is_main:
                ckpt_dir = output_dir / f"dpo_step{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                raw = raw_policy
                bf16_sd = {k: v.to(torch.bfloat16) for k, v in raw.state_dict().items()}
                torch.save(bf16_sd, ckpt_dir / "pytorch_model.bin")
                raw.cfg.to_json(str(ckpt_dir / "model_config.json"))
                logger.info(f"Saved DPO checkpoint: {ckpt_dir}")

            accum_loss = 0.0
            micro_step = 0

    if is_main:
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        raw = raw_policy
        bf16_sd = {k: v.to(torch.bfloat16) for k, v in raw.state_dict().items()}
        torch.save(bf16_sd, final_dir / "pytorch_model.bin")
        torch.save(bf16_sd, final_dir / "model_bf16.pt")
        raw.cfg.to_json(str(final_dir / "model_config.json"))
        logger.info(f"DPO complete. Model exported to {final_dir}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
