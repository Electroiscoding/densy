#!/usr/bin/env python3
import os
import sys
import time
import math
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from config import DensyConfig
from model import build_densy_330m
from trainer import build_optimizer, CosineScheduler
from hub_uploader import upload_folder_async

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_dpo")


class DPODataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int, max_pairs: Optional[int] = None, seed: int = 42):
        self.seq_len = seq_len
        npz_files = sorted(Path(data_dir).glob("**/*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")
        self.samples = []
        for f in npz_files:
            try:
                data = np.load(str(f), allow_pickle=True)
                cf = data["chosen_full_ids"] if "chosen_full_ids" in data else data.get("chosen_input_ids")
                rf = data["rejected_full_ids"] if "rejected_full_ids" in data else data.get("rejected_input_ids")
                cm = data["chosen_loss_mask"]
                rm = data["rejected_loss_mask"]
                for i in range(len(cf)):
                    self.samples.append((cf[i], rf[i], cm[i], rm[i]))
            except Exception as e:
                logger.warning(f"Error loading {f}: {e}")
        if max_pairs and max_pairs < len(self.samples):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(self.samples), max_pairs, replace=False)
            self.samples = [self.samples[i] for i in idx]
        logger.info(f"DPODataset: loaded {len(self.samples):,} preference pairs")

    def __len__(self) -> int:
        return len(self.samples)

    def _pad_trunc(self, arr: np.ndarray, pad_val: int) -> torch.Tensor:
        t = torch.from_numpy(arr[:self.seq_len]).long()
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
    p.add_argument("--batch_pairs",      type=int,   default=2)
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

    policy = build_densy_330m(cfg)
    ref = build_densy_330m(cfg)

    for fn in ("pytorch_model.bin", "model_bf16.pt", "model.pt"):
        wp = base_dir / fn
        if wp.exists():
            sd = torch.load(str(wp), map_location="cpu")
            if "model" in sd:
                sd = sd["model"]
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            policy.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
            ref.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
            if is_main:
                logger.info(f"Loaded base model weights from {wp}")
            break

    policy = policy.to(device)
    ref = ref.to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    if args.bf16:
        policy = policy.to(torch.bfloat16)
        ref = ref.to(torch.bfloat16)
    if not args.no_compile:
        policy = torch.compile(policy)
        ref = torch.compile(ref)
    if world_size > 1:
        policy = DDP(policy, device_ids=[local_rank], find_unused_parameters=False)

    raw_policy = policy.module if hasattr(policy, "module") else policy
    raw_ref = ref.module if hasattr(ref, "module") else ref

    dataset = DPODataset(args.data_dir, seq_len=args.seq_len, max_pairs=args.max_pairs)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_pairs, sampler=sampler, shuffle=(sampler is None),
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    optimizer = build_optimizer(raw_policy, lr_max=args.lr, weight_decay=0.01)
    scheduler = CosineScheduler(optimizer, lr_max=args.lr, lr_min=args.lr * 0.1, warmup_tokens=int(args.max_steps * args.batch_pairs * world_size * 0.05), total_tokens=args.max_steps * args.batch_pairs * world_size)

    if is_main:
        logger.info(f"Densy DPO | {world_size} GPUs | beta={args.beta} | max_steps={args.max_steps}")

    policy.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    accum_loss = 0.0
    accum_c_rew = 0.0
    accum_r_rew = 0.0
    start_t = time.perf_counter()

    for epoch in range(100):
        if global_step >= args.max_steps:
            break
        if sampler:
            sampler.set_epoch(epoch)
        for batch in loader:
            if global_step >= args.max_steps:
                break
            ci = batch["chosen_input_ids"].to(device, non_blocking=True)
            cl = batch["chosen_labels"].to(device, non_blocking=True)
            ri = batch["rejected_input_ids"].to(device, non_blocking=True)
            rl = batch["rejected_labels"].to(device, non_blocking=True)

            is_last = (micro_step == args.ga_steps - 1)
            ctx = policy.no_sync() if (world_size > 1 and not is_last) else torch.nullcontext() if hasattr(torch, "nullcontext") else __import__("contextlib").nullcontext()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                with torch.inference_mode():
                    ref_c_out = raw_ref(ci)
                    ref_r_out = raw_ref(ri)
                    ref_c = get_log_probs(ref_c_out["logits"], cl)
                    ref_r = get_log_probs(ref_r_out["logits"], rl)

                with ctx:
                    pi_c_out = policy(ci)
                    pi_r_out = policy(ri)
                    pi_c = get_log_probs(pi_c_out["logits"], cl)
                    pi_r = get_log_probs(pi_r_out["logits"], rl)
                    loss, c_rew, r_rew = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=args.beta)
                    scaled_loss = loss / args.ga_steps

                scaled_loss.backward()

            accum_loss += loss.item() / args.ga_steps
            accum_c_rew += c_rew.mean().item() / args.ga_steps
            accum_r_rew += r_rew.mean().item() / args.ga_steps
            micro_step += 1

            if micro_step == args.ga_steps:
                grad_norm = torch.nn.utils.clip_grad_norm_(raw_policy.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step(args.batch_pairs * world_size)
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if is_main and (global_step <= 5 or global_step % 5 == 0):
                    elapsed = time.perf_counter() - start_t
                    logger.info(
                        f"DPO Step {global_step}/{args.max_steps} | "
                        f"Loss {accum_loss:.4f} | "
                        f"Margin {(accum_c_rew - accum_r_rew):+.3f} | "
                        f"Norm {grad_norm:.3f} | {elapsed:.1f}s"
                    )
                    start_t = time.perf_counter()

                if global_step % args.save_every_steps == 0 and is_main:
                    ckpt_dir = output_dir / f"step_{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    raw = raw_policy
                    bf16_sd = {k: v.to(torch.bfloat16) for k, v in raw.state_dict().items()}
                    torch.save(bf16_sd, ckpt_dir / "pytorch_model.bin")
                    raw.cfg.to_json(str(ckpt_dir / "model_config.json"))
                    logger.info(f"Saved DPO checkpoint: {ckpt_dir}")
                    upload_folder_async(str(ckpt_dir), repo_id="Phase-Technologies/Densy-330M-Chat", path_in_repo=ckpt_dir.name)

                accum_loss = 0.0
                accum_c_rew = 0.0
                accum_r_rew = 0.0
                micro_step = 0

    if is_main:
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        raw = raw_policy
        bf16_sd = {k: v.to(torch.bfloat16) for k, v in raw.state_dict().items()}
        torch.save(bf16_sd, final_dir / "pytorch_model.bin")
        torch.save(bf16_sd, final_dir / "model_bf16.pt")
        raw.cfg.to_json(str(final_dir / "model_config.json"))
        logger.info(f"DPO complete. Final aligned model exported to {final_dir}")
        upload_folder_async(str(final_dir), repo_id="Phase-Technologies/Densy-330M-Chat", path_in_repo="final")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
