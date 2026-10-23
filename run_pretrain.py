#!/usr/bin/env python3
import os
import sys
import logging
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_pretrain")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",              required=True)
    p.add_argument("--output_dir",            default="./checkpoints/densy_330m")
    p.add_argument("--total_tokens",          type=int,   default=15_000_000_000)
    p.add_argument("--seq_len",               type=int,   default=4096)
    p.add_argument("--global_batch_tokens",   type=int,   default=4_194_304)
    p.add_argument("--micro_batch_per_gpu",   type=int,   default=8)
    p.add_argument("--lr_max",                type=float, default=3e-4)
    p.add_argument("--lr_min",                type=float, default=3e-5)
    p.add_argument("--warmup_ratio",          type=float, default=0.02)
    p.add_argument("--grad_clip",             type=float, default=1.0)
    p.add_argument("--weight_decay",          type=float, default=0.1)
    p.add_argument("--d_model",               type=int,   default=768)
    p.add_argument("--n_heads",               type=int,   default=12)
    p.add_argument("--n_layers",              type=int,   default=24)
    p.add_argument("--mlp_ratio",             type=int,   default=4)
    p.add_argument("--bf16",                  action="store_true", default=True)
    p.add_argument("--no_compile",            action="store_true")
    p.add_argument("--save_every_steps",      type=int,   default=500)
    p.add_argument("--resume_from",           default=None)
    p.add_argument("--hourly_cost",           type=float, default=12.0)
    p.add_argument("--num_workers",           type=int,   default=8)
    return p.parse_args()


def main():
    args = parse_args()

    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    from config import DensyConfig
    from model import build_densy_500m
    from trainer import Trainer
    from data.dataset import PackedTokenDataset, build_dataloader

    cfg = DensyConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        seq_len=args.seq_len,
    )

    model = build_densy_500m(cfg)

    dataset = PackedTokenDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        split="train",
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_per_gpu,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )

    trainer = Trainer(
        model=model,
        cfg=cfg,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        output_dir=args.output_dir,
        total_tokens=args.total_tokens,
        global_batch_tokens=args.global_batch_tokens,
        micro_batch_per_gpu=args.micro_batch_per_gpu,
        seq_len=args.seq_len,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        use_compile=not args.no_compile,
        save_every_steps=args.save_every_steps,
        resume_from=args.resume_from,
        hourly_cost=args.hourly_cost,
    )

    trainer.train(loader)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
