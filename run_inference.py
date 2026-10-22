#!/usr/bin/env python3
import sys
import argparse
import torch
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",         default=".")
    p.add_argument("--prompt",             default="<|bos|><|usr|>Hello, who are you?<|asst|>")
    p.add_argument("--max_new_tokens",     type=int,   default=128)
    p.add_argument("--temperature",        type=float, default=0.8)
    p.add_argument("--top_k",             type=int,   default=50)
    p.add_argument("--top_p",             type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_path)
    sys.path.insert(0, str(model_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from tokenizer.crayon_wrapper import ForgeTokenizer
    tok_dir = model_dir / "tokenizer"
    tokenizer = ForgeTokenizer.from_pretrained(str(tok_dir) if tok_dir.exists() else str(model_dir))

    from config import DensyConfig
    from model import build_densy_500m
    cfg = DensyConfig.from_json(str(model_dir / "model_config.json"))
    model = build_densy_500m(cfg)

    for fname in ("pytorch_model.bin", "model_bf16.pt", "model.pt"):
        wp = model_dir / fname
        if wp.exists():
            sd = torch.load(str(wp), map_location="cpu")
            model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
            break

    model = model.to(device).to(torch.bfloat16).eval()
    input_ids = torch.tensor([tokenizer.encode(args.prompt, add_bos=False, add_eos=False)], device=device)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.special_tokens.get("<|eos|>", 2),
    )

    print(tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False))


if __name__ == "__main__":
    main()
