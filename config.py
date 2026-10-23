from dataclasses import dataclass, field, asdict
import json
from pathlib import Path


@dataclass
class DensyConfig:
    vocab_size: int = 206464
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 24
    mlp_ratio: int = 4
    seq_len: int = 4096
    dropout: float = 0.0
    rope_theta: float = 10000.0
    weight_tie: bool = True
    pad_token_id: int = 0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def mlp_dim(self) -> int:
        return self.d_model * self.mlp_ratio

    def to_json(self, path: str):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str) -> "DensyConfig":
        return cls(**json.loads(Path(path).read_text()))
