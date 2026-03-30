from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 200019        # must match encoding_name
    seq_len: int = 1024
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    encoding_name: str = "o200k_base"


@dataclass
class TrainConfig:
    batch_size: int = 32            # per-GPU
    lr: float = 3e-4
    weight_decay: float = 0.1
    epochs: int = 10
    grad_clip: float = 1.0
    log_interval: int = 100         # steps
    save_interval: int = 1000       # steps
    checkpoint_dir: str = "checkpoints"
