import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast

from config import ModelConfig, TrainConfig
from model import LLM
from dataset import get_dataset, build_dataloader


def setup():
    dist.init_process_group(backend="nccl")


def cleanup():
    dist.destroy_process_group()


def train():
    setup()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    model = LLM(model_cfg).to(device, dtype=torch.bfloat16)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    dataset         = get_dataset(model_cfg, split="train")
    loader, sampler = build_dataloader(dataset, train_cfg.batch_size, rank, world_size)

    if rank == 0:
        os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    step = 0
    for epoch in range(train_cfg.epochs):
        sampler.set_epoch(epoch)
        model.train()

        for tokens in loader:
            tokens = tokens.to(device, non_blocking=True)   # (B, seq_len + 1)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(tokens[:, :-1], tokens[:, 1:]) # scalar

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if rank == 0 and step % train_cfg.log_interval == 0:
                print(f"epoch {epoch}  step {step}  loss {loss.item():.4f}", flush=True)

            if rank == 0 and step > 0 and step % train_cfg.save_interval == 0:
                ckpt = os.path.join(train_cfg.checkpoint_dir, f"ckpt_{step:07d}.pt")
                torch.save(
                    {
                        "step":      step,
                        "model":     model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_cfg": model_cfg,
                        "train_cfg": train_cfg,
                    },
                    ckpt,
                )

            step += 1

    cleanup()


if __name__ == "__main__":
    train()
