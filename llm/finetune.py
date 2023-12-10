import os
from pathlib import Path
import sys
import time
from typing import Optional

import lightning as L
import torch
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies import FSDPStrategy

from lit_gpt.lora import (
    GPT,
    Block,
    Config,
    lora_filter,
    mark_only_lora_as_trainable,
    merge_lora_weights
)
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    load_checkpoint,
    num_parameters,
)

class HParams(dict):
    __getattr__ = dict.get

    def save(self, path):
        import json

        to_save = {k: str(v) for k, v in self.items()}
        with open(path / "hparams.json", "w") as f:
            json.dump(to_save, f)


def finetune(
    model_name: str = "microsoft/phi-1_5",
    data_dir: str = "data/alpaca",
    out_dir: str = "out/finetune",
    out_checkpoint: str = "lit_model.pth",
    max_iter: Optional[int] = None,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    n_epochs: int = 3,
    learning_rate: float = 3e-4,
    max_seq_length: Optional[int] = 1024,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    eval_interval: int = 100,
    save_interval: int = 100,
    eval_iters: int = 100,
    log_interval: int = 1,
):
    checkpoint_dir = Path("checkpoints") / model_name
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    hparams = HParams(**locals())

    torch.set_float32_matmul_precision("high")

    precision = "bf16-true" if torch.cuda.is_available() else "16-mixed"

    strategy = "auto"

    if torch.cuda.device_count() > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
        )

    logger = CSVLogger(out_dir.parent, flush_logs_every_n_steps=log_interval)

    fabric = L.Fabric(strategy=strategy, precision=precision, loggers=logger)
    fabric.launch(train, hparams)

    return hparams


def train(fabric, hparams):
    check_valid_checkpoint_dir(hparams.checkpoint_dir)

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        hparams.out_dir.mkdir(exist_ok=True, parents=True)

    train_data = torch.load(hparams.data_dir / "train.pt")
    val_data = torch.load(hparams.data_dir / "test.pt")

    checkpoint_path = hparams.checkpoint_dir / "lit_model.pth"

    config = Config.from_name(
        name=hparams.checkpoint_dir.name,
        r=8,
        alpha=16,
        dropout=0.05,
        to_query=True,
        to_key=False,
        to_value=True,
        to_projection=False,
        to_mlp=False,
        to_head=False,
    )

    fabric.print(f"Loading model {str(checkpoint_path)!r}")

    with fabric.init_module(empty_init=True):
        model = GPT(config)

    mark_only_lora_as_trainable(model)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")

    model = fabric.setup_module(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
    optimizer = fabric.setup_optimizers(optimizer)

    # strict=False because missing keys due to LoRA weights not contained in state dict
    load_checkpoint(fabric, model, checkpoint_path, strict=False)

    fabric.seed_everything(1337 + fabric.global_rank)

    tokenizer = Tokenizer(hparams.checkpoint_dir)

    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in train_data]
    longest_seq_length = max(lengths)
    model.max_seq_length = min(longest_seq_length, hparams.max_seq_length or float("inf"))

    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    max_iters = min(len(train_data) * hparams.n_epochs, hparams.max_iter or float("inf"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters // hparams.batch_size)

    gradient_accumulation_iters = hparams.batch_size // hparams.micro_batch_size
    assert gradient_accumulation_iters > 0

    step_count = 0
    total_lengths = 0
    total_t0 = time.perf_counter()
    train_time = time.perf_counter()

    for iter_num in range(1, max_iters + 1):
        if step_count <= hparams.warmup_steps:
            # linear warmup
            lr = hparams.learning_rate * step_count / hparams.warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids, targets = get_batch(fabric, train_data, hparams)

        is_accumulating = iter_num % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, lm_head_chunk_size=128)
            # shift the targets such that output n predicts token n+1
            logits[-1] = logits[-1][..., :-1, :]
            loss = chunked_cross_entropy(logits, targets[..., 1:])
            fabric.backward(loss / gradient_accumulation_iters)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            if step_count > hparams.warmup_steps:
                scheduler.step()
            step_count += 1

        total_lengths += input_ids.numel()
        if iter_num % hparams.log_interval == 0:
            loss_item = loss.item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            fabric.print(
                f"iter {iter_num} step {step_count}: loss {loss_item:.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )

        if not is_accumulating and step_count % hparams.eval_interval == 0:
            t0 = time.perf_counter()
            with torch.no_grad():
                model.eval()
                losses = torch.zeros(max_iters)
                for k in range(max_iters):
                    input_ids, targets = get_batch(fabric, val_data, hparams)
                    logits = model(input_ids)
                    losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
                val_loss = losses.mean()
                model.train()
            t1 = time.perf_counter() - t0
            fabric.print(f"step {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.barrier()

        if not is_accumulating and step_count % hparams.save_interval == 0:
            checkpoint_path = hparams.out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            save_lora_checkpoint(fabric, model, checkpoint_path)

    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")

    # Save the final checkpoint at the end of training
    merge_lora_weights(model)

    save_path = hparams.out_checkpoint
    fabric.print(f"Saving weights to {str(save_path)!r}")

    # remove lora parameters and the lora linear substring
    state_dict = {k.replace("linear.", ""): v for k, v in model.state_dict().items() if not lora_filter(k, v)}
    torch.save(state_dict, save_path)


def get_batch(fabric, data, hparams):
    ix = torch.randint(len(data), (hparams.micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    # Truncate if needed
    if hparams.max_seq_length:
        x = x[:, :hparams.max_seq_length]
        y = y[:, :hparams.max_seq_length]

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(finetune)
