"""
train_llm.py
============
A module for training large language models (LLMs) using Ray Train.

:copyright: (c) 2026 by Wonkyo Choe.
:license: MIT, see LICENSE for more details.
"""


from datetime import datetime
import json
import time
from typing import Dict, Any
import tempfile
import uuid
import os
from pathlib import Path


import ray
import ray.train
import ray.train.torch
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
from ray.train import Checkpoint
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DownloadConfig

import torch
import numpy as np

# Configuration
from config import load_config, ExperimentalConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AuditLogger:
    def __init__(self):
        self.timings = {}

    def log(self, phase, duration):
        if phase not in self.timings:
            self.timings[phase] = []
        self.timings[phase].append(duration)

    def summary(self):
        summary = {}
        for phase, duration in self.timings.items():
            summary[phase] = {
                'mean': np.mean(duration),
                'std': np.std(duration),
                'min': np.min(duration),
                'max': np.max(duration),
                'total': np.sum(duration),
                'count': len(duration),
            }
        return summary
    
timer = AuditLogger()

def setup_dataloader(model_name: str, dataset_name: str, seq_length: int, batch_size: int) -> DataLoader:
    # Load dataset
    if ":" in dataset_name:
        repo, cfg = dataset_name.split(":", 1)
        dataset = load_dataset(repo, data_dir=cfg, split="train[:1%]", download_config=DownloadConfig(disable_tqdm=True))
    else:
        dataset = load_dataset(dataset_name, split="train[:1%]", download_config=DownloadConfig(disable_tqdm=True))

    # Preprocessing
    def _tokenize_function(examples):
        return tokenizer(examples["content"], padding='max_length', max_length=seq_length, truncation=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token

    #dataset = dataset.filter(lambda x: "fn " in x["content"]).take(2000)
    tokenized_datasets = dataset.map(_tokenize_function, batched=True, num_proc=1, keep_in_memory=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Create DataLoader
    # batch_size is a batch size per worker
    data_loader = DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=True)

    # Use prepare_data_loader for distributed training
    return ray.train.torch.prepare_data_loader(data_loader)

def report_metrics_and_save_checkpoint(
    model: torch.nn.Module,
    metrics: Dict[str, Any]
) -> None:
    epoch_value = metrics['epoch']
    ctx = ray.train.get_context()
    rank = ctx.get_world_rank()

    # Only rank 0 saves the checkpoint
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint = None 

        if rank == 0:
            start = time.time()
            checkpoint_dir = os.path.join(tmp_dir, "checkpoint")
            os.makedirs(checkpoint_dir, exist_ok=True)

            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(model_state, os.path.join(checkpoint_dir, "model.pt"))

            epoch_file = os.path.join(checkpoint_dir, "epoch.txt")
            with open(epoch_file, "w") as f:
                f.write(str(epoch_value))
            
            timer.log("ckpt_saving", time.time() - start)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            experiment_name = ctx.get_experiment_name()
            print(f"Experiment {experiment_name}, Epoch {epoch_value}, Metrics: {metrics}")

        # ALL ranks must call report (collective operation)
        ray.train.report(metrics, checkpoint=checkpoint)

def load_checkpoint(
    model: torch.nn.Module,
    ckpt: Checkpoint
) -> tuple[int, torch.nn.Module]: 
    # Reference: https://docs.ray.io/en/latest/train/user-guides/checkpoints.html#restore-training-state-from-a-checkpoint
    next_epoch = 0
    try:
        with ckpt.as_directory() as checkpoint_dir:
            # Load the last checkpoint
            print(f"Loading checkpoint from {checkpoint_dir}")
            # Restore a checkpoint
            model_state_dict = torch.load(
                os.path.join(checkpoint_dir, "model.pt")
            )

            if hasattr(model, 'module'):
                model.module.load_state_dict(model_state_dict)
            else:
                model.load_state_dict(model_state_dict)

            epoch_file = os.path.join(checkpoint_dir, "epoch.txt")
            # Read last epoch
            if os.path.isfile(epoch_file):
                with open(epoch_file, "r", encoding="utf-8") as f:
                    last_epoch = int(f.read().strip())
                next_epoch = last_epoch + 1

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e
    return next_epoch, model


def train_loop(config: Dict[str, Any]) -> None:
    # Setup dataloader
    start = time.time()
    train_loader = setup_dataloader(config["model_name"], config["dataset_name"], config["seq_length"], config["batch_size"])
    steps_per_epoch = len(train_loader)
    timer.log("data_preparation", time.time() - start)

    # Setup a model
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    model = ray.train.torch.prepare_model(model)

    timer.log("model_loading", time.time() - start)

    # Define a loss function
    # This won't be used because the model internally calcaulates the loss
    # loss_fn = torch.nn.CrossEntropyLoss()

    # Define an optimizer
    start = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    timer.log("optimizer_setup", time.time() - start)

    # Load an existing checkpoint if available
    start = time.time()
    start_epoch = 0
    ckpt = ray.train.get_checkpoint()
    if ckpt:
        start_epoch, model = load_checkpoint(model, ckpt)
    timer.log("ckpt_loading", time.time() - start)

    # Get a device info
    device = ray.train.torch.get_device()

    for epoch in range(start_epoch, config["epochs"]):
        epoch_start = time.time()

        model.train()
        if ray.train.get_context().get_world_size() > 1 and hasattr(train_loader, "sampler"):
            sampler = getattr(train_loader, "sampler", None)
            if sampler is not None:
                sampler.set_epoch(epoch)
        running_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            data_transfer_start = time.time()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            timer.log("data_transfer", time.time() - data_transfer_start)

            forward_start = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            timer.log("forward_pass", time.time() - forward_start)
            print(f"Epoch {epoch}, Step {step + 1}/{steps_per_epoch}, Loss: {loss.item()}")

            optimizer.zero_grad()
            backward_start = time.time()
            loss.backward()
            timer.log("backward_pass", time.time() - backward_start)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if step + 1 >= config.get("tutorial_steps", steps_per_epoch):
                print(f"Reached tutorial steps limit at step {step + 1}. Ending epoch early.")
                break

        timer.log("epoch_training", time.time() - epoch_start)
        report_metrics_and_save_checkpoint(model, {"loss": running_loss / num_batches, "epoch": epoch})

    if ray.train.get_context().get_world_rank() == 0:
        audit_summary = timer.summary()
        print("\n=== Training phase timing breakdown ===")
        for phase, stats in audit_summary.items():
            print(f"{phase}:")
            print(f"mean={stats['mean']:.4f}s")
            print(f"std={stats['std']:.4f}s")
            print(f"min={stats['min']:.4f}s")
            print(f"max={stats['max']:.4f}s")
            print(f"total={stats['total']:.4f}s over {stats['count']} calls")
            print("-----------------------------------")
        
        with open("/mnt/shared/ray/training_timing_audit.txt", "w") as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'audit_summary': audit_summary
            }, f, indent=2)



if __name__ == "__main__":
    # Connect to Ray Cluster
    ray.init(address="auto")

    config_path = PROJECT_ROOT / "configs" / "tune-rust.yaml"
    config = load_config(str(config_path))

    scaling_config = ScalingConfig(num_workers=config.ray.num_workers, use_gpu=config.ray.use_gpu > 0)

    train_loop_config = {
        "epochs": config.training.num_epochs,
        "learning_rate": config.training.learning_rate,
        "batch_size": config.training.batch_size,
        "model_name": config.training.model_name,
        "dataset_name": config.training.dataset_name,
        "seq_length": config.training.seq_length,
        "tutorial_steps": config.training.tutorial_steps,
    }

    run_config = RunConfig(
        storage_path=config.training.storage_path,
        #name=f"{config.project_name}-{str(uuid.uuid4())[:8]}",
        name=f"{config.project_name}-experiment",
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    result = trainer.fit()


    print(f"Training finished. Result: {result}")
