"""
train_llm.py
============
A module for training large language models (LLMs) using Ray Train.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

:copyright: (c) 2025 by Wonkyo Choe.
:license: MIT, see LICENSE for more details.
"""


from typing import Dict, Any
import tempfile
import uuid
import os

import deepspeed
import torch

import ray
import ray.train
import ray.train.torch
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
from ray.train import Checkpoint
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DownloadConfig

# Configuration
from config import load_config, ExperimentalConfig

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
            checkpoint_dir = os.path.join(tmp_dir, "checkpoint")
            os.makedirs(checkpoint_dir, exist_ok=True)

            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(model_state, os.path.join(checkpoint_dir, "model.pt"))

            epoch_file = os.path.join(checkpoint_dir, "epoch.txt")
            with open(epoch_file, "w") as f:
                f.write(str(epoch_value))

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
                os.path.join(checkpoint_dir, f"model.pt")
            )

            model.module.load_state_dict(model_state_dict)

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
    train_loader = setup_dataloader(config["model_name"], config["dataset_name"], config["seq_length"], config["batch_size"])
    steps_per_epoch = len(train_loader)

    # Setup a model
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    model = ray.train.torch.prepare_model(model)

    # Define a loss function
    # This won't be used because the model internally calcaulates the loss
    # loss_fn = torch.nn.CrossEntropyLoss()

    # Define an optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # Load an existing checkpoint if available
    start_epoch = 0
    ckpt = ray.train.get_checkpoint()
    if ckpt:
        start_epoch, model = load_checkpoint(model, ckpt)

    # Get a device info
    device = ray.train.torch.get_device()
    
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        if ray.train.get_context().get_world_size() > 1 and hasattr(train_loader, "sampler"):
            sampler = getattr(train_loader, "sampler", None)
            if sampler is not None:
                sampler.set_epoch(epoch)
        running_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            print(f"Epoch {epoch}, Step {step + 1}/{steps_per_epoch}, Loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if step + 1 >= config.get("tutorial_steps", steps_per_epoch):
                print(f"Reached tutorial steps limit at step {step + 1}. Ending epoch early.")
                break
        
        report_metrics_and_save_checkpoint(model, {"loss": running_loss / num_batches, "epoch": epoch})


if __name__ == "__main__":
    config = load_config("configs/tune-rust.yaml")

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
