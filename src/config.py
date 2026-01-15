import yaml
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator


class RayConfig(BaseModel):
    num_workers: int = Field(default=2, description="Number of Ray workers.")
    cpus_per_worker: int = Field(default=4, description="CPUs per Ray worker.")
    use_gpu: int = Field(default=2, description="Number of GPUs to use (0 for CPU only).")
    gpus_per_worker: float = Field(default=1.0, description="GPUs per Ray worker.")

class TrainingConfig(BaseModel):
    model_name: str = Field(default="gpt2", description="Name of the pre-trained model.")
    dataset_name: str = Field(default="wikitext", description="Name of the dataset to use.")
    batch_size: int = Field(default=16, description="Batch size for training.")
    seq_length: int = Field(default=512, description="Sequence length for tokenization.")
    learning_rate: float = Field(default=5e-5, description="Learning rate for the optimizer.")
    num_epochs: int = Field(default=3, description="Number of training epochs.")
    storage_path: Optional[str] = Field(default=None, description="Storage path for Ray workers.")
    tutorial_steps: int = Field(default=1000, description="Number of tutorial steps to run.")

class DeepSpeedConfig(BaseModel):
    batch_size: int = Field(default=16, description="Micro batch size per GPU.")
    gradient_accumulation_steps: int = Field(default=1, description="Number of gradient accumulation steps.")
    zero_stage: int = Field(default=2, description="ZeRO optimization stage.")

class ExperimentalConfig(BaseModel):
    project_name: str
    ray: RayConfig
    training: TrainingConfig
    deepspeed: DeepSpeedConfig

def load_config(path: str) -> ExperimentalConfig:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return ExperimentalConfig(**data)