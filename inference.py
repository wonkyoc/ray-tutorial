import os
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Inference Script")
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to the model checkpoint.")
    #checkpoint_path = "checkpoints/ray-tune-tutorial-experiment/checkpoint_2025-12-30_01-25-28.816231/model.pt"
    args = parser.parse_args()
    checkpoint_path = args.ckpt_path
    
    # Setup model and tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load the checkpoint
    model_state_dict = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(model_state_dict)
    model.eval()  # Set to evaluation mode

    pipeline_model = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    input_text = "Can you give me some basic examples of Rust?"

    outputs = pipeline_model(input_text, max_length=250, num_return_sequences=1)

    print(outputs)