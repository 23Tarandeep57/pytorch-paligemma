from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os
import torch

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    import gc
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    print("Creating model structure on meta device (no memory allocated yet)...")
    # Create the model on meta device first (doesn't allocate real memory)
    with torch.device('meta'):
        model = PaliGemmaForConditionalGeneration(config)
    
    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    
    print(f"Loading weights from {len(safetensors_files)} files...")
    # Load tensors one file at a time to minimize memory usage
    tensors = {}
    for i, safetensors_file in enumerate(safetensors_files):
        print(f"Loading shard {i+1}/{len(safetensors_files)}: {os.path.basename(safetensors_file)}")
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                # Convert to float16 to save memory (reduces model size by ~50%)
                if tensor.dtype == torch.float32:
                    tensors[key] = tensor.half()
                else:
                    tensors[key] = tensor
        # Force garbage collection after each file
        gc.collect()
    
    print("Materializing model with loaded weights...")
    # Now materialize the model from meta device with the loaded weights
    model = model.to_empty(device=device)
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(tensors, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys[:5]}...")  # Show first 5
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
    
    # Clear tensors dict to free memory
    del tensors
    gc.collect()
    
    # Re-initialize buffers that may not have been properly initialized from meta device
    # to_empty() creates uninitialized tensors, so we need to reinitialize position_ids
    print("Re-initializing position buffers...")
    for name, module in model.named_modules():
        if hasattr(module, 'position_ids') and hasattr(module, 'num_positions'):
            # Re-register the position_ids buffer with correct values
            module.register_buffer(
                "position_ids",
                torch.arange(module.num_positions, device=device).expand((1, -1)),
                persistent=False,
            )
            print(f"  Reinitialized position_ids for {name}")

    # Tie weights
    model.tie_weights()

    print("Model loaded successfully!")
    return (model, tokenizer)