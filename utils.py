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
    
    print("Materializing model in float16...")
    # Materialize the model directly in float16 to save memory
    # We need to manually convert each parameter to float16 after to_empty
    model = model.to_empty(device=device)
    
    # Convert all parameters to float16 to match our weight loading
    # This must be done before loading weights to avoid dtype mismatches
    for name, param in model.named_parameters():
        param.data = param.data.half()
    
    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    
    print(f"Loading weights from {len(safetensors_files)} files directly into model...")
    # Get the state dict to map parameter names
    state_dict = model.state_dict()
    
    # Load tensors one file at a time and copy directly into model
    # This avoids keeping two copies in memory
    missing_keys = set(state_dict.keys())
    
    for i, safetensors_file in enumerate(safetensors_files):
        print(f"Loading shard {i+1}/{len(safetensors_files)}: {os.path.basename(safetensors_file)}")
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in state_dict:
                    tensor = f.get_tensor(key)
                    # Convert to float16 if needed
                    if tensor.dtype == torch.float32:
                        tensor = tensor.half()
                    
                    # Copy directly into the model's parameter
                    state_dict[key].copy_(tensor)
                    missing_keys.discard(key)
                    
                    # Free the tensor immediately
                    del tensor
        
        # Force garbage collection after each file
        gc.collect()
    
    if missing_keys:
        print(f"Warning: Missing keys: {list(missing_keys)[:5]}...")  # Show first 5
    
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