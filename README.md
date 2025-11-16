# pytorch-paligemma (Memory-Optimized Fork)

This is a fork of the original PaliGemma PyTorch implementation with **significant memory optimizations** to enable running the model on systems with limited RAM.

## 🚀 Key Improvements

### Memory Optimization
The original implementation required excessive RAM (~22GB+) to load the 3B parameter model, causing Out-of-Memory (OOM) errors on most consumer hardware. This fork implements several optimizations:

1. **Meta Device Loading**: Model structure is created on `torch.device('meta')` without allocating memory initially
2. **Float16 Precision**: Automatic conversion of weights to float16, reducing memory usage by ~50%
3. **Efficient Materialization**: Uses `to_empty()` to materialize the model structure before loading weights
4. **Proper Buffer Initialization**: Fixes uninitialized position_ids buffers that occur when using meta device loading
5. **Incremental Garbage Collection**: Forces garbage collection after loading each model shard

### Result
- **Before**: ~22GB RAM required (model failed to load on 13GB RAM systems)
- **After**: ~11GB RAM required (successfully runs on 13GB RAM + swap)
- Memory usage reduced by approximately **50%**

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Minimum system requirements:
- 13GB RAM + 8GB swap (or 16GB+ RAM)
- CPU or CUDA-capable GPU

## 🔧 Setup

1. Clone this repository:
```bash
git clone https://github.com/23Tarandeep57/pytorch-paligemma.git
cd pytorch-paligemma
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the PaliGemma model weights using Git LFS:
```bash
# Install Git LFS if not already installed
sudo apt-get install git-lfs  # On Ubuntu/Debian
# or
brew install git-lfs  # On macOS

# Clone the model repository
git lfs install
git clone https://huggingface.co/google/paligemma-3b-pt-224
```

## 🚀 Usage

Run inference using the provided script:

```bash
bash launch_inference.sh
```

Or run directly with Python:

```bash
python inference.py \
    --model_path /path/to/paligemma-3b-pt-224 \
    --prompt "describe this image" \
    --image_file_path test_images/your_image.jpg \
    --max_tokens_to_generate 100 \
    --temperature 0.8 \
    --top_p 0.9 \
    --do_sample False \
    --only_cpu True
```

## 🛠️ Technical Details

### Memory Optimization Implementation

The key changes are in `utils.py`:

```python
# Create model on meta device (no memory allocation)
with torch.device('meta'):
    model = PaliGemmaForConditionalGeneration(config)

# Load weights in float16 to save memory
for safetensors_file in safetensors_files:
    with safe_open(safetensors_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if tensor.dtype == torch.float32:
                tensors[key] = tensor.half()  # Convert to float16
            else:
                tensors[key] = tensor

# Materialize model efficiently
model = model.to_empty(device=device)
model.load_state_dict(tensors, strict=False)

# Fix uninitialized buffers from meta device
for name, module in model.named_modules():
    if hasattr(module, 'position_ids') and hasattr(module, 'num_positions'):
        module.register_buffer(
            "position_ids",
            torch.arange(module.num_positions, device=device).expand((1, -1)),
            persistent=False,
        )
```

## 🐛 Known Issues

- Model runs on CPU by default for maximum compatibility
- Inference is slower on CPU compared to GPU
- Some minor precision differences may occur due to float16 conversion


## �🙏 Acknowledgments

- Original PaliGemma implementation
- Google Research for the PaliGemma model
- HuggingFace for the model hosting and transformers library

## 🔗 Links

- Original Repository: https://github.com/hkproj/pytorch-paligemma
- PaliGemma Model: https://huggingface.co/google/paligemma-3b-pt-224
- Paper: https://arxiv.org/abs/2407.07726