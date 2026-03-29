# ORLM-LLaMA-3-8B Setup Guide — Apple Silicon (MLX)

> A step-by-step tutorial for running the ORLM-LLaMA-3-8B model locally on an
> Apple Silicon Mac using MLX.

---

## 1. Machine Requirements

| Requirement | Minimum | Recommended | This Machine |
|---|---|---|---|
| Chip | Apple M1 | M2 Pro / M4 Pro | ✅ Apple M4 Pro |
| Unified Memory | 16 GB | 24+ GB | ✅ 24 GB |
| Free Disk Space | 10 GB | 20 GB | ✅ 252 GB free |
| macOS | 13.5 (Ventura) | 14+ (Sonoma) | ✅ |
| Python | 3.10+ | 3.13 | ✅ 3.13.12 |
| Metal GPU | Metal 3 | Metal 4 | ✅ Metal 4, 16 cores |

**Why not vLLM?** The ORLM repo's default inference uses
[vLLM](https://github.com/vllm-project/vllm) which requires NVIDIA CUDA GPUs.
On Apple Silicon, we use [MLX](https://github.com/ml-explore/mlx) — Apple's
native ML framework that leverages unified memory and Metal GPU acceleration.

---

## 2. Create a Python Virtual Environment

```bash
# Use Python 3.13 (from Homebrew)
python3.13 -m venv ~/ll4or-mlx-env
source ~/ll4or-mlx-env/bin/activate

# Verify
python --version   # Python 3.13.x
which python       # ~/ll4or-mlx-env/bin/python
```

---

## 3. Install Dependencies

```bash
source ~/ll4or-mlx-env/bin/activate
pip install mlx-lm huggingface_hub
```

This installs:
- `mlx` + `mlx-metal` — Apple's ML framework with Metal GPU backend
- `mlx-lm` — LLM inference/conversion tools for MLX
- `transformers` — HuggingFace tokenizer support
- `huggingface_hub` — Model downloading from HuggingFace

Verify the installation:
```bash
pip list | grep -iE "mlx|huggingface|transformers"
```

Expected output (versions may vary):
```
huggingface_hub   1.8.0
mlx               0.31.1
mlx-lm            0.31.1
mlx-metal         0.31.1
transformers      5.4.0
```

---

## 4. Download & Convert the Model

This downloads the ORLM-LLaMA-3-8B model from HuggingFace, converts it to MLX
format, and quantizes to 4-bit for efficient inference.

```bash
source ~/ll4or-mlx-env/bin/activate

python -m mlx_lm.convert \
  --hf-path CardinalOperations/ORLM-LLaMA-3-8B \
  -q \
  --q-bits 4 \
  --mlx-path ~/ll4or-mlx-env/models/ORLM-LLaMA-3-8B-4bit
```

> **Note:** This downloads ~16 GB (full model) then quantizes to ~4.2 GB.
> The download may take a while depending on your internet connection.
> You may need a HuggingFace account and token if the model requires accepting
> the LLaMA 3 license agreement.

Verify the model:
```bash
ls -lh ~/ll4or-mlx-env/models/ORLM-LLaMA-3-8B-4bit/
du -sh ~/ll4or-mlx-env/models/ORLM-LLaMA-3-8B-4bit/
```

Expected: ~4.2 GB total, containing `model.safetensors`, `config.json`,
`tokenizer.json`, etc.

### Alternative: 8-bit quantization (higher quality, more memory)

```bash
python -m mlx_lm.convert \
  --hf-path CardinalOperations/ORLM-LLaMA-3-8B \
  -q \
  --q-bits 8 \
  --mlx-path ~/ll4or-mlx-env/models/ORLM-LLaMA-3-8B-8bit
```

This uses ~8.5 GB RAM but produces higher quality output.

---

## 5. Run Inference

### Quick test with the built-in sample question

```bash
cd datasets/ORLM

source ~/ll4or-mlx-env/bin/activate

python scripts/inference_mlx.py \
  --model_path ~/ll4or-mlx-env/models/ORLM-LLaMA-3-8B-4bit
```

### Run with a custom question

```bash
python scripts/inference_mlx.py \
  --model_path ~/ll4or-mlx-env/models/ORLM-LLaMA-3-8B-4bit \
  --question "A factory produces two products A and B. Product A requires 2 hours of labor and 3 kg of material. Product B requires 4 hours of labor and 2 kg of material. The factory has 100 hours of labor and 120 kg of material available. Product A sells for \$40 and product B sells for \$50. How many of each product should be produced to maximize revenue?"
```

### Run with a question from a file

```bash
python scripts/inference_mlx.py \
  --model_path ~/ll4or-mlx-env/models/ORLM-LLaMA-3-8B-4bit \
  --question_file my_question.txt \
  --output_file my_answer.txt
```

### All command-line options

| Flag | Default | Description |
|---|---|---|
| `--model_path` | `~/ll4or-mlx-env/models/ORLM-LLaMA-3-8B-4bit` | Path to MLX model directory |
| `--question` | *(built-in sample)* | Inline OR question text |
| `--question_file` | `None` | Path to text file with OR question |
| `--decoding_method` | `greedy` | `greedy` or `sampling` |
| `--temperature` | `0.7` | Sampling temperature (only for `sampling`) |
| `--top_p` | `0.95` | Nucleus sampling threshold |
| `--max_tokens` | `4096` | Maximum tokens to generate |
| `--output_file` | `None` | Save response to a file |

---

## 6. Expected Performance

Benchmarked on MacBook Pro M4 Pro (24 GB):

| Metric | Value |
|---|---|
| Model load time | ~1.5 seconds |
| Prompt processing | ~99 tokens/sec |
| Generation speed | ~54 tokens/sec |
| Peak memory usage | ~4.8 GB |
| Typical response time | 10–20 seconds (for ~500–1000 token responses) |

---

## 7. Troubleshooting

### "No module named 'mlx'" or similar import errors
Make sure you activated the virtual environment:
```bash
source ~/ll4or-mlx-env/bin/activate
```

### HuggingFace download authentication errors
Login to HuggingFace and accept the LLaMA 3 license:
```bash
pip install huggingface_hub
huggingface-cli login
```
Then visit https://huggingface.co/CardinalOperations/ORLM-LLaMA-3-8B and accept
the license agreement.

### Out of memory errors
- Use 4-bit quantization (default) — uses only ~4.8 GB
- Close other memory-heavy applications
- If still failing, try reducing `--max_tokens`

### Slow generation
- Ensure no other heavy processes are competing for memory bandwidth
- The first generation after loading is slightly slower (Metal shader compilation)
- Check Activity Monitor → Memory Pressure (should be green/yellow)

### Model produces `coptpy` code but you want `gurobipy`
The ORLM model was trained to generate `coptpy` code. You can:
1. Modify the prompt template to request `gurobipy` instead
2. Post-process the output to replace solver calls

---

## 8. File Reference

```
ll4or/
├── SETUP_GUIDE.md                          ← This file
├── datasets/
│   └── ORLM/
│       ├── scripts/
│       │   ├── inference.py                ← Original (vLLM, CUDA)
│       │   └── inference_mlx.py            ← Apple Silicon version (MLX)
│       └── ...
~/ll4or-mlx-env/
├── bin/activate                            ← Virtual environment
├── models/
│   └── ORLM-LLaMA-3-8B-4bit/              ← Quantized model weights
│       ├── model.safetensors
│       ├── config.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
```

---

## 9. Quick Start (TL;DR)

```bash
# Activate environment
source ~/ll4or-mlx-env/bin/activate

# Run inference
cd datasets/ORLM
python scripts/inference_mlx.py \
  --model_path ~/ll4or-mlx-env/models/ORLM-LLaMA-3-8B-4bit \
  --question "Your OR question here"
```
