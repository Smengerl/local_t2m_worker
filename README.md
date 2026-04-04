# Text-to-Image Local Inference

Run Hugging Face text-to-image models (Stable Diffusion) locally with a single Python script.

## Prerequisites

- Python 3.9+
- Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU

## Setup

```bash
# Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Edit the configuration block at the top of `generate.py` to change the prompt or model, then run:

```bash
python generate.py
```

The generated image is saved to `outputs/output.png`.

## Configuration (`generate.py`)

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `stabilityai/stable-diffusion-2-1` | Hugging Face model ID |
| `PROMPT` | `"A futuristic city…"` | Text prompt |
| `NEGATIVE_PROMPT` | `"blurry, …"` | Things to avoid in the image |
| `NUM_INFERENCE_STEPS` | `30` | More steps → better quality, slower |
| `GUIDANCE_SCALE` | `7.5` | How closely to follow the prompt |

## Notes

- Model weights are downloaded automatically on first run and cached in `~/.cache/huggingface`.
- On Apple Silicon the MPS backend is used automatically for faster inference.
