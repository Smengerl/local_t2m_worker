# Text-to-Image Local Inference Project

Python project for running Hugging Face text-to-image models (Stable Diffusion) locally using the `diffusers` library.

## Stack
- Python 3.x
- `diffusers` (Hugging Face)
- `transformers`
- `torch` (PyTorch)
- `Pillow`
- `accelerate`

## Project Structure
- `generate.py` – Main script to generate images from text prompts
- `requirements.txt` – Python dependencies
- `outputs/` – Generated images are saved here

## Notes
- Use MPS (Apple Silicon) or CUDA if available, otherwise CPU
- Model weights are cached in `~/.cache/huggingface`
