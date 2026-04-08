# Contributing

Thanks for your interest in contributing to Local Text-to-Image Inference! Contributions are welcome and appreciated. To make collaboration smooth, please follow these guidelines.

## How to Contribute

1. **Fork the repository** and create a feature branch
2. **Make your changes** in a clearly named branch (e.g., `fix/mps-memory-leak` or `feat/add-flux-lora`)
3. **Write clear commit messages** following [Conventional Commits](https://www.conventionalcommits.org/)
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `refactor:` for code refactoring
4. **Update `README.md`** if you change CLI flags, config keys, supported backends, or add a config file — see the [README.md maintenance rules](.github/copilot-instructions.md) for exactly which sections to keep in sync
5. **Open a Pull Request** with a clear description of what you changed and why

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/local_t2m_worker.git
cd local_t2m_worker

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (Optional) Store your HuggingFace token for gated models
echo "hf_..." > .hf_token

# Test a generation
./run.sh "a simple test prompt"
```

## Adding a New Model Config

The easiest contribution is adding a new config file for a Hugging Face model:

1. Create `configs/my_model.json` with the correct `pipeline_type` and parameters
2. Test it: `./run.sh -c configs/my_model.json "a test prompt"`
3. Add it to the **"Configuration files" table** in `README.md`

> 💡 **GitHub Copilot users:** Use the built-in skills to speed this up:
> - `#config-builder` — researches the model card and generates a complete config file automatically
> - `#config-tester` — smoke-tests an existing config and auto-fixes common issues

See `configs/CONFIGS.md` and existing config files for reference.

## Adding a New Pipeline Backend

1. Create `pipelines/my_pipeline.py` inheriting `BasePipeline`
2. Implement `generate(prompt, negative_prompt) -> PIL.Image`
3. Register it in `_REGISTRY` in `pipelines/__init__.py`
4. Add an entry to the **"Supported backends" table** in `README.md`
5. Create at least one example config in `configs/`

## Code Style

- Follow **PEP 8** for Python code
- Add **type hints** for function parameters and return values
- Keep functions small; add comments for non-obvious logic
- If adding dependencies, update `requirements.txt`

## Reporting Issues

- **Search existing issues** before opening a new one
- **Provide clear reproduction steps** with expected vs actual behavior
- **Include your environment**: OS, Python version, hardware (CPU/MPS/CUDA), and the config file used
- **Attach the error output** or generated image if relevant

## Questions?

Check the [README.md](README.md) first — usage, CLI flags, config keys, and the batch system are all documented there. For anything else, open an issue.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
