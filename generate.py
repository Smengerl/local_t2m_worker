"""
Text-to-Image generation entry point.

Delegates all model loading and inference to the pipeline class selected by
the "pipeline_type" key in the config file.  The pipeline_type is mandatory —
every config file must declare it explicitly.

Available pipeline types:
  sd    Stable Diffusion 1.5  (diffusers)
  sdxl  Stable Diffusion XL   (diffusers)
  anima Anima / AnimaYume     (Cosmos-Predict2, via sd-cli binary)

Run via run.sh or directly:
    python generate.py --config configs/sd15_default.json "a sunset"
    python generate.py --help
"""

from cli import build_config, parse_args, print_config
from pipelines import create_pipeline


def main() -> None:
    args = parse_args()

    # Merge defaults → config file → CLI flags; resolve output path
    cfg, output_path = build_config(args)
    print_config(cfg, args, output_path)

    pipeline = create_pipeline(cfg)
    image = pipeline.generate(
        prompt=cfg["_effective_prompt"],
        negative_prompt=args.negative_prompt,
    )

    image.save(output_path)
    print(f"✅ Image saved to: {output_path}")


if __name__ == "__main__":
    main()
