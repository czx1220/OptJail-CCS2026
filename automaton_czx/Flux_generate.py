import argparse
import torch
from diffusers import FluxPipeline
from pathlib import Path

DEFAULT_MODEL_PATH = "black-forest-labs/FLUX.1-schnell"

pipe = FluxPipeline.from_pretrained(DEFAULT_MODEL_PATH, torch_dtype=torch.bfloat16)

pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
pipe.enable_sequential_cpu_offload()

def generate_image(prompt, save_dir=None, index=0, save=True):
    with torch.no_grad():
        seed = 42
        try:
            image = pipe(
                prompt,
                output_type="pil",
                num_inference_steps=4,
                generator=torch.Generator("cuda").manual_seed(seed)
            ).images[0]
            if save:
                image.save(save_dir / f"safe_figure_{index:03d}.png")
            return image
        except Exception as e:
            print(f"Generation error: prompt {index} - {str(e)}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../results/OptJail_prompt.txt")
    parser.add_argument("--output_dir", type=str, default="../figures_files/OptJail_prompt")
    args = parser.parse_args()

    save_dir = Path(args.output_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    with open(args.input, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    success_count = 0
    for idx, original_prompt in enumerate(prompts):
        print(f"Processing prompt {idx+1}/{len(prompts)}")
        if generate_image(original_prompt, save_dir, idx+1):
            success_count += 1

    print(f"Done. Successfully generated {success_count} images.")