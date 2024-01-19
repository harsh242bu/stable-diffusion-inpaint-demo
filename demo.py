import argparse
import os
import time

from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from xformers.ops import (
    MemoryEfficientAttentionCutlassOp,
    MemoryEfficientAttentionFlashAttentionOp,
)


def inpaint_image(args: argparse.ArgumentParser):
    image = Image.open(args.img_file)
    mask_image = Image.open(args.mask_img_file)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        # torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    pipe.enable_xformers_memory_efficient_attention(MemoryEfficientAttentionCutlassOp)

    # pipe.enable_xformers_memory_efficient_attention(
    #     attention_op=MemoryEfficientAttentionFlashAttentionOp
    # )
    # pipe.enable_vae_slicing()
    # pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)

    prompt = args.prompt
    start_time = time.time()
    # The mask structure is white for inpainting and black for keeping as is

    prompt_input = [prompt] * args.num_images
    img_list = pipe(
        prompt=prompt_input,
        image=image,
        mask_image=mask_image,
        num_inference_steps=args.num_inference_steps,
    ).images
    print("Images generated: ", len(img_list))

    prompt_suffix = prompt.replace(" ", "_").lower()
    for idx, img in enumerate(img_list):
        img.save(args.img_file.replace(".png", f"_inpaint_{prompt_suffix}_{idx}.png"))

    print(f"Time taken: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable diffusion inpainting demo")
    parser.add_argument(
        "--img_file",
        type=str,
        default="images/dog/dog.png",
        # required=True,
        help="Path to image file",
    )
    parser.add_argument(
        "--mask_img_file",
        type=str,
        default=None,
        help="Path to mask image file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Face of a yellow cat, high resolution, sitting on a park bench",
        # prompt = "Face of a white male with eyes closed"
        help="Prompt for the model",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=60,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of images to generate",
    )

    args = parser.parse_args()
    print(f"args: {args}")

    if args.mask_img_file is None:
        args.mask_img_file = args.img_file.replace(".png", "_mask.png")
        print("mask_file: ", args.mask_img_file)

    if not os.path.exists(args.img_file):
        raise ValueError(f"Image file {args.img_file} does not exist")
    if not os.path.exists(args.mask_img_file):
        raise ValueError(f"Mask image file {args.mask_img_file} does not exist")

    inpaint_image(args)
