import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import time
import os
import json
from tqdm.auto import tqdm
import sys
import functools
from functools import lru_cache

from diffusers import (
    StableDiffusionXLPipeline,
    DiffusionPipeline,
    PixArtAlphaPipeline,
    AutoPipelineForText2Image,
    StableDiffusion3Pipeline,
)
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler



class GEN:
    def __init__(self, model_name):
        self.model_name = model_name
        self.pipe = self.setup_model(model_name)

    def setup_model(self, model_name):
        if model_name == "sd1.5":
            pipe = DiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif model_name == "sd2.1":
            pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif model_name == "sdxl":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
        elif model_name == "sd3.5 large":
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float16
            )
        elif model_name == "pixart":
            pipe = PixArtAlphaPipeline.from_pretrained(
                "PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16
            )
        elif model_name == "pg2.5":
            pipe = DiffusionPipeline.from_pretrained(
                "playgroundai/playground-v2.5-1024px-aesthetic",
                torch_dtype=torch.float16,
                variant="fp16",
            )
        elif model_name == "deepfloyd":
            stage_1 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-I-XL-v1.0",
                variant="fp16",
                torch_dtype=torch.float16,
                safety_checker=None,
            ).to("cuda")
            stage_2 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-II-L-v1.0",
                text_encoder=None,
                variant="fp16",
                torch_dtype=torch.float16,
            ).to("cuda")
            safety_modules = {
                "feature_extractor": stage_1.feature_extractor,
                "safety_checker": stage_1.safety_checker,
                "watermarker": stage_1.watermarker,
            }
            stage_3 = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler",
                **safety_modules,
                torch_dtype=torch.float16,
            ).to("cuda")

            def deepfloyd_pipe(prompt, generator, num_inference_steps=None):
                prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

                stage_1_output = stage_1(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    generator=generator,
                    output_type="pt",
                ).images

                stage_2_output = stage_2(
                    image=stage_1_output,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    generator=generator,
                    output_type="pt",
                ).images

                stage_3_output = stage_3(
                    prompt=prompt,
                    image=stage_2_output,
                    noise_level=100,
                    generator=generator,
                )
                return stage_3_output
            pipe = deepfloyd_pipe
            
        else:
            raise ValueError(
                "Valid options are : ['sd1.5', 'sd2.1', 'sdxl', 'sd3.5 large', 'pixart', 'pg2.5', 'deepfloyd']"
            )

        if model_name != 'deepfloyd':
            pipe = pipe.to("cuda")

        return pipe

    def generate(self, prompt, save_img=False):
        SEED = 0
        num_steps = 28 if self.model_name == 'sd3.5 large' else 50
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        image = self.pipe(prompt, num_inference_steps=num_steps, generator=generator).images[0]
        if save_img:
            out_dir = './TEST'
            os.makedirs(f"{out_dir}/{self.model_name}", exist_ok=True)
            image.save(f"{out_dir}/{self.model_name}/base_img.png")
        return image


if __name__ == "__main__":
    prompt = sys.argv[1]
    out_dir = sys.argv[2]
    model = sys.argv[3]

    genny = GEN("sdxl")
    genny.generate(prompt, save_img=True)
