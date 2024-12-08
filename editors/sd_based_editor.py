from __future__ import annotations
import os

import PIL.Image
os.environ['TRANSFORMERS_OFFLINE']='1'
import math
import random
import sys
from pathlib import Path

current_file_path = Path(__file__).resolve()
ROOT = os.path.join(str(current_file_path.parent), 'instruct-pix2pix')
sys.path.insert(0, os.path.join(str(current_file_path.parent), 'instruct-pix2pix'))
print(os.path.join(str(current_file_path.parent), 'instruct-pix2pix'))

from argparse import ArgumentParser
import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
import PIL
from PIL import Image, ImageOps
from torch import autocast
from tqdm.auto import tqdm
import json
import os
import pickle
import torch.nn.functional as F


sys.path.append(os.path.join(str(current_file_path.parent), 'instruct-pix2pix/stable_diffusion'))


from stable_diffusion.ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)



def read_json(path):
    with open(path, 'r') as f:
        dct = json.load(f)
    return dct



def load_image(path):
    input_image = Image.open(path).convert("RGB")
    width, height = input_image.size
    factor = 512 / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
    return input_image



def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model



class Editor:
    def __init__(self, ckpt_path):
        self.setup_model(ckpt_path)


    def setup_model(self, ckpt_path):
        config = os.path.join(ROOT, 'configs/generate.yaml')
        if 'aurora' in ckpt_path:
            self.steps = 50
        else:
            self.steps =100

        self.config = OmegaConf.load(config)
        self.model = load_model_from_config(self.config,ckpt_path,None)
        self.model.eval().cuda()
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.null_token = self.model.get_learned_conditioning([""])



    def edit_image(self, input_image, edit_instruction, seed=0, text_guidance_scale=7.5, img_guidance_scale=1.5):

        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert('RGB').resize((512,512))
        else:
            input_image = input_image.resize((512,512))

        with torch.no_grad(), autocast("cuda"), self.model.ema_scope():
            cond = {}
            cond["c_crossattn"] = [self.model.get_learned_conditioning([edit_instruction])]
            input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image, "h w c -> 1 c h w").to(self.model.device)
            cond["c_concat"] = [self.model.encode_first_stage(input_image).mode()]

            uncond = {}
            uncond["c_crossattn"] = [self.null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = self.model_wrap.get_sigmas(self.steps)

            extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": text_guidance_scale,
            "image_cfg_scale": img_guidance_scale,
            }


            torch.manual_seed(seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, z, sigmas, extra_args=extra_args)
            x = self.model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
            return edited_image







