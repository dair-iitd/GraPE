from pathlib import Path
import sys, os

# current_file_path = Path(__file__).resolve()
# ROOT = os.path.join(str(current_file_path.parent), 'PixEdit')
# sys.path.insert(0, os.path.join(str(current_file_path.parent), 'PixEdit'))
# print(os.path.join(str(current_file_path.parent), 'PixEdit'))

sys.path.insert(0, '/home/scai/phd/aiz228170/scratch/PixEdit')
ROOT = '/home/scai/phd/aiz228170/scratch/PixEdit'



import torch
import torch.nn as nn
import torchvision.transforms as T
from diffusion.utils.misc import read_config
from diffusion.model.nets import PixArtMS
from transformers import T5EncoderModel, T5Tokenizer

from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import InterpolationMode
from diffusers.models import AutoencoderKL
import matplotlib.pyplot as plt
from diffusion import IDDPM, DPMS
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
import os
from diffusion.data.transforms import get_transform
from torchvision.datasets.folder import default_loader
import json
import sys


class Editor:
    def __init__(self, ckpt_path):
        self.setup_model(ckpt_path)

    def setup_model(self, ckpt_path):
        ROOT_DIR = os.path.join(ROOT, "ckpt")
        CKPT_PATH = ckpt_path
        CONFIG_PATH = os.path.join(ROOT, 'configs/pixart_sigma_config/editing_at_512.py')
        PIPELINE_LOAD_PATH = os.path.join(ROOT, "output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers")
        print(PIPELINE_LOAD_PATH)

        SD = torch.load(CKPT_PATH, 'cpu')
        CONFIG = read_config(CONFIG_PATH)
        kwargs = {'config': CONFIG}
        kwargs['config']['input_size'] = 64

        CONFIG.seed = init_random_seed(CONFIG.get('seed', None))
        set_random_seed(CONFIG.seed)
        self.config = CONFIG


        pixart_model = PixArtMS(depth=28, hidden_size=1152, patch_size=2, num_heads=16, in_channels=8, **kwargs)

        pixart_model.load_state_dict(SD['state_dict'], strict=False)

        device = "cuda"
        self.pixart_model = pixart_model.to(device).eval()

        image_size = CONFIG.image_size
        self.vae = AutoencoderKL.from_pretrained(os.path.join(ROOT, CONFIG.vae_pretrained)).to(device).to(torch.float16)
        self.tokenizer = T5Tokenizer.from_pretrained(PIPELINE_LOAD_PATH, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(
                    PIPELINE_LOAD_PATH, subfolder="text_encoder", torch_dtype=torch.float16).to(device)

        self.transform = get_transform('default_train', image_size)
        null_y = torch.load(os.path.join(ROOT, f'output/pretrained_models/null_embed_diffusers_{CONFIG.model_max_length}token.pth'))
        self.null_y = null_y['uncond_prompt_embeds'].to(device)


    def edit_image(self, input_image, edit_instruction, text_guidance=4.5, img_guidance=1.5):
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert('RGB').resize((512,512))
        else:
            input_image = input_image.resize((512,512))

        set_random_seed(0)
        latent_size = self.config.image_size//8
        device = "cuda"
        z = torch.randn(1, 4, latent_size, latent_size, device=device)
        src_image = self.transform(input_image)

        txt_tokens = self.tokenizer(
            edit_instruction, max_length=self.config.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).to("cuda")
        caption_embs = self.text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0]

        latent_src = self.vae.encode(src_image.to(device, dtype=self.vae.dtype).unsqueeze(0)).latent_dist.mode()

        uncond_latent_src = torch.zeros_like(latent_src)
        ar = 1.
        hw = torch.tensor([[512, 512]], dtype=torch.float, device=device).repeat(1, 1)
        ar = torch.tensor([[ar]], device=device).repeat(1, 1)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=txt_tokens.attention_mask, 
                                    z_src=torch.cat([latent_src, latent_src, uncond_latent_src]))
        


        dpm_solver = DPMS(self.pixart_model.forward_with_dpmsolver,
                      condition=caption_embs,
                      uncondition=self.null_y,
                      cfg_scale=text_guidance,
                      model_kwargs=model_kwargs,
                      img_cfg_scale=img_guidance)
        
        denoised = dpm_solver.sample(
            z,
            steps=14,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )

        latent = denoised.to(torch.float16)
        samples = self.vae.decode(latent.detach() / self.vae.config.scaling_factor).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
        image = Image.fromarray(samples)
        return image


if __name__ == "__main__":
    ImgEditor = Editor("/home/scai/phd/aiz228170/scratch/Generate-then-Edit/PixArt-sigma/output/aurora_ft_ep_40_seed_real_only_40_more_from_ep_40_bs_256_grad_clip_0_5_ip2p_concat/")

    edited_image = ImgEditor("Image", "prompt")

