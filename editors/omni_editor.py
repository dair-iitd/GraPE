from pathlib import Path
import os
import sys
current_file_path = Path(__file__).resolve()

ROOT = os.path.join(str(current_file_path.parent), 'OmniGen')
sys.path.insert(0, os.path.join(str(current_file_path.parent), 'OmniGen'))
print(os.path.join(str(current_file_path.parent), 'OmniGen'))

from OmniGen import OmniGenPipeline



class Editor:
    def __init__(self, ckpt_path):
        self.setup_model(ckpt_path)
    
    def setup_model(self, ckpt_path):
        self.pipe = OmniGenPipeline.from_pretrained(ckpt_path) 

    def edit_image(self, input_image, edit_instruction, seed=0, text_guidance_scale=2.5, img_guidance_scale=1.6):
        images = self.pipe(
            prompt=f"<img><|image_1|></img> {edit_instruction}",
            input_images=[input_image],
            use_input_image_size_as_output=True,
            guidance_scale=text_guidance_scale, 
            img_guidance_scale=img_guidance_scale,
            seed=seed
        )[0]
    
        return images



