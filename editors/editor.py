import tempfile
import os

class EDIT:
    def __init__(self, editing_model_name, ckpt_path):
        if editing_model_name == 'PixEdit':
            from .pix_editor import Editor
        elif editing_model_name in ['AURORA', 'MagicBrush', 'InstructPix2Pix']:
            from .sd_based_editor import Editor
        elif editing_model_name == 'OmniGen':
            from .omni_editor import Editor

        self.editor = Editor(ckpt_path=ckpt_path)
        self.editing_model_name = editing_model_name
        self.temp_files_path =[ ]


    def save_pil_image_temp(self, pil_image):
        '''
        Saves a PIL Image to a temporary file and returns its path.
        '''

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_path = temp_file.name
        pil_image.save(temp_path)
        self.temp_files_path.append(temp_path)
        return temp_path



    def edit(self, input_image, plans, seed=0):
        intermediate_images = []
        
        for plan in plans:
            print(plan)
            input_image = self.editor.edit_image(input_image, plan, seed=seed)

            intermediate_images.append(input_image)

            if self.editing_model_name == 'OmniGen':
                input_image = self.save_pil_image_temp(input_image)

        # Remove any temp file created during execution
        for k in self.temp_files_path:
            try:
                os.unlink(k)
            except:
                pass

        return input_image, intermediate_images

