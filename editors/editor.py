class EDIT:
    def __init__(self, editing_model_name, ckpt_path):
        if editing_model_name == 'PixEdit':
            from .pix_editor import Editor
        elif editing_model_name in ['AURORA', 'MagicBrush', 'InstructPix2Pix']:
            from .ip2p_based_editor import Editor
        elif editing_model_name == 'OmniGen':
            from .omni_editor import Editor

        self.editor = Editor(ckpt_path=ckpt_path)

    def edit(self, input_image, plans):
        intermediate_images = []
        for plan in plans:
            input_image = self.editor.edit_image(input_image, plan)
            intermediate_images.append(input_image)
        return input_image, intermediate_images

