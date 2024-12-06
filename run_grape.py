from generators import GEN
from FEW_SHOT import OPENAI_KEY, SYS_PROMPT, EXAMPLES

from plan.generate_plan import run_inference as PLAN
from edit.edit_img import run_inference as EDIT




def GraPE(prompt):
    GEN = ModelLoader()
    base_img = GEN.run_inference(prompt, out_dir, model_name, save_img=False)

    planner = Planner('gpt-4o', SYS_PROMPT, EXAMPLES)
    plans = planner.run_inference(img_path, prompt)


    
    return



'''
Ideal use-case

from generators import GEN
from planners import PLAN
from editors import EDIT

genny = GEN("sdxl")
planner = PLAN("gpt-4o")
editor = EDIT("PixEdit")

GRAPE = create_grape_pipeline(genny, planner, editor)

GRAPE(text)

'''