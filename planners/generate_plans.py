from openai import OpenAI
import base64
import os
# os.environ['TRANSFORMERS_OFFLINE']='1'
# os.environ['HF_HOME'] = '/scratch/scai/phd/aiz228170/qwen-vl-72b'

import requests
import json
from functools import lru_cache
import sys
from .FEW_SHOT import OPENAI_KEY, SYS_PROMPT, EXAMPLES

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



class PLAN:
    def __init__(self, model_name, sys_prompt, examples):
        self.sys_prompt = sys_prompt
        self.examples = examples
        if model_name not in ['gpt4o', 'Qwen/Qwen2-VL-72B-Instruct', 'Qwen/Qwen2-VL-7B-Instruct']:
            raise ValueError('model name not found')

        self.model_name = model_name
        if 'gpt' in model_name:
            self.model = self.get_gpt_op
            self.processor = None
        else:
            self.model, self.processor = self.setup_qwen(self.model_name)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.IMAGE_DIR = os.path.join(BASE_DIR, 'planners', 'few_shot_images')


    def gpt_template(self, img_path, prompt):
        SYS = {
                "role" : "system",
                "content": [
                    {
                    "type" : "text",
                    "text" : self.sys_prompt
                    }
                ]
              }

        example_list = []
        for (ex_ipath, ex_prompt, assistant_resp) in self.examples:
            D1 = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Image:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{encode_image(os.path.join(self.IMAGE_DIR, ex_ipath))}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text", 
                            "text": f'Textual Prompt: "{ex_prompt}"'
                        }
                    ],
                }
            
            example_list.append(D1)

            D2 = {
                    "role": "assistant",
                    "content": [
                        {
                        "type": "text",
                        "text": assistant_resp
                        }
                    ]
                }

            example_list.append(D2)



        example_list.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Image:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{encode_image(img_path)}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text", 
                            "text": f'Textual Prompt: "{prompt}"'
                        }
                    ],
                },
        )
        return [SYS] + example_list 




    def qwen_template(self, img_path, prompt):
        SYS = {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": self.sys_prompt
                    }
                ]
              }

        example_list = []
        for (ex_ipath, ex_prompt, assistant_resp) in self.examples:
            D1 = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Image:"},
                        {
                            "type": "image",
                            "image": os.path.join(self.IMAGE_DIR, ex_ipath),
                            "resized_height": 512,
                            "resized_width": 512,
                        },
                        {
                            "type": "text", 
                            "text": f'Textual Prompt: "{ex_prompt}"'
                        }
                    ],
                }
            example_list.append(D1)

            D2 = {
                    "role": "assistant",
                    "content": [
                        {
                        "type": "text",
                        "text": assistant_resp
                        }
                    ]
                }
            example_list.append(D2)

            
            example_list.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Image:"},
                        {
                            "type": "image",
                            "image": img_path,
                        },
                        {
                            "type": "text", 
                            "text": f'Textual Prompt: "{prompt}"'
                        }
                    ],
                },
            )

            return [SYS] + example_list 


    def setup_qwen(self, model_id):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", cache_dir="/scratch/scai/phd/aiz228170/qwen-vl-72b"
        )

        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor



    def get_gpt_op(self, img_path, prompt):
        client = OpenAI(api_key=OPENAI_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=self.gpt_template(img_path, prompt),
            temperature=0,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content


    def get_qwen_op(self, model, processor, img_path, prompt):
        messages=self.qwen_template(img_path, prompt)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs,max_new_tokens=1024)

        generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text



    def generate(self, img_path, prompt):
        if 'gpt' in self.model_name:
            response_str = self.model(img_path, prompt)
        elif 'qwen' in self.model_name.lower():
            response_str = self.get_qwen_op(self.model, self.processor, img_path, prompt)
        else:
            raise ValueError("Supported MLLMs for Planner are 'gpt-4o', 'Qwen/Qwen2-VL-72B-Instruct' or 'Qwen/Qwen2-VL-7B-Instruct' ")

        print(response_str)
        if 'no further feedback is necessary' in response_str.lower():
            plan_lst = []   
        elif "<REGENERATE>" in response_str:
            plan_lst = ["<REGENERATE>"]
        else:
            try:
                plan_lst = [k.replace('- ','').strip().replace('.','') for k in response_str.split('**Feedback**:')[1].split('\n') if k != '']
            except:
                plan_lst = []
                print(response_str)
        return plan_lst






if __name__ == "__main__":
    img_path = sys.argv[1]
    prompt = sys.argv[2]
    
    planner = PLAN('Qwen/Qwen2-VL-72B-Instruct', SYS_PROMPT, EXAMPLES)
    plans = planner.generate(img_path, prompt)
    print(plans)
