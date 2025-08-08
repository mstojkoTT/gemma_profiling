import time
import sys
import os
sys.path.append(os.getcwd())

import threading
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch
import profiling_custom

import profiling_custom

model_id = "google/gemma-3-27b-it"

def read_first_line(path):
    with open(path, 'r') as f:
        return f.readline().rstrip('\n')

os.environ["HF_TOKEN"] = read_first_line("hf_token.txt")

print(read_first_line("hf_token.txt"))

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id, use_fast=True) # use_fast is here

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://s1.1zoom.me/big3/229/375981-alexfas01.jpg"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]
# https://s1.1zoom.me/big3/229/375981-alexfas01.jpg

t0 = time.perf_counter()

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

# print("Thread ID:", threading.get_ident())
# print("Thread Name:", threading.current_thread().name)

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    generation = generation[0][input_len:]

time_to_first_token = time.perf_counter() - t0



print()
print()
print()

print(f'{time_to_first_token=}')

print()
print()
print()


print('time from t0 until start of image processing:', profiling_custom.pre_image_processing_time - t0)
image_processing_time = profiling_custom.post_image_processing_time - profiling_custom.pre_image_processing_time
print('image processing time:', image_processing_time)


print()
print()
print('time from end of image processing until start of image encoder', profiling_custom.image_encoder_t0 - profiling_custom.post_image_processing_time)
print('image encoder: time in vision tower', profiling_custom.vision_tower_time)
print('image encoder: time in multi modal projector', profiling_custom.multi_modal_projector_time)
total_image_encoder_time = profiling_custom.vision_tower_time + profiling_custom.multi_modal_projector_time
print('total image encoder time', total_image_encoder_time)
print()

print('time between image encoder end and llm start (without image encoder)', profiling_custom.llm_start - profiling_custom.image_encoder_t0 - total_image_encoder_time)
print('time in llm prefill (without image encoder)', profiling_custom.llm_end - profiling_custom.llm_start)

#print('average llm time: ', profiling_custom.llm_total_time / profiling_custom.llm_call_count, ' called', profiling_custom.llm_call_count, 'times')
decoded = processor.decode(generation, skip_special_tokens=True)
print('Model output', decoded)

# **Overall Impression:** The image is a close-up shot of a vibrant garden scene, 
# focusing on a cluster of pink cosmos flowers and a busy bumblebee. 
# It has a slightly soft, natural feel, likely captured in daylight.
