"""
import sys, os, io, traceback

_THIS_FILE = os.path.abspath(__file__)

class AnnotatedStream(io.TextIOBase):
    def __init__(self, orig):
        self._orig = orig
        self._buf = ""

    def writable(self): return True

    def write(self, s):
        # Buffer until newline to avoid breaking partial writes
        self._buf += s
        out = []
        while True:
            if "\n" not in self._buf:
                break
            line, self._buf = self._buf.split("\n", 1)
            prefix = self._loc()
            out.append(f"[{prefix}] {line}\n")
        if out:
            # IMPORTANT: write to the original stream to avoid recursion
            self._orig.write("".join(out))
        return len(s)

    def flush(self):
        if self._buf:
            self._orig.write(f"[{self._loc()}] {self._buf}")
            self._buf = ""
        self._orig.flush()

    def _loc(self):
        # Find the first frame that's not in this file
        # (print() is C, so the next Python frame is usually the caller)
        stack = traceback.extract_stack(limit=50)
        for frame in reversed(stack[:-1]):  # skip current frame
            fname = os.path.abspath(frame.filename)
            if fname != _THIS_FILE:
                # make path short-ish
                try:
                    rel = os.path.relpath(fname)
                except Exception:
                    rel = fname
                return f"{rel}:{frame.lineno}"
        return "?:?"

# Wrap both stdout and stderr
sys.stdout = AnnotatedStream(sys.__stdout__)
sys.stderr = AnnotatedStream(sys.__stderr__)
"""



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

import sys, traceback

model_id = "google/gemma-3-12b-it"

def read_first_line(path):
    with open(path, 'r') as f:
        return f.readline().rstrip('\n')

os.environ["HF_TOKEN"] = read_first_line("hf_token.txt")

print(read_first_line("hf_token.txt"))

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="cpu"
).eval()

processor = AutoProcessor.from_pretrained(model_id, use_fast=False) # use_fast is here

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/home/ttuser/mstojko/gemma_profiling/picture.jpg"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

t0 = time.perf_counter()

# for k, v in model.named_parameters():
#     assert 'cuda' in str(v.device)

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    generation = generation[0][input_len:]

time_to_first_token = time.perf_counter() - t0



print()
print()
print()

print(f'{time_to_first_token=}')
image_loading_time = profiling_custom.image_loading_end - profiling_custom.image_loading_starting
print('time to first token without image loading time (from disk): ', time_to_first_token - image_loading_time)

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
