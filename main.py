import time
import sys
import gc
import os
sys.path.append(os.getcwd())
from collections import defaultdict
import pandas as pd

import threading
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch
import profiling_custom

import profiling_custom

import sys, traceback

def get_model_vram_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size  # in bytes
    return total_size / (1024 ** 2)  # MB

def do_profile(model_variant, device, use_fast, dtype, nr_iterations):
    avg_times = defaultdict(lambda: 0)

    for _ in range(nr_iterations):
        individual_times = do_profile_individual(model_variant=model_variant, device=device, use_fast=use_fast, dtype=dtype)
        for k, v in individual_times.items():
            avg_times[k] += v
    
    for k, v in avg_times.items():
        avg_times[k] = avg_times[k] / nr_iterations
    
    return avg_times
        

def do_profile_individual(model_variant, device, use_fast, dtype):

    model_id = "google/gemma-3-" + model_variant + "-it"

    free, total = torch.cuda.mem_get_info()
    print(f"Before model load - Free: {free / 1024**2:.2f} MB")
    print(f"Before model load - Used: {(total - free) / 1024**2:.2f} MB")
    print(f"Before model load - Total: {total / 1024**2:.2f} MB")

    def read_first_line(path):
        with open(path, 'r') as f:
            return f.readline().rstrip('\n')

    os.environ["HF_TOKEN"] = read_first_line("hf_token.txt")

    print(read_first_line("hf_token.txt"))

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map=device, torch_dtype=dtype,
    ).eval()

    free, total = torch.cuda.mem_get_info()
    print()
    print()
    print(f"Before preprocessing - Free: {free / 1024**2:.2f} MB")
    print(f"Before preprocessing - Used: {(total - free) / 1024**2:.2f} MB")
    print(f"Before preprocessing - Total: {total / 1024**2:.2f} MB")

    processor = AutoProcessor.from_pretrained(model_id, use_fast=use_fast) # use_fast is here

    free, total = torch.cuda.mem_get_info()
    print()
    print()
    print(f"After preprocessing - Free: {free / 1024**2:.2f} MB")
    print(f"After preprocessing - Used: {(total - free) / 1024**2:.2f} MB")
    print(f"After preprocessing - Total: {total / 1024**2:.2f} MB")

    for k, v in model.named_parameters():
        assert device in str(v.device)

    print(f"Model size on VRAM: {get_model_vram_size(model):.2f} MB")

    from itertools import chain

    type_dict = defaultdict(lambda: 0)
    for k, v in chain(model.named_parameters(), model.named_buffers()):
        # type_dict[str(v.dtype)] += v.numel() * v.element_size()
        type_dict[str(v.dtype)] += 1
        if v.dtype != dtype:
            print('non', dtype, k)

    # print(type_dict)

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

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(device, dtype=dtype)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        generation = generation[0][input_len:]

    time_to_first_token_with_img_load = time.perf_counter() - t0



    print()
    print()
    print()

    # print(f'{time_to_first_token=}')
    image_loading_time = profiling_custom.image_loading_end - profiling_custom.image_loading_starting
    time_to_first_token = time_to_first_token_with_img_load - image_loading_time
    print('time to first token without image loading time (from disk): ', time_to_first_token)

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
    llm_time = profiling_custom.llm_end - profiling_custom.llm_start
    print('time in llm prefill (without image encoder)', llm_time)

    #print('average llm time: ', profiling_custom.llm_total_time / profiling_custom.llm_call_count, ' called', profiling_custom.llm_call_count, 'times')
    decoded = processor.decode(generation, skip_special_tokens=True)
    print('Model output', decoded)

    ratio = image_processing_time / time_to_first_token * 100

    ret_dict = {
        'ttft': time_to_first_token,
        'preprocessing' : image_processing_time,
        'img encoder' : total_image_encoder_time,
        'text model prefill' : llm_time,
        'preproc./ttft [%]': ratio,
        'img. enc. tower time': profiling_custom.vision_tower_time,
        'img. enc. multi-modal projector time': profiling_custom.multi_modal_projector_time
    }

    ret_dict = {k + ' [s]' if k != 'preproc./ttft [%]' else k: v for k, v in ret_dict.items()}

    return ret_dict

def refresh_gpu_cuda():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

# ret_dict = do_profile(model_variant='27b', device='cuda', use_fast=True, dtype=torch.bfloat16)

NR_ITERATIONS = 3

tables = dict()
nan_ret_dict = dict()
for model_variant in ['4b']:
# for model_variant in ['4b', '27b', '12b']:
# for model_variant in ['4b', '27b']:

    for dtype in [torch.bfloat16, torch.float32]:
        rows = []

        for use_fast in [False, True]:
            for device in ['cuda', 'cpu']:
                if dtype == torch.float32 and model_variant == '27b':
                    ret_dict = defaultdict(lambda: float('nan'))
                else:

                    # to avoid running out of vram
                    refresh_gpu_cuda()

                    ret_dict = do_profile(model_variant=model_variant, device=device, use_fast=use_fast, dtype=dtype, nr_iterations=NR_ITERATIONS)

                    refresh_gpu_cuda()

                    if len(nan_ret_dict) == 0:
                        nan_ret_dict = {k: float('nan') for k, _ in ret_dict.items()}

                rows.append({
                    # "Config": f"{device} use_fast={use_fast} {str(dtype)[6:]}",
                    "Config": f"{device} use_fast={use_fast}",
                    **ret_dict
                })

        df = pd.DataFrame(rows)

        # Convert to clean markdown
        markdown_table = df.to_markdown(index=False)
        tables[model_variant + ' ' + str(dtype)[6:]] = markdown_table

output_file = "table.md"
with open(output_file, "w") as f:
    f.write('Runs to average each datapoint: {}\n'.format(NR_ITERATIONS))
    for k, v in tables.items():
        f.write("## " + k + "\n")
        f.write(v + "\n\n")

assert NR_ITERATIONS >= 5