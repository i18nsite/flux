#!/usr/bin/env python

import torch
from transformers import (
  AutoModelForCausalLM,
  AutoProcessor,
  GenerationConfig,
  StopStringCriteria,
)
from PIL import Image
import time

# For 2 x 24 GB. If using 1 x 48 GB or more (lucky you), you can just use device_map="auto"
device_map = {
  "model.vision_backbone": "cpu",  # Seems to be required to not run out of memory at 48 GB
  "model.transformer.wte": 0,
  "model.transformer.ln_f": 0,
  "model.transformer.ff_out": 1,
}
# For 2 x 24 GB, this works for *only* 38 or 39. Any higher or lower and it'll either only work for 1 token of output or fail completely.
switch_point = 38  # layer index to switch to second GPU
device_map |= {f"model.transformer.blocks.{i}": 0 for i in range(0, switch_point)}
device_map |= {f"model.transformer.blocks.{i}": 1 for i in range(switch_point, 80)}

model_name = "SeanScripts/Molmo-72B-0924-nf4"
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  use_safetensors=True,
  device_map=device_map,
  trust_remote_code=True,  # Required for Molmo at the moment.
)
model.model.vision_backbone.float()  # vision backbone needs to be in FP32 for this

processor = AutoProcessor.from_pretrained(
  model_name,
  trust_remote_code=True,  # Required for Molmo at the moment.
)

torch.cuda.empty_cache()

image = Image.open("test.png")
inputs = processor.process(images=image, text="Caption this image.")
inputs = {k: v.to("cuda:0").unsqueeze(0) for k, v in inputs.items()}
prompt_tokens = inputs["input_ids"].size(1)
print("Prompt tokens:", prompt_tokens)

t0 = time.time()
output = model.generate_from_batch(
  inputs,
  generation_config=GenerationConfig(
    max_new_tokens=256,
  ),
  stopping_criteria=[
    StopStringCriteria(tokenizer=processor.tokenizer, stop_strings=["<|endoftext|>"])
  ],
  tokenizer=processor.tokenizer,
)
t1 = time.time()
total_time = t1 - t0
generated_tokens = output.size(1) - prompt_tokens
time_per_token = generated_tokens / total_time
print(
  f"Generated {generated_tokens} tokens in {total_time:.3f} s ({time_per_token:.3f} tok/s)"
)

response = processor.tokenizer.decode(
  output[0, prompt_tokens:], skip_special_tokens=True
)
print(response)

torch.cuda.empty_cache()
