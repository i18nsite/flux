#!/usr/bin/env python

from tqdm import tqdm
import os
from hit_srf_arch import HiT_SRF
import cv2
from os.path import dirname, join, abspath

root = dirname(dirname(abspath(__file__)))
# use GPU (True) or CPU (False)
cuda_flag = False

# initialize model (change model and upscale according to your setting)
model = HiT_SRF(upscale=4)

# load model (change repo_name according to your setting)
repo_name = "XiangZ/hit-srf-4x"
model = model.from_pretrained(repo_name)
# if cuda_flag:
#     model.cuda()

input_dir = join(root, "out")
output_dir = join(root, "scale")
os.makedirs(output_dir, exist_ok=True)

li = []
for root, dirs, files in os.walk(input_dir):
  for file in files:
    if file.endswith(".png"):
      file_path = os.path.join(root, file)
      li.append((file, file_path))

for file, file_path in tqdm(li):
  print(file)
  sr_results = model.infer_image(file_path, cuda=cuda_flag)
  cv2.imwrite(join(output_dir, file), sr_results)
  os.remove(file_path)
