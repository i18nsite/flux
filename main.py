#!/usr/bin/env python

from random import choice
import os
import torch
from diffusers.pipelines import FluxPipeline
from os.path import dirname, join, abspath
import yaml
from PIL import PngImagePlugin
from time import time
from baseconv import base36

分形 = [
  "mandelbrot",  # 曼德布洛特集合
  "julia",  # 朱利亚集合
  "sierpinski",  # 谢尔宾斯基三角形
  "koch",  # 科赫曲线
  "dragon",  # 龙形曲线
  "barnsley",  # 巴恩斯利蕨
  "mandelbox",  # 曼德布洛特盒子
  "apollonian",  # 阿波罗尼奥斯圆盘
  "newton",  # 牛顿分形
  "lyapunov",  # 李雅普诺夫分形
  "burning ship",  # 燃烧船分形
  "tricorn",  # 三曲翼分形
  "mandelbulb",  # 曼德布洛特球
  "ifs",  # 迭代函数系统
  "fractal tree",  # 分形树
  "hilbert",  # 希尔伯特曲线
  "peano",  # 皮亚诺曲线
  "menger",  # 门格海绵
  "cantor",  # 康托尔集
  "l system",  # 兰顿蚂蚁
  "penrose",  # 彭罗斯铺砖
  "gosper",  # 高斯帕曲线
  "quaternion",  # 四元数分形
  "hypercomplex",  # 超复数分形
  "attractor",  # 吸引子
  "chaos game",  # 混沌游戏
  "voronoi",  # 沃罗诺伊图
  "dla",  # 扩散限制凝聚
  "perlin noise",  # 柏林噪声
  "worley noise",  # 沃利噪声
  "fractal flame",  # 分形火焰
  "harmonic",  # 调和分形
  "spectral",  # 光谱分形
  "wavelet",  # 小波分形
  "tessellation",  # 镶嵌分形
  "cellular automaton",  # 元胞自动机
  "game of life",  # 生命游戏
  "fractal landscape",  # 分形景观
  "fractal art",  # 分形艺术
  "fractal music",  # 分形音乐
  "fractal geometry",  # 分形几何
  "fractal dimension",  # 分形维数
  "fractal pattern",  # 分形图案
  "fractal recursion",  # 分形递归
  "fractal algorithm",  # 分形算法
  "fractal animation",  # 分形动画
  "fractal zoom",  # 分形缩放
  "fractal color",  # 分形着色
  "fractal texture",  # 分形纹理
  "fractal flow",  # 分形流动
  "fractal wave",  # 分形波
  "fractal cloud",  # 分形云
  "fractal mountain",  # 分形山脉
  "fractal river",  # 分形河流
  "fractal forest",  # 分形森林
  "fractal sky",  # 分形天空
  "fractal sea",  # 分形海洋
  "fractal island",  # 分形岛屿
  "fractal desert",  # 分形沙漠
  "fractal snowflake",  # 分形雪花
  "fractal crystal",  # 分形水晶
  "fractal flower",  # 分形花朵
  "fractal leaf",  # 分形叶子
  "fractal shell",  # 分形贝壳
  "fractal butterfly",  # 分形蝴蝶
  "fractal dragonfly",  # 分形蜻蜓
  "fractal spiral",  # 分形螺旋
  "fractal star",  # 分形星形
  "fractal hexagon",  # 分形六边形
  "fractal pentagon",  # 分形五边形
  "fractal triangle",  # 分形三角形
  "fractal square",  # 分形正方形
  "fractal circle",  # 分形圆形
  "fractal ellipse",  # 分形椭圆
  "fractal polygon",  # 分形多边形
  "fractal curve",  # 分形曲线
  "fractal surface",  # 分形曲面
  "fractal volume",  # 分形体积
  "fractal boundary",  # 分形边界
  "fractal interior",  # 分形内部
  "fractal exterior",  # 分形外部
  "fractal boundary condition",  # 分形边界条件
  "fractal initial condition",  # 分形初始条件
  "fractal iteration",  # 分形迭代
  "fractal convergence",  # 分形收敛
  "fractal divergence",  # 分形发散
  "fractal escape time",  # 分形逃逸时间
  "fractal orbit",  # 分形轨道
  "fractal attractor",  # 分形吸引子
  "fractal repeller",  # 分形排斥子
  "fractal invariant",  # 分形不变量
  "fractal symmetry",  # 分形对称性
  "fractal self similarity",  # 分形自相似性
  "fractal scale invariance",  # 分形尺度不变性
  "fractal hausdorff dimension",  # 分形豪斯多夫维数
  "fractal box counting dimension",  # 分形盒计数维数
  "fractal minkowski dimension",  # 分形闵可夫斯基维数
  "fractal correlation dimension",  # 分形关联维数
  "fractal information dimension",  # 分形信息维数
  "fractal lyapunov exponent",  # 分形李雅普诺夫指数
  "fractal entropy",  # 分形熵
  "fractal complexity",  # 分形复杂度
  "fractal randomness",  # 分形随机性
  "fractal determinism",  # 分形确定性
  "fractal chaos",  # 分形混沌
  "fractal order",  # 分形有序性
  "fractal beauty",  # 分形美感
  "fractal aesthetics",  # 分形美学
  "fractal inspiration",  # 分形灵感
  "fractal creativity",  # 分形创造力
  "fractal imagination",  # 分形想象力
  "fractal mystery",  # 分形神秘感
  "fractal wonder",  # 分形惊奇感
  "fractal magic",  # 分形魔力
  "fractal enigma",  # 分形谜题
  "fractal puzzle",  # 分形拼图
  "fractal challenge",  # 分形挑战
  "fractal exploration",  # 分形探索
  "fractal discovery",  # 分形发现
  "fractal innovation",  # 分形创新
  "fractal revolution",  # 分形革命
  "fractal evolution",  # 分形进化
  "fractal dynamics",  # 分形动力学
  "fractal mechanics",  # 分形力学
  "fractal physics",  # 分形物理学
  "fractal mathematics",  # 分形数学
  "fractal geometry",  # 分形几何学
  "fractal topology",  # 分形拓扑学
  "fractal analysis",  # 分形分析
  "fractal theory",  # 分形理论
  "fractal model",  # 分形模型
  "fractal simulation",  # 分形模拟
  "fractal experiment",  # 分形实验
  "fractal research",  # 分形研究
  "fractal science",  # 分形科学
  "fractal technology",  # 分形技术
]

tones = [
  "agate",  # 瑪瑙色
  "anthracite",  # 无烟煤色
  "ash",  # 灰烬色
  "basalt",  # 火山岩色
  "black",  # 黑色
  "carbon",  # 碳黑色
  "char",  # 炭色
  "charcoal",  # 木炭色
  "cinder",  # 灰烬色
  "coal",  # 煤色
  "cosmic",  # 宇宙黑
  "dark",  # 深色
  "dusk",  # 暮色
  "ebony",  # 乌木色
  "gloss",  # 光泽黑
  "granite",  # 花岗岩色
  "graphite",  # 石墨色
  "gray",  # 灰色
  "hematite",  # 赤铁矿色
  "ink",  # 墨水色
  "iron",  # 铁灰色
  "jasper",  # 猪油石色
  "jet",  # 喷气黑
  "lava",  # 熔岩色
  "lead",  # 铅色
  "licorice",  # 甘草色
  "limestone",  # 石灰岩色
  "malachite",  # 孔雀石色
  "marble",  # 大理石色
  "matt",  # 哑光黑
  "midnight",  # 午夜蓝
  "navy blue",  # 海军蓝
  "night",  # 夜色
  "obsidian",  # 黑曜石色
  "oil",  # 油黑色
  "onyx",  # 缟玛瑙色
  "opal",  # 欧泊色
  "pearl",  # 珍珠色
  "pitch",  # 柏油色
  "raven",  # 渡鸦黑
  "sable",  # 黑貂色
  "satin",  # 缎面黑
  "shadow",  # 阴影色
  "slate",  # 石板灰
  "smoke",  # 烟灰色
  "soot",  # 煤灰色
  "space",  # 太空黑
  "steel",  # 钢铁灰
  "storm",  # 暴风色
  "suede",  # 绒面革色
  "sulfur",  # 硫磺色
  "tar",  # 焦油色
  "thunder",  # 雷霆色
  "umber",  # 赤土色
  "velvet",  # 天鹅绒黑
  "void",  # 虚空色
  "platinum",  # 铂金色
  "chrome",  # 铬色
  "moonstone",  # 月光石色
  "starlight",  # 星光色
  "smoke gray",  # 烟雾灰
  "gunmetal",  # 枪灰色
  "nightfall",  # 夜幕色
  "blackout",  # 黑暗色
  "deep space",  # 深空黑
  "stone",  # 石色
  "frost",  # 霜白色
  "graphite gray",  # 石墨灰
  "plumb",  # 紫褐色
  "shale",  # 页岩色
  "dust",  # 灰尘色
  "twilight",  # 黄昏色
  "carbon black",  # 碳黑色
  "slate gray",  # 石板灰
  "meteor",  # 流星色
  "shadow gray",  # 阴影灰
  "midnight blue",  # 午夜蓝
  "steel gray",  # 钢灰色
  "shadow black",  # 阴影黑
  "mink",  # 水貂色
  "space gray",  # 太空灰
  "carbon gray",  # 碳灰色
  "stone gray",  # 石灰灰
  "icy black",  # 冰霜黑
]


def 八的倍数(num):
  return num + 8 - num % 8


width = 2880 // 4
height = 1778 // 4

height = 八的倍数(height)
width = 八的倍数(width)

print(f"宽 {width} 高 {height}")

model_id = "Freepik/flux.1-lite-8B-alpha"
# adapter_id = "alimama-creative/FLUX.1-Turbo-Alpha"

root = dirname(abspath(__file__))

with open(join(root, "prompt.yml"), "r") as file:
  prompt_li = yaml.safe_load(file)

prompt_li_len = len(prompt_li)

n_steps = 28
num_images_per_prompt = 1
device = "mps"

guidance_scale = 3.5

torch_dtype = torch.bfloat16

pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)

# pipe.load_lora_weights(adapter_id)
# pipe.fuse_lora()

output_dir = join(root, "out")

os.makedirs(output_dir, exist_ok=True)

file_id = int(time() * 100)

for i in range(10000):
  en, zh = prompt_li[i % prompt_li_len]

  en = choice(("gradient ", "")) + en

  if choice((0, 1)):
    en = choice(分形) + "," + en

  if choice((0, 1)):
    en = choice(tones) + " tones," + en

  en = choice(("realistic,", "abstract scene,", "")) + en

  images = pipe(
    prompt=en,
    guidance_scale=guidance_scale,
    height=height,
    width=width,
    num_inference_steps=n_steps,
    num_images_per_prompt=num_images_per_prompt,
  ).images

  for image in images:
    file_id += 1
    name = base36.encode(file_id)
    fp = join(output_dir, f"{name}.png")
    print(f"\n{name}\n{en}\n{zh}\n")

    # 创建 PngInfo 对象并添加元信息
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("en", en)
    metadata.add_text("zh", zh)

    # 保存图像并嵌入元信息
    image.save(fp, "PNG", pnginfo=metadata)
    del image

print("done")
