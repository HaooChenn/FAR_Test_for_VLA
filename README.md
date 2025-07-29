# Frequency Autoregressive Image Generation with Continuous Tokens <br><sub>Official PyTorch Implementation</sub>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2503.05305-b31b1b.svg)](https://arxiv.org/abs/2503.05305)&nbsp;
[![project page](https://img.shields.io/badge/Project_page-More_demo-green)](https://yuhuustc.github.io//projects/FAR.html)&nbsp;
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-far-yellow)](https://huggingface.co/figereatfish/FAR)&nbsp;
<p align="center">
  <img src="demo/Visual_ImageNet.png" width="720">
</p>

## Test
Installing requirements:
```
pip install transformers==4.35.0 diffusers==0.24.0 matplotlib==3.7.2 Pillow==9.5.0 tqdm==4.66.1 IPython==8.12.2
```
### 📊 评估任务对比

| 评估类型 | 条件输入 | 模型架构 | 评估目标 |
|---------|---------|---------|---------|
| **ImageNet类别条件生成** | 类别ID (0-999) | Vision Transformer | 测试对特定物体类别的生成质量 |
| **文本到图像生成** | 自然语言描述 | 交叉注意力Transformer | 测试对复杂文本描述的理解能力 |

## 🗂️ 文件结构

```
FAR/
├── run_complete_evaluation.sh          # 主评估脚本
├── imagenet_class_evaluation.py        # ImageNet类别评估专用脚本
├── main_far_t2i.py                    # T2I评估脚本 (原有)
├── prompts/                           # 提示词和类别定义
│   ├── imagenet_classes.txt           # 20个精选ImageNet类别
│   ├── simple_prompts.txt             # 简单文本提示词
│   ├── medium_prompts.txt             # 中等复杂度提示词
│   └── complex_prompts.txt            # 复杂提示词
└── complete_evaluation_results/        # 评估结果
    ├── imagenet_far_base_steps50/     # ImageNet评估结果
    ├── imagenet_far_large_steps50/
    ├── imagenet_far_huge_steps50/
    ├── t2i_steps50_simple/            # T2I评估结果
    ├── t2i_steps50_medium/
    └── t2i_steps50_complex/
```

## 🚀 快速开始

### 第一步：确保权重文件就位
```bash
# 检查权重文件是否存在
ls pretrained_models/far/far_base/checkpoint-last.pth      # FAR Base模型
ls pretrained_models/far/far_large/checkpoint-last.pth     # FAR Large模型  
ls pretrained_models/far/far_huge/checkpoint-last.pth      # FAR Huge模型
ls pretrained_models/far/far_t2i/checkpoint-last.pth       # FAR T2I模型

# 检查VAE权重
ls pretrained/vae/kl16.ckpt

# 检查文本编码器
ls pretrained/Qwen2-VL-1.5B-Instruct/
```

### 第二步：准备评估脚本
```bash
# 给脚本执行权限
chmod +x run_complete_evaluation.sh

# 将ImageNet评估脚本放在正确位置
cp imagenet_class_evaluation.py ./
```

### 第三步：运行完整评估
```bash
# 运行完整评估 (大约需要2-3小时)
./run_complete_evaluation.sh
```

## 📈 评估内容详解

### Phase 1: ImageNet类别条件生成
**测试模型：** FAR Base, Large, Huge  
**测试类别：** 20个精心选择的ImageNet类别，涵盖：
- **动物类**：金毛寻回犬、虎斑猫、红狐、帝王蝶
- **植物类**：雏菊、玫瑰
- **建筑类**：灯塔、城堡、小屋
- **交通工具**：跑车、蒸汽机车、帆船、航空母舰、山地自行车
- **食物类**：披萨、草莓
- **日用品**：咖啡杯、小提琴、背包、雨伞

**评估指标：**
- 生成质量 (视觉质量、类别一致性)
- 生成速度 (每张图片用时、吞吐量)
- 模型可扩展性 (不同规模模型的性能对比)

### Phase 2: 文本到图像生成
**测试模型：** FAR T2I  
**测试提示词：** 3种复杂度 × 20个提示词 = 60个测试用例

#### 提示词复杂度分析：
1. **简单提示词** - 测试基础物体生成
   - 例：`"A red apple on a white counter"`
   - 考察：基本物体识别、颜色理解、空间关系

2. **中等提示词** - 测试场景组合能力  
   - 例：`"A laptop next to a coffee cup on a desk, morning workspace"`
   - 考察：多物体组合、环境理解、氛围营造

3. **复杂提示词** - 测试细节描述理解
   - 例：`"A chef's kitchen with ingredients on cutting boards, pans on stove..."`
   - 考察：复杂场景理解、细节还原、逻辑一致性

## 📊 结果分析指南

### 定量指标
- **生成速度**：每张图片生成时间、GPU利用率
- **内存使用**：峰值显存占用、内存效率
- **一致性**：同类别/同提示词生成的一致性

### 定性分析
- **视觉质量**：清晰度、真实感、艺术性
- **语义准确性**：是否符合输入条件的要求
- **创造性**：生成结果的多样性和新颖性

### 对比维度
1. **模型规模效应**：Base vs Large vs Huge在质量和速度上的权衡
2. **条件类型影响**：类别条件 vs 文本条件的生成效果差异
3. **复杂度处理**：简单 vs 复杂提示词的处理能力
4. **采样步数影响**：50步 vs 100步的质量-速度权衡

## 🔧 自定义配置

### 修改测试类别
编辑 `prompts/imagenet_classes.txt`：
```
# 添加新的ImageNet类别
your_class_name:class_id
```

### 修改测试提示词
编辑对应的提示词文件：
- `prompts/simple_prompts.txt`
- `prompts/medium_prompts.txt`  
- `prompts/complex_prompts.txt`

### 调整评估参数
在 `run_complete_evaluation.sh` 中修改：
```bash
EVAL_BSZ=8              # 批次大小
SAMPLING_STEPS=(50 100) # 采样步数
CFG=3.0                 # 分类器自由指导强度
TEMPERATURE=1.0         # 采样温度
```

## 📰 News

- [2025-3-7] We release the code and checkpoint of `FAR` for class-to-image generation on ImageNet dataset.
- [2025-3-7] The [tech report](https://arxiv.org/abs/2503.05305) of `FAR` is available.


## Preparation

### Installation

Download the code:
```
git clone https://github.com/yuhuUSTC/FAR.git
cd FAR
```

A suitable [conda](https://conda.io/) environment named `far` can be created and activated with:

```
conda env create -f environment.yaml
conda activate far
```

### Dataset
Download [ImageNet](http://image-net.org/download) dataset, and place it in your `IMAGENET_PATH`.


### Pretrained Weights
Download pre-trained [VAE](https://huggingface.co/figereatfish/FAR/tree/main/vae), and place it in `/pretrained/vae/`.

Download [.npz](https://huggingface.co/figereatfish/FAR/tree/main/fid_stats) of ImageNet 256x256 for calculating the FID metric, and place it in `/fid_stats/`.

Download the weights of [FAR_B](https://huggingface.co/figereatfish/FAR/tree/main), and place it in `/pretrained_models/far/far_base/`.

Download the weights of [FAR_L](https://huggingface.co/figereatfish/FAR/tree/main), and place it in `/pretrained_models/far/far_large/`.

Download the weights of [FAR_H](https://huggingface.co/figereatfish/FAR/tree/main), and place it in `/pretrained_models/far/far_huge/`.

Download the weights of [FAR_T2I](https://huggingface.co/figereatfish/FAR_T2I), and place it in `pretrained_models/far/far_t2i/`.

For convenience, our pre-trained MAR models can be downloaded directly here as well:

| MAR Model                                                              | FID-50K | Inception Score | #params | 
|------------------------------------------------------------------------|---------|-----------------|---------|
| [FAR-B](https://huggingface.co/figereatfish/FAR/tree/main) | 4.83    | 247.4           | 208M    |
| [FAR-L](https://huggingface.co/figereatfish/FAR/tree/main) | 3.92    | 288.9           | 451M    |
| [FAR-H](https://huggingface.co/figereatfish/FAR/tree/main) | 3.71    | 304.9           | 812M    |

### (Optional) Caching VAE Latents

Given that our data augmentation consists of simple center cropping and random flipping, 
the VAE latents can be pre-computed and saved to `CACHED_PATH` to save computations during MAR training:

```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_cache.py \
--img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 \
--batch_size 128 \
--data_path ${IMAGENET_PATH} --cached_path ${CACHED_PATH}
```


## FAR Framework
<p align="center">
  <img src="demo/FAR_framework.png" width="720">
</p>



## Training (ImageNet 256x256)
Run the following command, which contains the scripts for training various model size (FAR-B, FAR-L, FAR-H).
```
bash train.sh
```

Specifically, take the default script for FAR-L for example:
```
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_far.py \
--img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model far_large --diffloss_d 3 --diffloss_w 1024 \
--epochs 400 --warmup_epochs 100 --batch_size 64 --blr 1.0e-4 --diffusion_batch_mul 4 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH}
```
- (Optional) Add `--online_eval` to evaluate FID during training (every 40 epochs).
- (Optional) To enable uneven loss weight strategy, add `--loss_weight` to the arguments. 
- (Optional) To train with cached VAE latents, add `--use_cached --cached_path ${CACHED_PATH}` to the arguments. 





## Evaluation (ImageNet 256x256)
Run the following command, which contains the scripts for the inference of various model size (FAR-B, FAR-L, FAR-H).
```
bash samle.sh
```

Specifically, take the default inference script for FAR-L for example:
```
torchrun --nnodes=1 --nproc_per_node=8  main_far.py \
--img_size 256 --vae_path pretrained/vae_mar/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model far_large --diffloss_d 3 --diffloss_w 1024 \
--eval_bsz 32 --num_images 1000 \
--num_iter 10 --num_sampling_steps 100 --cfg 3.0 --cfg_schedule linear --temperature 1.0 \
--output_dir pretrained_models/far/far_large \
--resume pretrained_models/far/far_large \
--data_path ${IMAGENET_PATH} --evaluate
```
- Add `--mask` to increase the generation diversity.
- We adopt 10 autoregressive steps by default.
- Generation speed can be further increased by reducing the number of diffusion steps (e.g., `--num_sampling_steps 50`).



## Training (T2I)
Script for the default setting:
```
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_far_t2i.py \
--img_size 256 --vae_path pretrained/vae_mar/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model far_t2i --diffloss_d 3 --diffloss_w 1024 \
--epochs 400 --warmup_epochs 100 --batch_size 64 --blr 1.0e-4 --diffusion_batch_mul 4 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--text_model_path pretrained/Qwen2-VL-1.5B-Instruct  \
--data_path ${T2I_PATH}
```

- The `text encoder` employs [Qwen2-VL-1.5B](https://huggingface.co/mit-han-lab/Qwen2-VL-1.5B-Instruct/tree/main), download it and place it in your `pretrained/Qwen2-VL-1.5B-Instruct/`.
- Replace `T2I_PATH` with the path to your Text-to-image dataset path.


## Evaluation (T2I)
Script for the default setting:
```
torchrun --nnodes=1 --nproc_per_node=8  main_far_t2i.py \
--img_size 256 --vae_path pretrained/vae_mar/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model far_t2i --diffloss_d 3 --diffloss_w 1024 \
--eval_bsz 32 \
--num_iter 10 --num_sampling_steps 100 --cfg 3.0 --cfg_schedule linear --temperature 1.0 \
--output_dir pretrained_models/far/far_t2i \
--resume pretrained_models/far/far_t2i \
--text_model_path pretrained/Qwen2-VL-1.5B-Instruct  \
--data_path ${T2I_PATH} --evaluate
```
- Add `--mask` to increase the generation diversity.
- We adopt 10 autoregressive steps by default.
- Generation speed can be further increased by reducing the number of diffusion steps (e.g., `--num_sampling_steps 50`).


## Acknowledgements

A large portion of codes in this repo is based on [MAE](https://github.com/facebookresearch/mae), and [MAR](https://github.com/LTH14/mar). Thanks for these great work and open source。

## Contact

If you have any questions, feel free to contact me through email (yuhu520@mail.ustc.edu.cn). Enjoy!

## Citation
```BibTeX
@article{yu2025frequency,
  author    = {Hu Yu and Hao Luo and Hangjie Yuan and Yu Rong and Feng Zhao},
  title     = {Frequency Autoregressive Image Generation with Continuous Tokens},
  journal   = {arxiv: 2503.05305},
  year      = {2025}
}
```
