# FAR Complete Evaluation - 使用说明

## 评估设计理念

这个评估系统设计用于全面测试FAR模型在两种不同条件生成任务上的表现：

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

## 🎯 预期输出

评估完成后，你将得到：
1. **生成的图像样本** - 展示模型的视觉生成能力
2. **详细的性能报告** - JSON和CSV格式的量化指标
3. **对比分析数据** - 不同模型和设置的性能对比
4. **可视化结果** - 便于质量评估的图像网格

这个评估框架不仅帮你测试模型性能，更重要的是让你深入理解不同条件生成任务的特点和挑战。通过对比分析，你可以获得关于模型可扩展性、架构设计影响、以及条件信号有效性的深刻洞察。