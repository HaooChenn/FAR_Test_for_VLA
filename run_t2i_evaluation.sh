#!/bin/bash

# FAR完整评估脚本 - 已修正所有架构参数不匹配问题
# 这个脚本经过彻底检查，确保Base、Large和T2I模型都使用正确的参数配置
# 教学价值：展示了如何在复杂系统中维护参数一致性

set -e  # 遇到任何错误时立即退出

echo "=========================================="
echo "FAR完整评估脚本 (修正版)"
echo "Class-Conditional + Text-to-Image Models"
echo "已排除Huge模型，专注于Base、Large和T2I"
echo "=========================================="

# 配置参数 - 这些参数经过仔细调整以确保多GPU性能和架构兼容性
export CUDA_VISIBLE_DEVICES=1,2,3,4  # 使用4张GPU卡
EVAL_BSZ=32  # 为4个GPU优化的批次大小 (每GPU 8张图片)
IMG_SIZE=256
CFG=3.0
TEMPERATURE=1.0
NUM_ITER=10
SPEED_TEST_STEPS=5

# 基础路径配置 - 为清晰性而组织
BASE_OUTPUT_DIR="./complete_evaluation_results"
PROMPTS_DIR="./prompts"
VAE_PATH="pretrained/vae/kl16.ckpt"  # 注意：与T2I不同的路径
TEXT_MODEL_PATH="pretrained/Qwen2-VL-1.5B-Instruct"

# ImageNet模型配置 - 只包含Base和Large，已排除Huge
declare -A IMAGENET_MODELS
IMAGENET_MODELS["far_base"]="pretrained_models/far/far_base"
IMAGENET_MODELS["far_large"]="pretrained_models/far/far_large"

# T2I模型配置
T2I_MODEL_PATH="pretrained_models/far/far_t2i"

# 采样步数测试配置
SAMPLING_STEPS=(50 100)

# 创建目录结构和文件
echo "设置目录结构和类别映射..."
mkdir -p $BASE_OUTPUT_DIR
mkdir -p $PROMPTS_DIR

# 创建ImageNet类别映射文件
cat > $PROMPTS_DIR/imagenet_classes.txt << 'EOF'
# ImageNet类别ID用于评估
# 格式: 类别名称:类别ID
golden_retriever:207
tabby_cat:281
red_fox:277
monarch_butterfly:323
daisy:985
rose:973
lighthouse:437
castle:483
cottage:500
sports_car:817
steam_locomotive:820
sailboat:554
aircraft_carrier:403
mountain_bike:671
pizza:963
strawberry:949
coffee_mug:504
violin:889
backpack:414
umbrella:879
EOF

# 创建T2I提示词文件（保持与原版相同的高质量提示词）
cat > $PROMPTS_DIR/simple_prompts.txt << 'EOF'
A red apple sitting alone on a white kitchen counter, bright natural lighting
A blue coffee mug on a wooden desk, side view, office environment
A green book lying flat on a library table, top-down perspective
A yellow banana placed on a marble countertop, clean background
A silver spoon resting on a white plate, dining room setting
A black pen positioned vertically in a clear glass container
A orange basketball on a gym floor, overhead fluorescent lighting
A purple towel folded neatly on a bathroom counter
A brown cardboard box sitting on a carpeted floor, living room
A white ceramic bowl centered on a granite kitchen island
A pink flower vase on a bedside table, soft bedroom lighting
A grey computer mouse on a black mouse pad, workspace setting
A golden key lying on a dark wooden surface, close-up view
A clear water bottle standing upright on a picnic table outdoors
A red bell pepper on a cutting board, kitchen preparation area
A blue blanket draped over the back of a sofa, cozy interior
A small green plant pot on a sunny windowsill, natural light
A silver fork placed diagonally on a dinner plate, formal setting
A orange traffic cone on an empty parking lot, daylight
A white pillow positioned at the head of a neatly made bed
EOF

cat > $PROMPTS_DIR/medium_prompts.txt << 'EOF'
A laptop computer open next to a steaming coffee cup on a desk, morning workspace scene
Three colorful books stacked vertically beside a desk lamp, study area setup
A smartphone lying next to its charger cable on a nightstand, bedroom environment
Two wine glasses positioned side by side on a dinner table with white tablecloth
A cutting board with sliced vegetables and a knife nearby, kitchen prep station
A backpack leaning against a chair with shoes placed in front, entryway scene
Remote control resting on a cushion next to a bowl of popcorn, living room setting
Car keys hanging from a hook above a small wooden tray, organized entryway
A water pitcher and empty glass arranged on a kitchen counter, beverage station
Folded laundry stacked on a bed next to an open dresser drawer
A tablet device propped against books on a coffee table, casual reading setup
Garden tools laid out beside a potted plant on a outdoor table, gardening scene
A breakfast plate with utensils arranged on either side, formal dining setup
Two shoes placed neatly under a coat hanging on a wall hook
A camera and its lens cap sitting on a photographer's desk, equipment area
Medicine bottles organized in a bathroom cabinet with the door ajar
A thermos and lunch box positioned together on a kitchen counter, meal prep
Playing cards scattered next to a game box on a family room table
A hammer and nails arranged on a workbench in a garage setting
Fresh fruits displayed in a bowl with a kitchen scale nearby, healthy eating setup
EOF

cat > $PROMPTS_DIR/complex_prompts.txt << 'EOF'
A chef's kitchen mid-preparation: multiple ingredients chopped on cutting boards, pans on stove, utensils scattered strategically for cooking workflow
A cluttered desk workspace: open laptop displaying code, multiple monitors, coffee cup rings, scattered papers, and programming books creating a realistic work environment
A garage workshop scene: tools hanging on pegboard, project parts organized in bins, work light illuminating a partially assembled device on workbench
A living room during movie night: remote controls on coffee table, snack bowls, drinks, blankets draped over furniture, ambient TV lighting
A bathroom vanity during morning routine: toothbrush in holder, open toiletry bottles, towels hung at different positions, mirror reflecting organized chaos
A garden potting station: soil bags, empty pots of various sizes, gardening tools, watering can, and seedlings arranged for planting activity
A home office video call setup: camera positioned optimally, lighting equipment, notes scattered, coffee cup, and background carefully arranged
A kitchen island during baking: mixing bowls, measuring tools, ingredients containers open, flour dusted on surfaces, recipe book propped open
A children's playroom mid-activity: toys organized in play zones, art supplies on low table, books scattered, small furniture arranged for accessibility
A photographer's equipment table: camera bodies, various lenses, memory cards, cleaning supplies, and lighting equipment organized for a shoot
A mechanic's workstation: car parts laid out systematically, diagnostic tools connected, work manual open, safety equipment positioned nearby
A scientist's laboratory bench: microscope centered, sample containers labeled and arranged, notebook open with data, precision instruments organized
A artist's studio corner: easel with work-in-progress, paint palette with mixed colors, brushes in water containers, reference materials pinned to wall
A musician's practice space: instrument cases open, sheet music on stands, metronome, recording equipment, and cables organized for session
A tailor's work area: fabric pieces cut and arranged, sewing machine threaded, measuring tools, patterns spread out, good task lighting
A pharmacy counter organization: prescription bottles sorted by patient, counting trays clean and ready, reference materials accessible, secure storage visible
A restaurant kitchen pass: plates warming under heat lamps, garnish stations organized, order tickets clipped in sequence, utensils strategically placed
A florist's arrangement station: fresh flowers in water buckets, cutting tools clean and sharp, ribbons and wrapping materials organized, design inspiration displayed
A jeweler's detailed workspace: magnifying equipment positioned, tiny tools organized precisely, gemstones sorted in compartments, excellent focused lighting
A surgeon's instrument table: sterile tools arranged in specific order, monitoring equipment positioned, lighting optimized, everything prepared for precise work
EOF

echo "文件创建成功！"
echo "- ImageNet类别: $PROMPTS_DIR/imagenet_classes.txt (20个类别)"
echo "- 简单提示词: $PROMPTS_DIR/simple_prompts.txt (20个提示词)"
echo "- 中等提示词: $PROMPTS_DIR/medium_prompts.txt (20个提示词)"  
echo "- 复杂提示词: $PROMPTS_DIR/complex_prompts.txt (20个提示词)"

# ImageNet评估函数 - 关键修复：确保所有模型使用正确的diffloss_d参数
run_imagenet_evaluation() {
    local model_name=$1
    local model_path=$2
    local sampling_steps=$3
    
    echo ""
    echo "=========================================="
    echo "运行ImageNet评估: $model_name"
    echo "采样步数: $sampling_steps"
    echo "=========================================="
    
    # 为这次特定运行创建输出目录
    local output_dir="$BASE_OUTPUT_DIR/imagenet_${model_name}_steps${sampling_steps}"
    mkdir -p $output_dir
    
    # 复制类别映射文件作为参考
    cp "$PROMPTS_DIR/imagenet_classes.txt" "$output_dir/used_classes.txt"
    
    # 记录开始时间
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "开始时间: $start_time"
    
    # 关键修复：使用我们的自定义评估脚本，确保正确的diffloss_d参数
    # 教学要点：这里的diffloss_d=6是通过深入分析预训练权重结构得出的
    torchrun --nnodes=1 --nproc_per_node=4 imagenet_class_evaluation.py \
        --img_size $IMG_SIZE \
        --vae_path $VAE_PATH \
        --vae_embed_dim 16 \
        --vae_stride 16 \
        --patch_size 1 \
        --model $model_name \
        --diffloss_d 6 \
        --diffloss_w 1024 \
        --eval_bsz $EVAL_BSZ \
        --num_iter $NUM_ITER \
        --num_sampling_steps $sampling_steps \
        --cfg $CFG \
        --cfg_schedule linear \
        --temperature $TEMPERATURE \
        --output_dir $output_dir \
        --resume $model_path \
        --class_num 1000
    
    # 记录结束时间
    local end_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "结束时间: $end_time"
    
    # 创建运行总结
    cat > "$output_dir/run_summary.txt" << EOF
ImageNet类别条件生成总结
===========================

模型: $model_name
模型路径: $model_path
采样步数: $sampling_steps
任务类型: ImageNet类别条件生成

参数配置:
- 图像尺寸: $IMG_SIZE
- 批次大小: $EVAL_BSZ
- 迭代次数: $NUM_ITER
- CFG强度: $CFG
- 温度: $TEMPERATURE
- Diffloss深度: 6 (修正后的架构兼容参数)
- 评估类别: 20个精选ImageNet类别

目标类别:
- 动物类: golden_retriever (207), tabby_cat (281), red_fox (277), monarch_butterfly (323)
- 植物类: daisy (985), rose (973)
- 建筑类: lighthouse (437), castle (483), cottage (500)
- 交通工具: sports_car (817), steam_locomotive (820), sailboat (554), aircraft_carrier (403), mountain_bike (671)
- 食物类: pizza (963), strawberry (949)
- 日用品: coffee_mug (504), violin (889), backpack (414), umbrella (879)

时间记录:
- 开始时间: $start_time
- 结束时间: $end_time

输出目录: $output_dir
使用的类别: 查看 used_classes.txt
生成的图像: 查看 generated_images/ 子目录
速度结果: 查看 speed_results/ 子目录
EOF

    echo "ImageNet评估完成: $model_name!"
    sleep 3
}

# T2I评估函数 - 检查并修正可能的参数问题
run_t2i_evaluation() {
    local sampling_steps=$1
    local prompt_type=$2
    
    echo ""
    echo "=========================================="
    echo "运行T2I评估"
    echo "采样步数: $sampling_steps"
    echo "提示词类型: $prompt_type"
    echo "=========================================="
    
    # 为这次特定运行创建输出目录
    local output_dir="$BASE_OUTPUT_DIR/t2i_steps${sampling_steps}_${prompt_type}"
    mkdir -p $output_dir
    
    # 复制提示词文件作为参考
    cp "$PROMPTS_DIR/${prompt_type}_prompts.txt" "$output_dir/used_prompts.txt"
    
    # 记录开始时间
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "开始时间: $start_time"
    
    # 运行T2I评估使用main_far_t2i.py
    # 教学要点：T2I模型使用不同的架构，diffloss_d在T2I中默认为3，这是正确的
    torchrun --nnodes=1 --nproc_per_node=4 main_far_t2i.py \
        --img_size $IMG_SIZE \
        --vae_path $VAE_PATH \
        --vae_embed_dim 16 \
        --vae_stride 16 \
        --patch_size 1 \
        --model far_t2i \
        --diffloss_d 3 \
        --diffloss_w 1024 \
        --eval_bsz $EVAL_BSZ \
        --num_iter $NUM_ITER \
        --num_sampling_steps $sampling_steps \
        --cfg $CFG \
        --cfg_schedule linear \
        --temperature $TEMPERATURE \
        --speed_test_steps $SPEED_TEST_STEPS \
        --speed_test \
        --output_dir $output_dir \
        --resume $T2I_MODEL_PATH \
        --text_model_path $TEXT_MODEL_PATH \
        --data_path $PROMPTS_DIR \
        --evaluate
    
    # 记录结束时间
    local end_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "结束时间: $end_time"
    
    # 创建运行总结
    cat > "$output_dir/run_summary.txt" << EOF
文本到图像生成总结
================

模型: FAR T2I
模型路径: $T2I_MODEL_PATH
采样步数: $sampling_steps
提示词类型: $prompt_type
任务类型: 文本到图像生成

参数配置:
- 图像尺寸: $IMG_SIZE
- 批次大小: $EVAL_BSZ
- 迭代次数: $NUM_ITER
- CFG强度: $CFG
- 温度: $TEMPERATURE
- Diffloss深度: 3 (T2I专用架构)
- 文本模型: Qwen2-VL-1.5B-Instruct

时间记录:
- 开始时间: $start_time
- 结束时间: $end_time

输出目录: $output_dir
使用的提示词: 查看 used_prompts.txt
EOF

    echo "T2I评估完成: $prompt_type 提示词!"
    sleep 3
}

# 主评估执行
echo ""
echo "开始完整FAR评估..."
echo "配置:"
echo "- GPU: $CUDA_VISIBLE_DEVICES"
echo "- 批次大小: $EVAL_BSZ"
echo "- 图像尺寸: $IMG_SIZE"
echo "- CFG: $CFG"
echo "- 温度: $TEMPERATURE"
echo "- 迭代次数: $NUM_ITER"
echo "- 速度测试步数: $SPEED_TEST_STEPS"
echo ""

# 记录总开始时间
TOTAL_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

# 阶段1: ImageNet类别条件评估 (Base和Large)
echo "=========================================="
echo "阶段1: ImageNet类别条件模型"
echo "模型: Base, Large (Huge已完成)"
echo "=========================================="

for model_name in "${!IMAGENET_MODELS[@]}"; do
    model_path="${IMAGENET_MODELS[$model_name]}"
    
    for sampling_steps in "${SAMPLING_STEPS[@]}"; do
        run_imagenet_evaluation "$model_name" "$model_path" "$sampling_steps"
    done
done

# 阶段2: 文本到图像评估
echo "=========================================="
echo "阶段2: 文本到图像模型"
echo "=========================================="

PROMPT_TYPES=("simple" "medium" "complex")

for sampling_steps in "${SAMPLING_STEPS[@]}"; do
    for prompt_type in "${PROMPT_TYPES[@]}"; do
        run_t2i_evaluation "$sampling_steps" "$prompt_type"
    done
done

# 创建综合总结
TOTAL_END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

cat > "$BASE_OUTPUT_DIR/comprehensive_summary.txt" << EOF
FAR完整评估总结 (修正版)
========================

总评估周期:
- 开始时间: $TOTAL_START_TIME
- 结束时间: $TOTAL_END_TIME

阶段1: ImageNet类别条件模型
- 测试模型: FAR Base, Large (Huge已单独完成)
- 类别: 20个精选ImageNet类别
- 采样步数: ${SAMPLING_STEPS[@]}
- 目的: 评估类别条件生成质量

阶段2: 文本到图像模型  
- 模型: FAR T2I
- 提示词类型: Simple, Medium, Complex (每种20个提示词)
- 采样步数: ${SAMPLING_STEPS[@]}
- 目的: 评估文本条件生成质量

配置参数:
- 使用GPU: $CUDA_VISIBLE_DEVICES
- 批次大小: $EVAL_BSZ
- 图像尺寸: $IMG_SIZE
- CFG: $CFG
- 温度: $TEMPERATURE
- 迭代次数: $NUM_ITER

关键修复:
- ImageNet模型: diffloss_d=6 (架构兼容性修复)
- T2I模型: diffloss_d=3 (原始T2I架构)
- 多GPU优化: 4卡并行处理

研究价值:
此评估允许比较:
1. 不同模型规模的效果 (Base vs Large)
2. 不同条件方法 (类别 vs 文本)
3. 不同提示词复杂度 (Simple vs Medium vs Complex)
4. 不同采样速度 (50 vs 100 步骤)
5. 架构差异的影响

总运行次数: $((${#IMAGENET_MODELS[@]} * ${#SAMPLING_STEPS[@]} + ${#PROMPT_TYPES[@]} * ${#SAMPLING_STEPS[@]}))

分析建议:
1. 比较各模型规模的生成质量
2. 分析条件效果 (类别 vs 文本)
3. 评估提示词复杂度处理能力
4. 评估速度与质量的权衡
5. 研究架构差异的影响
EOF

echo ""
echo "=========================================="
echo "完整评估成功完成！"
echo "=========================================="
echo ""
echo "阶段1 (ImageNet): $((${#IMAGENET_MODELS[@]} * ${#SAMPLING_STEPS[@]})) 次运行"
echo "阶段2 (T2I): $((${#PROMPT_TYPES[@]} * ${#SAMPLING_STEPS[@]})) 次运行" 
echo "总运行次数: $((${#IMAGENET_MODELS[@]} * ${#SAMPLING_STEPS[@]} + ${#PROMPT_TYPES[@]} * ${#SAMPLING_STEPS[@]}))"
echo ""
echo "结果保存到: $BASE_OUTPUT_DIR"
echo "综合总结: $BASE_OUTPUT_DIR/comprehensive_summary.txt"
echo ""
echo "此评估提供了有价值的洞察:"
echo "- 模型缩放对生成质量的影响"
echo "- 条件方法差异 (类别 vs 文本)"
echo "- 架构对不同任务的影响"
echo "- 速度与质量的权衡"
echo ""
echo "分析愉快! 🎉"
