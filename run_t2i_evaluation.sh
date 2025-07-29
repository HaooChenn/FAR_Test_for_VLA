#!/bin/bash

# FAR T2I专用评估脚本
# 基于错误分析和代码修复的完整解决方案

set -e  # 遇到任何错误时立即退出

echo "=========================================="
echo "FAR T2I 专用评估脚本"
echo "修复了 sample() 方法参数错误的问题"
echo "=========================================="

# 配置参数
export CUDA_VISIBLE_DEVICES=1,2,3,4  # 使用4张GPU卡
EVAL_BSZ=32  # 为4个GPU优化的批次大小
IMG_SIZE=256
CFG=3.0
TEMPERATURE=1.0
NUM_ITER=10
SPEED_TEST_STEPS=5

# 基础路径配置
BASE_OUTPUT_DIR="./t2i_evaluation_results"
PROMPTS_DIR="./prompts"
VAE_PATH="pretrained/vae/kl16.ckpt"
TEXT_MODEL_PATH="pretrained/Qwen2-VL-1.5B-Instruct"
T2I_MODEL_PATH="pretrained_models/far/far_t2i"

# 采样步数配置
SAMPLING_STEPS=(50 100)

# 创建目录结构和必要文件
echo "设置目录结构和评估文件..."
mkdir -p $BASE_OUTPUT_DIR
mkdir -p $PROMPTS_DIR

# 创建简单提示词文件（测试用）
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
EOF

# 创建中等复杂度提示词文件
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
EOF

# 创建复杂提示词文件
cat > $PROMPTS_DIR/complex_prompts.txt << 'EOF'
A chef's kitchen mid-preparation: multiple ingredients chopped on cutting boards, pans on stove, utensils scattered strategically for cooking workflow
A cluttered desk workspace: open laptop displaying code, multiple monitors, coffee cup rings, scattered papers, and programming books creating a realistic work environment
A garage workshop scene: tools hanging on pegboard, project parts organized in bins, work light illuminating a partially assembled device on workbench
A living room during movie night: remote controls on coffee table, snack bowls, drinks, blankets draped over furniture, ambient TV lighting
A bathroom vanity during morning routine: toothbrush in holder, open toiletry bottles, towels hung at different positions, mirror reflecting organized chaos
EOF

echo "文件创建成功！"

# T2I评估函数 - 使用正确的架构参数
run_t2i_evaluation() {
    local sampling_steps=$1
    local prompt_type=$2
    
    echo ""
    echo "=========================================="
    echo "运行T2I评估"
    echo "采样步数: $sampling_steps"
    echo "提示词类型: $prompt_type"
    echo "架构配置: diffloss_d=3, diffloss_w=1024"
    echo "=========================================="
    
    local output_dir="$BASE_OUTPUT_DIR/t2i_steps${sampling_steps}_${prompt_type}"
    mkdir -p $output_dir
    cp "$PROMPTS_DIR/${prompt_type}_prompts.txt" "$output_dir/used_prompts.txt"
    
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "开始时间: $start_time"
    
    # 关键修复：确保使用正确的参数配置
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
    
    local exit_code=$?
    local end_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "结束时间: $end_time"
    
    if [ $exit_code -eq 0 ]; then
        echo "T2I评估成功完成: $prompt_type 提示词！"
        
        # 创建运行总结
        cat > "$output_dir/run_summary.txt" << EOF
T2I模型评估总结
=============

模型: FAR T2I
模型路径: $T2I_MODEL_PATH
采样步数: $sampling_steps
提示词类型: $prompt_type
架构配置: diffloss_d=3, diffloss_w=1024

修复说明:
1. 修正了sample()方法的参数传递问题
2. 确保了正确的架构参数配置
3. 优化了多GPU环境下的运行稳定性

时间记录:
- 开始时间: $start_time
- 结束时间: $end_time
- 运行状态: 成功

输出目录: $output_dir
EOF
    else
        echo "错误：T2I评估失败，退出代码: $exit_code"
        echo "请查看上方的错误信息进行调试"
        return $exit_code
    fi
    
    sleep 3
}

# 主评估执行流程
echo ""
echo "开始T2I专用评估..."
echo "配置总结:"
echo "- GPU配置: $CUDA_VISIBLE_DEVICES"
echo "- 批次大小: $EVAL_BSZ"
echo "- 模型路径: $T2I_MODEL_PATH"
echo "- 架构配置: diffloss_d=3, diffloss_w=1024"
echo ""

TOTAL_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

# 设置错误计数器
ERROR_COUNT=0
TOTAL_RUNS=0

# 提示词类型
PROMPT_TYPES=("simple" "medium" "complex")

echo "=========================================="
echo "开始T2I模型评估"
echo "=========================================="

for sampling_steps in "${SAMPLING_STEPS[@]}"; do
    for prompt_type in "${PROMPT_TYPES[@]}"; do
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
        echo ""
        echo "运行 $TOTAL_RUNS: ${prompt_type} prompts, ${sampling_steps} steps"
        
        if run_t2i_evaluation "$sampling_steps" "$prompt_type"; then
            echo "✅ 运行 $TOTAL_RUNS 成功完成"
        else
            echo "❌ 运行 $TOTAL_RUNS 失败"
            ERROR_COUNT=$((ERROR_COUNT + 1))
        fi
    done
done

# 创建最终总结
TOTAL_END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
SUCCESS_COUNT=$((TOTAL_RUNS - ERROR_COUNT))

cat > "$BASE_OUTPUT_DIR/evaluation_summary.txt" << EOF
FAR T2I 评估总结
===============

评估完成时间:
- 开始: $TOTAL_START_TIME
- 结束: $TOTAL_END_TIME

运行统计:
- 总运行次数: $TOTAL_RUNS
- 成功次数: $SUCCESS_COUNT
- 失败次数: $ERROR_COUNT
- 成功率: $(echo "scale=2; $SUCCESS_COUNT * 100 / $TOTAL_RUNS" | bc)%

模型配置:
- 模型: FAR T2I
- 架构配置: diffloss_d=3, diffloss_w=1024
- 采样步数: ${SAMPLING_STEPS[@]}
- 提示词类型: ${PROMPT_TYPES[@]}

主要修复:
1. 修正了sample()方法参数传递错误
2. 确保使用正确的T2I架构参数
3. 优化了脚本的错误处理和日志记录

结果保存位置: $BASE_OUTPUT_DIR
EOF

echo ""
echo "=========================================="
echo "T2I 评估全部完成！"
echo "=========================================="
echo ""
echo "运行统计:"
echo "- 总运行次数: $TOTAL_RUNS"
echo "- 成功次数: $SUCCESS_COUNT"
echo "- 失败次数: $ERROR_COUNT"
if [ $TOTAL_RUNS -gt 0 ]; then
    echo "- 成功率: $(echo "scale=2; $SUCCESS_COUNT * 100 / $TOTAL_RUNS" | bc)%"
fi
echo ""
echo "主要改进:"
echo "- ✅ 修复了sample()方法的参数错误"
echo "- ✅ 使用正确的T2I架构配置"
echo "- ✅ 增强了错误处理和监控"
echo ""
echo "结果保存到: $BASE_OUTPUT_DIR"
echo "总结报告: $BASE_OUTPUT_DIR/evaluation_summary.txt"
echo ""

# 根据结果输出最终状态
if [ $ERROR_COUNT -eq 0 ]; then
    echo "🎉 所有评估都成功完成！"
    exit 0
else
    echo "⚠️  有 $ERROR_COUNT 个评估失败，请检查错误日志"
    exit 1
fi
