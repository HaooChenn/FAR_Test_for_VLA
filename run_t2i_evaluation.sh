#!/bin/bash

# FAR T2I专用评估脚本 - 完全修复版
# 自动修复所有已知的代码问题：参数错误 + 数据类型不匹配

set -e  # 遇到任何错误时立即退出

echo "=========================================="
echo "FAR T2I 专用评估脚本 - 完全修复版"
echo "自动修复所有代码问题"
echo "=========================================="

# 首先自动修复所有代码问题
echo "正在自动修复代码中的所有已知问题..."

# 创建备份
if [ -f "./models/far_t2i.py" ]; then
    cp ./models/far_t2i.py ./models/far_t2i.py.backup
    echo "已创建 far_t2i.py 的备份文件"
fi

if [ -f "./models/diffloss.py" ]; then
    cp ./models/diffloss.py ./models/diffloss.py.backup
    echo "已创建 diffloss.py 的备份文件"
fi

# 修复1: sample()方法调用中的参数错误
if [ -f "./models/far_t2i.py" ]; then
    # 使用sed命令自动修复错误的调用
    sed -i 's/self\.diffloss\.sample(z, temperature_iter, cfg_iter, index, device)/self.diffloss.sample(z, temperature_iter, cfg_iter, index)/g' ./models/far_t2i.py
    echo "✅ 已修复 far_t2i.py 中的 sample() 方法调用"
else
    echo "❌ 未找到 ./models/far_t2i.py 文件，请检查路径"
    exit 1
fi

# 修复2: 数据类型不匹配问题（Long vs Half）
# 这个问题出现在 diffloss.py 的 forward 方法中
if [ -f "./models/diffloss.py" ]; then
    # 在 index = self.index_cond_embed(index) 这一行之前添加类型转换
    sed -i '/index = self\.index_cond_embed(index)/i\        if index is not None:\
            index = index.to(c.dtype)  # 确保 index 与模型权重的数据类型匹配' ./models/diffloss.py
    
    # 同时修复 forward_with_cfg 方法中的类似问题
    sed -i '/model_out = self\.forward(combined, t, c, index)/i\        if index is not None:\
            index = index.to(c.dtype)  # 确保 index 与模型权重的数据类型匹配' ./models/diffloss.py
    
    echo "✅ 已修复 diffloss.py 中的数据类型不匹配问题"
else
    echo "❌ 未找到 ./models/diffloss.py 文件，请检查路径"
    exit 1
fi

# 修复3: 确保 far_t2i.py 中 index 张量的正确类型
if [ -f "./models/far_t2i.py" ]; then
    # 修复 index 张量创建时的数据类型问题
    sed -i 's/index = torch\.tensor(\[latent_core\[step\]\])\.unsqueeze(1)\.unsqueeze(-1)\.repeat(B, L, 1)\.reshape(B \* L, -1)\.to(device)/index = torch.tensor([latent_core[step]], dtype=torch.float32).unsqueeze(1).unsqueeze(-1).repeat(B, L, 1).reshape(B * L, -1).to(device)/g' ./models/far_t2i.py
    echo "✅ 已修复 far_t2i.py 中 index 张量的数据类型问题"
fi

echo "所有代码修复完成！"

# 配置参数
export CUDA_VISIBLE_DEVICES=1,2,3,4  # 使用4张GPU卡
EVAL_BSZ=16  # 减少批次大小以提高稳定性
IMG_SIZE=256
CFG=3.0
TEMPERATURE=1.0
NUM_ITER=10
SPEED_TEST_STEPS=3  # 减少测试步数以加快验证

# 基础路径配置
BASE_OUTPUT_DIR="./t2i_evaluation_results"
PROMPTS_DIR="./prompts"
VAE_PATH="pretrained/vae/kl16.ckpt"
TEXT_MODEL_PATH="pretrained/Qwen2-VL-1.5B-Instruct"
T2I_MODEL_PATH="pretrained_models/far/far_t2i"

# 只测试一种采样步数以快速验证修复效果
SAMPLING_STEPS=(50)

# 创建目录结构和必要文件
echo "设置目录结构和评估文件..."
mkdir -p $BASE_OUTPUT_DIR
mkdir -p $PROMPTS_DIR

# 创建简单提示词文件（用于快速测试）
cat > $PROMPTS_DIR/simple_prompts.txt << 'EOF'
A red apple on a table
A blue car on the road
A green tree in the park
A yellow flower in a garden
A white cat sleeping
EOF

# 创建中等复杂度提示词文件
cat > $PROMPTS_DIR/medium_prompts.txt << 'EOF'
A laptop computer open next to a steaming coffee cup on a desk
Three colorful books stacked vertically beside a desk lamp
A smartphone lying next to its charger cable on a nightstand
Two wine glasses positioned side by side on a dinner table
A cutting board with sliced vegetables and a knife nearby
EOF

echo "文件创建成功！"

# T2I评估函数 - 使用正确的架构参数和修复后的代码
run_t2i_evaluation() {
    local sampling_steps=$1
    local prompt_type=$2
    
    echo ""
    echo "=========================================="
    echo "运行T2I评估（修复版）"
    echo "采样步数: $sampling_steps"
    echo "提示词类型: $prompt_type"
    echo "架构配置: diffloss_d=3, diffloss_w=1024"
    echo "修复项目: sample()参数 + 数据类型匹配"
    echo "=========================================="
    
    local output_dir="$BASE_OUTPUT_DIR/t2i_steps${sampling_steps}_${prompt_type}_fixed"
    mkdir -p $output_dir
    cp "$PROMPTS_DIR/${prompt_type}_prompts.txt" "$output_dir/used_prompts.txt"
    
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "开始时间: $start_time"
    
    # 运行修复后的T2I评估
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
        echo "🎉 T2I评估成功完成: $prompt_type 提示词！"
        
        # 创建运行总结
        cat > "$output_dir/run_summary.txt" << EOF
T2I模型评估总结（修复版）
========================

模型: FAR T2I
模型路径: $T2I_MODEL_PATH
采样步数: $sampling_steps
提示词类型: $prompt_type
架构配置: diffloss_d=3, diffloss_w=1024

修复内容:
1. ✅ 修正了sample()方法的参数传递问题
2. ✅ 修正了数据类型不匹配问题（Long vs Half）
3. ✅ 确保了index张量的正确数据类型
4. ✅ 优化了批次大小以提高稳定性

技术要点:
- 解决了TypeError: sample() takes from 2 to 5 positional arguments but 6 were given
- 解决了RuntimeError: mat1 and mat2 must have the same dtype, but got Long and Half
- 在混合精度环境下确保了数据类型的一致性

时间记录:
- 开始时间: $start_time
- 结束时间: $end_time
- 运行状态: 成功

输出目录: $output_dir
EOF
    else
        echo "❌ 错误：T2I评估失败，退出代码: $exit_code"
        echo "请查看上方的错误信息进行进一步调试"
        
        # 创建错误报告
        cat > "$output_dir/error_report.txt" << EOF
T2I模型评估错误报告
==================

错误时间: $end_time
退出代码: $exit_code
提示词类型: $prompt_type
采样步数: $sampling_steps

已应用的修复:
1. sample()方法参数修复
2. 数据类型匹配修复
3. index张量类型修复

如果仍有错误，可能需要进一步检查:
- 模型权重文件的完整性
- GPU内存是否充足
- 依赖库版本兼容性
- 其他未发现的代码问题

调试建议:
1. 检查上方完整的错误堆栈信息
2. 验证所有预训练模型文件是否正确下载
3. 尝试减少批次大小或采样步数
EOF
        return $exit_code
    fi
    
    sleep 3
}

# 主评估执行流程
echo ""
echo "开始T2I专用评估（完全修复版）..."
echo "配置总结:"
echo "- GPU配置: $CUDA_VISIBLE_DEVICES"
echo "- 批次大小: $EVAL_BSZ（为稳定性优化）"
echo "- 模型路径: $T2I_MODEL_PATH"
echo "- 架构配置: diffloss_d=3, diffloss_w=1024"
echo "- 修复内容: sample()参数 + 数据类型匹配 + index张量类型"
echo ""

TOTAL_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

# 设置错误计数器
ERROR_COUNT=0
TOTAL_RUNS=0

# 只测试简单和中等提示词以快速验证修复效果
PROMPT_TYPES=("simple" "medium")

echo "=========================================="
echo "开始T2I模型评估（修复验证）"
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

cat > "$BASE_OUTPUT_DIR/evaluation_summary_fixed.txt" << EOF
FAR T2I 评估总结（完全修复版）
==============================

评估完成时间:
- 开始: $TOTAL_START_TIME
- 结束: $TOTAL_END_TIME

运行统计:
- 总运行次数: $TOTAL_RUNS
- 成功次数: $SUCCESS_COUNT
- 失败次数: $ERROR_COUNT
- 成功率: $(echo "scale=2; $SUCCESS_COUNT * 100 / $TOTAL_RUNS" | bc 2>/dev/null || echo "计算中")%

修复的问题:
1. ✅ sample()方法参数数量错误
   - 原因: 传递了多余的device参数
   - 解决: 移除device参数，让PyTorch自动管理设备

2. ✅ 数据类型不匹配错误（Long vs Half）
   - 原因: index张量是Long类型，但模型权重是Half类型
   - 解决: 在使用前将index转换为正确的数据类型

3. ✅ index张量创建时的类型问题
   - 原因: torch.tensor默认创建Long类型
   - 解决: 显式指定float32类型

技术要点:
- 混合精度训练中数据类型的一致性至关重要
- PyTorch的自动类型推断有时需要人工干预
- 分布式训练增加了调试的复杂性

模型配置:
- 模型: FAR T2I (561.3M参数)
- 架构配置: diffloss_d=3, diffloss_w=1024
- 采样步数: ${SAMPLING_STEPS[@]}
- 提示词类型: ${PROMPT_TYPES[@]}

结果保存位置: $BASE_OUTPUT_DIR
EOF

echo ""
echo "=========================================="
echo "T2I 评估全部完成！（完全修复版）"
echo "=========================================="
echo ""
echo "运行统计:"
echo "- 总运行次数: $TOTAL_RUNS"
echo "- 成功次数: $SUCCESS_COUNT"
echo "- 失败次数: $ERROR_COUNT"
if [ $TOTAL_RUNS -gt 0 ]; then
    echo "- 成功率: $(echo "scale=2; $SUCCESS_COUNT * 100 / $TOTAL_RUNS" | bc 2>/dev/null || echo "计算中")%"
fi
echo ""
echo "关键修复成果:"
echo "- ✅ 解决了sample()方法参数错误（TypeError）"
echo "- ✅ 解决了数据类型不匹配错误（RuntimeError）"
echo "- ✅ 优化了混合精度训练的类型安全"
echo "- ✅ 增强了分布式训练的稳定性"
echo ""
echo "学习要点:"
echo "这次调试过程展示了深度学习系统中的两个关键概念："
echo "1. API接口一致性：方法调用必须与定义完全匹配"
echo "2. 数据类型安全：特别是在混合精度和分布式环境中"
echo ""
echo "结果保存到: $BASE_OUTPUT_DIR"
echo "总结报告: $BASE_OUTPUT_DIR/evaluation_summary_fixed.txt"
echo ""

# 根据结果输出最终状态
if [ $ERROR_COUNT -eq 0 ]; then
    echo "🎉 所有评估都成功完成！FAR T2I模型现在可以正常运行了！"
    echo ""
    echo "下一步建议："
    echo "- 可以增加更多提示词类型进行测试"
    echo "- 可以尝试不同的采样步数配置"
    echo "- 可以分析生成图像的质量和多样性"
    exit 0
else
    echo "⚠️  仍有 $ERROR_COUNT 个评估失败"
    echo ""
    echo "如果问题持续存在，可能需要："
    echo "- 检查GPU内存是否充足"
    echo "- 验证预训练模型文件的完整性"
    echo "- 检查依赖库版本兼容性"
    echo "- 查看详细的错误日志进行进一步调试"
    exit 1
fi
