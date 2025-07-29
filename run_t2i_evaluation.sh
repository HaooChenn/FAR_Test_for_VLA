#!/bin/bash

# FAR Large和T2I专用评估脚本 - 架构参数完全修正版
# 这个脚本基于深入的错误分析，为每个模型使用了正确的架构参数
# 教学价值：展示了如何根据预训练权重的实际结构调整模型配置

set -e  # 遇到任何错误时立即退出

echo "=========================================="
echo "FAR Large + T2I 专用评估脚本"
echo "基于架构分析的精确配置版本"
echo "=========================================="

# 配置参数 - 经过优化的多GPU设置
export CUDA_VISIBLE_DEVICES=1,2,3,4  # 使用4张GPU卡
EVAL_BSZ=32  # 为4个GPU优化的批次大小
IMG_SIZE=256
CFG=3.0
TEMPERATURE=1.0
NUM_ITER=10
SPEED_TEST_STEPS=5

# 基础路径配置
BASE_OUTPUT_DIR="./large_t2i_evaluation_results"
PROMPTS_DIR="./prompts"
VAE_PATH="pretrained/vae/kl16.ckpt"
TEXT_MODEL_PATH="pretrained/Qwen2-VL-1.5B-Instruct"

# 模型配置 - 只包含Large和T2I
LARGE_MODEL_PATH="pretrained_models/far/far_large"
T2I_MODEL_PATH="pretrained_models/far/far_t2i"

# 采样步数配置
SAMPLING_STEPS=(50 100)

# 创建目录结构和必要文件
echo "设置目录结构和评估文件..."
mkdir -p $BASE_OUTPUT_DIR
mkdir -p $PROMPTS_DIR

# 创建ImageNet类别映射文件
cat > $PROMPTS_DIR/imagenet_classes.txt << 'EOF'
# ImageNet类别ID用于Large模型评估
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

# 创建T2I提示词文件
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

# Large模型评估函数 - 关键修正：使用diffloss_d=3
run_large_evaluation() {
    local sampling_steps=$1
    
    echo ""
    echo "=========================================="
    echo "运行FAR Large模型评估"
    echo "采样步数: $sampling_steps"
    echo "架构配置: diffloss_d=3 (基于错误分析的修正)"
    echo "=========================================="
    
    local output_dir="$BASE_OUTPUT_DIR/imagenet_far_large_steps${sampling_steps}"
    mkdir -p $output_dir
    cp "$PROMPTS_DIR/imagenet_classes.txt" "$output_dir/used_classes.txt"
    
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "开始时间: $start_time"
    
    # 关键修正：Large模型使用diffloss_d=3而不是6
    # 这是基于错误信息分析得出的正确配置
    torchrun --nnodes=1 --nproc_per_node=4 imagenet_class_evaluation.py \
        --img_size $IMG_SIZE \
        --vae_path $VAE_PATH \
        --vae_embed_dim 16 \
        --vae_stride 16 \
        --patch_size 1 \
        --model far_large \
        --diffloss_d 3 \
        --diffloss_w 1024 \
        --eval_bsz $EVAL_BSZ \
        --num_iter $NUM_ITER \
        --num_sampling_steps $sampling_steps \
        --cfg $CFG \
        --cfg_schedule linear \
        --temperature $TEMPERATURE \
        --output_dir $output_dir \
        --resume $LARGE_MODEL_PATH \
        --class_num 1000
    
    local end_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "结束时间: $end_time"
    
    cat > "$output_dir/run_summary.txt" << EOF
FAR Large模型评估总结
==================

模型: FAR Large
模型路径: $LARGE_MODEL_PATH
采样步数: $sampling_steps
架构配置: diffloss_d=3 (修正配置)

参数说明:
基于错误分析，Large模型的预训练权重使用3层残差块架构，
这与Huge模型的6层架构不同，体现了不同规模模型的优化策略。

时间记录:
- 开始时间: $start_time
- 结束时间: $end_time

输出目录: $output_dir
EOF

    echo "FAR Large模型评估完成！"
    sleep 3
}

# T2I评估函数 - 确认使用正确的架构参数
run_t2i_evaluation() {
    local sampling_steps=$1
    local prompt_type=$2
    
    echo ""
    echo "=========================================="
    echo "运行T2I评估"
    echo "采样步数: $sampling_steps"
    echo "提示词类型: $prompt_type"
    echo "架构配置: diffloss_d=3 (T2I专用架构)"
    echo "=========================================="
    
    local output_dir="$BASE_OUTPUT_DIR/t2i_steps${sampling_steps}_${prompt_type}"
    mkdir -p $output_dir
    cp "$PROMPTS_DIR/${prompt_type}_prompts.txt" "$output_dir/used_prompts.txt"
    
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "开始时间: $start_time"
    
    # T2I模型使用diffloss_d=3，这与其预训练架构匹配
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
    
    local end_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "结束时间: $end_time"
    
    cat > "$output_dir/run_summary.txt" << EOF
T2I模型评估总结
=============

模型: FAR T2I
模型路径: $T2I_MODEL_PATH
采样步数: $sampling_steps
提示词类型: $prompt_type
架构配置: diffloss_d=3 (T2I优化架构)

架构说明:
T2I模型使用专门针对文本条件生成优化的架构，
其diffloss_d=3的配置与预训练权重完全匹配。

时间记录:
- 开始时间: $start_time
- 结束时间: $end_time

输出目录: $output_dir
EOF

    echo "T2I评估完成: $prompt_type 提示词！"
    sleep 3
}

# 主评估执行流程
echo ""
echo "开始Large模型和T2I评估..."
echo "配置总结:"
echo "- GPU配置: $CUDA_VISIBLE_DEVICES"
echo "- 批次大小: $EVAL_BSZ"
echo "- FAR Large: diffloss_d=3 (基于错误分析修正)"
echo "- FAR T2I: diffloss_d=3 (原始T2I架构)"
echo ""

TOTAL_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

# 阶段1: FAR Large模型评估
echo "=========================================="
echo "阶段1: FAR Large模型评估"
echo "使用修正的架构参数 (diffloss_d=3)"
echo "=========================================="

for sampling_steps in "${SAMPLING_STEPS[@]}"; do
    run_large_evaluation "$sampling_steps"
done

# 阶段2: T2I模型评估
echo "=========================================="
echo "阶段2: T2I模型评估"
echo "=========================================="

PROMPT_TYPES=("simple" "medium" "complex")

for sampling_steps in "${SAMPLING_STEPS[@]}"; do
    for prompt_type in "${PROMPT_TYPES[@]}"; do
        run_t2i_evaluation "$sampling_steps" "$prompt_type"
    done
done

# 创建最终总结
TOTAL_END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

cat > "$BASE_OUTPUT_DIR/evaluation_summary.txt" << EOF
FAR Large + T2I 评估总结
======================

评估完成时间:
- 开始: $TOTAL_START_TIME
- 结束: $TOTAL_END_TIME

阶段1 - FAR Large模型:
- 模型规模: Large (442M参数)
- 架构配置: diffloss_d=3 (基于错误分析的修正)
- 采样步数: ${SAMPLING_STEPS[@]}
- 评估类别: 20个精选ImageNet类别

阶段2 - T2I模型:
- 架构配置: diffloss_d=3 (T2I专用)
- 提示词类型: Simple, Medium, Complex
- 采样步数: ${SAMPLING_STEPS[@]}

关键教学点:
1. 不同规模的模型可能使用不同的架构深度
2. Huge模型(diffloss_d=6) vs Large模型(diffloss_d=3)体现了架构优化策略
3. T2I模型使用专门针对文本条件生成的架构设计
4. 错误分析是深度学习调试的重要技能

总运行次数: $((2 + 3 * 2)) = 8次评估
结果保存位置: $BASE_OUTPUT_DIR
EOF

echo ""
echo "=========================================="
echo "Large + T2I 评估全部完成！"
echo "=========================================="
echo ""
echo "阶段1 (Large模型): 2次运行"
echo "阶段2 (T2I模型): 6次运行"
echo "总计: 8次评估运行"
echo ""
echo "关键成果:"
echo "- 成功解决了Large模型的架构不匹配问题"
echo "- 验证了T2I模型的正确配置"
echo "- 展示了不同模型规模的架构差异"
echo ""
echo "结果保存到: $BASE_OUTPUT_DIR"
echo "总结报告: $BASE_OUTPUT_DIR/evaluation_summary.txt"
echo ""
echo "这次评估为你提供了宝贵的深度学习调试经验！🎉"
