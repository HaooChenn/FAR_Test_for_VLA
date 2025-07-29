#!/bin/bash

# FAR Complete Evaluation Script
# This script evaluates both ImageNet class-conditional and Text-to-Image models
# Educational purpose: Compare different conditioning methods and model scales

set -e  # Exit on any error

echo "=========================================="
echo "FAR Complete Evaluation Script"
echo "Class-Conditional + Text-to-Image Models"
echo "=========================================="

# Configuration
export CUDA_VISIBLE_DEVICES=1  # Use GPU card 1
EVAL_BSZ=8  # Conservative batch size for single GPU
IMG_SIZE=256
CFG=3.0
TEMPERATURE=1.0
NUM_ITER=10
SPEED_TEST_STEPS=5

# Base paths - organized for clarity
BASE_OUTPUT_DIR="./complete_evaluation_results"
PROMPTS_DIR="./prompts"
VAE_PATH="pretrained/vae/kl16.ckpt"  # Note: Different path than T2I
TEXT_MODEL_PATH="pretrained/Qwen2-VL-1.5B-Instruct"

# Model configurations for ImageNet class-conditional generation
declare -A IMAGENET_MODELS
IMAGENET_MODELS["far_base"]="pretrained_models/far/far_base"
IMAGENET_MODELS["far_large"]="pretrained_models/far/far_large" 
IMAGENET_MODELS["far_huge"]="pretrained_models/far/far_huge"

# Model configuration for Text-to-Image generation
T2I_MODEL_PATH="pretrained_models/far/far_t2i"

# Sampling steps to test
SAMPLING_STEPS=(50 100)

# Create directory structure and files
echo "Setting up directory structure and class mapping..."
mkdir -p $BASE_OUTPUT_DIR
mkdir -p $PROMPTS_DIR

# Create ImageNet class mapping file
cat > $PROMPTS_DIR/imagenet_classes.txt << 'EOF'
# ImageNet class IDs for evaluation
# Format: class_name:class_id
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

# Create T2I prompt files (same as before)
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

echo "Files created successfully!"
echo "- ImageNet classes: $PROMPTS_DIR/imagenet_classes.txt (20 classes)"
echo "- Simple prompts: $PROMPTS_DIR/simple_prompts.txt (20 prompts)"
echo "- Medium prompts: $PROMPTS_DIR/medium_prompts.txt (20 prompts)"  
echo "- Complex prompts: $PROMPTS_DIR/complex_prompts.txt (20 prompts)"

# Function to run ImageNet class-conditional evaluation
run_imagenet_evaluation() {
    local model_name=$1
    local model_path=$2
    local sampling_steps=$3
    
    echo ""
    echo "=========================================="
    echo "Running ImageNet evaluation: $model_name"
    echo "Sampling steps: $sampling_steps"
    echo "=========================================="
    
    # Create output directory for this specific run
    local output_dir="$BASE_OUTPUT_DIR/imagenet_${model_name}_steps${sampling_steps}"
    mkdir -p $output_dir
    
    # Copy the class mapping file for reference
    cp "$PROMPTS_DIR/imagenet_classes.txt" "$output_dir/used_classes.txt"
    
    # Record start time
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Start time: $start_time"
    
    # Run the ImageNet evaluation using our custom evaluation script
    # This script is specifically designed for evaluating selected ImageNet classes
    python imagenet_class_evaluation.py \
        --img_size $IMG_SIZE \
        --vae_path $VAE_PATH \
        --vae_embed_dim 16 \
        --vae_stride 16 \
        --patch_size 1 \
        --model $model_name \
        --diffloss_d 3 \
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
    
    # Record end time
    local end_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "End time: $end_time"
    
    # Create run summary
    cat > "$output_dir/run_summary.txt" << EOF
ImageNet Class-Conditional Generation Summary
===========================================

Model: $model_name
Model Path: $model_path
Sampling Steps: $sampling_steps
Task Type: ImageNet Class-Conditional

Parameters:
- Image Size: $IMG_SIZE
- Batch Size: $EVAL_BSZ
- Num Iterations: $NUM_ITER
- CFG: $CFG
- Temperature: $TEMPERATURE
- Classes Evaluated: 20 specific ImageNet classes

Target Classes:
- golden_retriever (207), tabby_cat (281), red_fox (277)
- monarch_butterfly (323), daisy (985), rose (973)
- lighthouse (437), castle (483), cottage (500)
- sports_car (817), steam_locomotive (820), sailboat (554)
- aircraft_carrier (403), mountain_bike (671), pizza (963)
- strawberry (949), coffee_mug (504), violin (889)
- backpack (414), umbrella (879)

Timing:
- Start Time: $start_time
- End Time: $end_time

Output Directory: $output_dir
Used Classes: See used_classes.txt
Generated Images: See generated_images/ subdirectory
Speed Results: See speed_results/ subdirectory
EOF

    echo "ImageNet evaluation completed for $model_name!"
    sleep 3
}

# Function to run T2I evaluation
run_t2i_evaluation() {
    local sampling_steps=$1
    local prompt_type=$2
    
    echo ""
    echo "=========================================="
    echo "Running T2I evaluation"
    echo "Sampling steps: $sampling_steps"
    echo "Prompt type: $prompt_type"
    echo "=========================================="
    
    # Create output directory for this specific run
    local output_dir="$BASE_OUTPUT_DIR/t2i_steps${sampling_steps}_${prompt_type}"
    mkdir -p $output_dir
    
    # Copy the prompt file for reference
    cp "$PROMPTS_DIR/${prompt_type}_prompts.txt" "$output_dir/used_prompts.txt"
    
    # Record start time
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Start time: $start_time"
    
    # Run the T2I evaluation using main_far_t2i.py
    # Note: T2I models have different architecture requirements
    torchrun --nnodes=1 --nproc_per_node=1 main_far_t2i.py \
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
    
    # Record end time
    local end_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "End time: $end_time"
    
    # Create run summary
    cat > "$output_dir/run_summary.txt" << EOF
Text-to-Image Generation Summary
===============================

Model: FAR T2I
Model Path: $T2I_MODEL_PATH
Sampling Steps: $sampling_steps
Prompt Type: $prompt_type
Task Type: Text-to-Image

Parameters:
- Image Size: $IMG_SIZE
- Batch Size: $EVAL_BSZ
- Num Iterations: $NUM_ITER
- CFG: $CFG
- Temperature: $TEMPERATURE
- Text Model: Qwen2-VL-1.5B-Instruct

Timing:
- Start Time: $start_time
- End Time: $end_time

Output Directory: $output_dir
Used Prompts: See used_prompts.txt
EOF

    echo "T2I evaluation completed for $prompt_type prompts!"
    sleep 3
}

# Main evaluation execution
echo ""
echo "Starting complete FAR evaluation..."
echo "Configuration:"
echo "- GPU: $CUDA_VISIBLE_DEVICES"
echo "- Batch Size: $EVAL_BSZ"
echo "- Image Size: $IMG_SIZE"
echo "- CFG: $CFG"
echo "- Temperature: $TEMPERATURE"
echo "- Num Iterations: $NUM_ITER"
echo "- Speed Test Steps: $SPEED_TEST_STEPS"
echo ""

# Track total start time
TOTAL_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

# Phase 1: ImageNet Class-Conditional Evaluation
echo "=========================================="
echo "PHASE 1: ImageNet Class-Conditional Models"
echo "=========================================="

for model_name in "${!IMAGENET_MODELS[@]}"; do
    model_path="${IMAGENET_MODELS[$model_name]}"
    
    for sampling_steps in "${SAMPLING_STEPS[@]}"; do
        run_imagenet_evaluation "$model_name" "$model_path" "$sampling_steps"
    done
done

# Phase 2: Text-to-Image Evaluation
echo "=========================================="
echo "PHASE 2: Text-to-Image Model"
echo "=========================================="

PROMPT_TYPES=("simple" "medium" "complex")

for sampling_steps in "${SAMPLING_STEPS[@]}"; do
    for prompt_type in "${PROMPT_TYPES[@]}"; do
        run_t2i_evaluation "$sampling_steps" "$prompt_type"
    done
done

# Create comprehensive summary
TOTAL_END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

cat > "$BASE_OUTPUT_DIR/comprehensive_summary.txt" << EOF
FAR Complete Evaluation Summary
==============================

Total Evaluation Period:
- Start Time: $TOTAL_START_TIME
- End Time: $TOTAL_END_TIME

Phase 1: ImageNet Class-Conditional Models
- Models Tested: FAR Base, Large, Huge
- Classes: 20 carefully selected ImageNet classes
- Sampling Steps: ${SAMPLING_STEPS[@]}
- Purpose: Evaluate class-conditional generation quality

Phase 2: Text-to-Image Model  
- Model: FAR T2I
- Prompt Types: Simple, Medium, Complex (20 prompts each)
- Sampling Steps: ${SAMPLING_STEPS[@]}
- Purpose: Evaluate text-conditional generation quality

Configuration:
- GPU Used: $CUDA_VISIBLE_DEVICES
- Batch Size: $EVAL_BSZ
- Image Size: $IMG_SIZE
- CFG: $CFG
- Temperature: $TEMPERATURE
- Num Iterations: $NUM_ITER

Research Value:
This evaluation allows comparison between:
1. Different model scales (Base vs Large vs Huge)
2. Different conditioning methods (Class vs Text)
3. Different prompt complexities (Simple vs Medium vs Complex)
4. Different sampling speeds (50 vs 100 steps)

Total Runs: $((${#IMAGENET_MODELS[@]} * ${#SAMPLING_STEPS[@]} + ${#PROMPT_TYPES[@]} * ${#SAMPLING_STEPS[@]}))

Analysis Recommendations:
1. Compare generation quality across model scales
2. Analyze conditioning effectiveness (class vs text)
3. Evaluate prompt complexity handling
4. Assess speed vs quality trade-offs
5. Study architectural differences impact
EOF

echo ""
echo "=========================================="
echo "COMPLETE EVALUATION FINISHED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Phase 1 (ImageNet): $((${#IMAGENET_MODELS[@]} * ${#SAMPLING_STEPS[@]})) runs"
echo "Phase 2 (T2I): $((${#PROMPT_TYPES[@]} * ${#SAMPLING_STEPS[@]})) runs" 
echo "Total runs: $((${#IMAGENET_MODELS[@]} * ${#SAMPLING_STEPS[@]} + ${#PROMPT_TYPES[@]} * ${#SAMPLING_STEPS[@]}))"
echo ""
echo "Results saved to: $BASE_OUTPUT_DIR"
echo "Comprehensive summary: $BASE_OUTPUT_DIR/comprehensive_summary.txt"
echo ""
echo "This evaluation provides valuable insights into:"
echo "- Model scaling effects on generation quality"
echo "- Conditioning method differences (class vs text)"
echo "- Architectural impact on different tasks"
echo "- Speed vs quality trade-offs"
echo ""
echo "Happy analyzing! 🎉"