#!/bin/bash

# FAR T2I Comprehensive Evaluation Script
# This script runs evaluation for all three model sizes with different sampling steps

set -e  # Exit on any error

echo "=========================================="
echo "FAR T2I Comprehensive Evaluation Script"
echo "=========================================="

# Configuration
export CUDA_VISIBLE_DEVICES=1  # Use GPU card 1
EVAL_BSZ=8  # Conservative batch size for single A100 with T2I
IMG_SIZE=256
CFG=3.0
TEMPERATURE=1.0
NUM_ITER=10
SPEED_TEST_STEPS=10

# Base paths
BASE_OUTPUT_DIR="./t2i_evaluation_results"
PROMPTS_DIR="./prompts"
VAE_PATH="pretrained/vae_mar/kl16.ckpt"
TEXT_MODEL_PATH="pretrained/Qwen2-VL-1.5B-Instruct"

# Model configurations
declare -A MODELS
MODELS["far_base"]="pretrained_models/far/far_base"
MODELS["far_large"]="pretrained_models/far/far_large" 
MODELS["far_huge"]="pretrained_models/far/far_huge"

# Sampling steps to test
SAMPLING_STEPS=(50 100)

# Create directory structure and prompt files
echo "Setting up directory structure and prompt files..."

mkdir -p $BASE_OUTPUT_DIR
mkdir -p $PROMPTS_DIR

# Create simple prompts file
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

# Create medium prompts file
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

# Create complex prompts file
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

echo "Prompt files created successfully!"
echo "- Simple prompts: $PROMPTS_DIR/simple_prompts.txt (20 prompts)"
echo "- Medium prompts: $PROMPTS_DIR/medium_prompts.txt (20 prompts)"
echo "- Complex prompts: $PROMPTS_DIR/complex_prompts.txt (20 prompts)"

# Function to run evaluation for a specific model and sampling steps
run_evaluation() {
    local model_name=$1
    local model_path=$2
    local sampling_steps=$3
    local prompt_type=$4
    
    echo ""
    echo "=========================================="
    echo "Running $model_name with $sampling_steps sampling steps"
    echo "Prompt type: $prompt_type"
    echo "=========================================="
    
    # Create output directory for this specific run
    local output_dir="$BASE_OUTPUT_DIR/${model_name}_steps${sampling_steps}_${prompt_type}"
    mkdir -p $output_dir
    
    # Copy the prompt file to the output directory for reference
    cp "$PROMPTS_DIR/${prompt_type}_prompts.txt" "$output_dir/used_prompts.txt"
    
    # Record start time
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Start time: $start_time"
    
    # Run the evaluation
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
        --resume $model_path \
        --text_model_path $TEXT_MODEL_PATH \
        --data_path $PROMPTS_DIR \
        --evaluate
    
    # Record end time
    local end_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "End time: $end_time"
    
    # Create a run summary
    cat > "$output_dir/run_summary.txt" << EOF
FAR T2I Evaluation Run Summary
==============================

Model: $model_name
Model Path: $model_path
Sampling Steps: $sampling_steps
Prompt Type: $prompt_type

Parameters:
- Image Size: $IMG_SIZE
- Batch Size: $EVAL_BSZ
- Num Iterations: $NUM_ITER
- CFG: $CFG
- Temperature: $TEMPERATURE
- Speed Test Steps: $SPEED_TEST_STEPS

Timing:
- Start Time: $start_time
- End Time: $end_time

Output Directory: $output_dir
Used Prompts: $PROMPTS_DIR/${prompt_type}_prompts.txt
EOF

    echo "Evaluation completed for $model_name with $sampling_steps steps!"
    echo "Results saved to: $output_dir"
    
    # Small delay between runs to ensure clean separation
    sleep 5
}

# Main evaluation loop
echo ""
echo "Starting comprehensive evaluation..."
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

# Test with different prompt complexities
PROMPT_TYPES=("simple" "medium" "complex")

# Run evaluation for each model, sampling steps, and prompt type
for model_name in "${!MODELS[@]}"; do
    model_path="${MODELS[$model_name]}"
    
    for sampling_steps in "${SAMPLING_STEPS[@]}"; do
        for prompt_type in "${PROMPT_TYPES[@]}"; do
            run_evaluation "$model_name" "$model_path" "$sampling_steps" "$prompt_type"
        done
    done
done

# Create overall summary
TOTAL_END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

cat > "$BASE_OUTPUT_DIR/overall_summary.txt" << EOF
FAR T2I Comprehensive Evaluation Summary
=======================================

Total Evaluation Period:
- Start Time: $TOTAL_START_TIME
- End Time: $TOTAL_END_TIME

Models Tested:
- FAR Base: ${MODELS["far_base"]}
- FAR Large: ${MODELS["far_large"]}
- FAR Huge: ${MODELS["far_huge"]}

Sampling Steps Tested: ${SAMPLING_STEPS[@]}
Prompt Types Tested: ${PROMPT_TYPES[@]}

Configuration:
- GPU Used: $CUDA_VISIBLE_DEVICES
- Batch Size: $EVAL_BSZ
- Image Size: $IMG_SIZE
- CFG: $CFG
- Temperature: $TEMPERATURE
- Num Iterations: $NUM_ITER

Total Runs: $((${#MODELS[@]} * ${#SAMPLING_STEPS[@]} * ${#PROMPT_TYPES[@]}))

Results Directory Structure:
$BASE_OUTPUT_DIR/
├── overall_summary.txt (this file)
├── far_base_steps50_simple/
├── far_base_steps50_medium/
├── far_base_steps50_complex/
├── far_base_steps100_simple/
├── far_base_steps100_medium/
├── far_base_steps100_complex/
├── far_large_steps50_simple/
├── far_large_steps50_medium/
├── far_large_steps50_complex/
├── far_large_steps100_simple/
├── far_large_steps100_medium/
├── far_large_steps100_complex/
├── far_huge_steps50_simple/
├── far_huge_steps50_medium/
├── far_huge_steps50_complex/
├── far_huge_steps100_simple/
├── far_huge_steps100_medium/
└── far_huge_steps100_complex/

Each subdirectory contains:
- Generated images
- Speed test results (JSON and CSV)
- Used prompts file
- Run summary

Analysis Recommendations:
1. Compare speed results between 50 vs 100 sampling steps
2. Evaluate image quality across different model sizes
3. Assess performance on different prompt complexities
4. Review generated images for VLA task suitability
EOF

echo ""
echo "=========================================="
echo "ALL EVALUATIONS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Total runs completed: $((${#MODELS[@]} * ${#SAMPLING_STEPS[@]} * ${#PROMPT_TYPES[@]}))"
echo "Results saved to: $BASE_OUTPUT_DIR"
echo "Overall summary: $BASE_OUTPUT_DIR/overall_summary.txt"
echo ""
echo "To analyze results:"
echo "1. Check speed comparisons in speed_results/ subdirectories"
echo "2. Review generated images for quality assessment"
echo "3. Compare performance across model sizes and sampling steps"
echo ""
echo "Happy analyzing! 🎉"