#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# This takes around 2 hours on a single 4090
SPLIT="llava_vqav2_mscoco_test-dev2015"
# This takes around 8 hours on a single 4090
# SPLIT="llava_vqav2_mscoco_test2015"

# siglip1
# MODEL_PATH="/home/ken/workspace/TinyLLaVA_Factory/checkpoints/llava_factory/tiny-llava-MicroLlama-siglip-so400m-patch14-384-base-finetune"
# MODEL_NAME="MicroLlava-siglip-so400m-patch14-384-base-finetune"

# TODO: change conv-mode here for microllama models

# siglip2 v1 - this is the current release 08/17/2025
MODEL_PATH="/home/ken/workspace/TinyLLaVA_Factory/checkpoints/llava_factory/tiny-llava-MicroLlama-siglip2-so400m-patch14-384-base-finetune"
MODEL_NAME="MicroLlava-siglip2-so400m-patch14-384-base-finetune"
CONV_MODE="llama"

# siglip2 v2 not as good as v1 so I uploaded v1
# MODEL_PATH="/home/ken/workspace/TinyLLaVA_Factory/checkpoints/llava_factory/tiny-llava-MicroLlama-siglip2-so400m-patch14-384-base-finetune-v2"
# MODEL_NAME="MicroLlava-siglip2-so400m-patch14-384-base-finetune-v2"

# TODO: change conv-mode here for microllama models

# qwen3 0.6B
# MODEL_PATH="/home/ken/workspace/TinyLLaVA_Factory/checkpoints/llava_factory/tiny-llava-Qwen3-0.6B-siglip2-so400m-patch14-384-qwen3-0_5b_base-finetune"
# MODEL_NAME="MicroLlava-qwen3-0.6B-siglip2-so400m-patch14-384-base-finetune"
# CONV_MODE="qwen3_instruct"

EVAL_DIR="/home/ken/workspace/TinyLLaVA_Factory/data/eval"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m tinyllava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file $EVAL_DIR/vqav2/$SPLIT.jsonl \
        --image-folder $EVAL_DIR/vqav2/test2015 \
        --answers-file $EVAL_DIR/vqav2/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $CONV_MODE &
done

wait

output_file=$EVAL_DIR/vqav2/answers/$SPLIT/$MODEL_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $EVAL_DIR/vqav2/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $MODEL_NAME --dir $EVAL_DIR/vqav2
