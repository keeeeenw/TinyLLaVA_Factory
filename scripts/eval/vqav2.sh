#!/bin/bash

# Override chunks for single GPU parallel processing
# Nvidia 4090 with 24G of VRAM can fit 4 qwen3-0.6 base.
# Adjust it accordingly for your use case.
CHUNKS=${CHUNKS_OVERRIDE:-4}
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"

# For single GPU parallel processing, use the same GPU for all chunks
# Setting this to false would fallback to the old script logic for
# one chunk per GPU which is a not good use of the resource for
# small models like qwen3-0.6 base or microllama 300m
SINGLE_GPU_MODE=true
if [ "$SINGLE_GPU_MODE" = "true" ]; then
    # Use single GPU for all processes
    GPULIST=()
    for ((i=0; i<CHUNKS; i++)); do
        GPULIST+=(${gpu_list%%,*})  # Use first GPU from list
    done
else
    # Original multi-GPU logic
    IFS=',' read -ra GPULIST <<< "$gpu_list"
    CHUNKS=${#GPULIST[@]}
fi

echo "GPU list: ${GPULIST}"
echo "CHUNKS: ${CHUNKS}"

# This takes around 2 hours on a single 4090 with 4 chunks for qwen3 0.6 and 1 chunk for microllama 300m
# SPLIT="llava_vqav2_mscoco_test-dev2015"
# This takes around 8 hours on a single 4090 with 4 chunks for qwen3 0.6 and 1 chunk for microllama 300m
SPLIT="llava_vqav2_mscoco_test2015"

# siglip1
# MODEL_PATH="/home/ken/workspace/TinyLLaVA_Factory/checkpoints/llava_factory/tiny-llava-MicroLlama-siglip-so400m-patch14-384-base-finetune"
# MODEL_NAME="MicroLlava-siglip-so400m-patch14-384-base-finetune"

# TODO: change conv-mode here for microllama models

# siglip2 v1 - this is the current release 08/17/2025
# MODEL_PATH="/home/ken/workspace/TinyLLaVA_Factory/checkpoints/llava_factory/tiny-llava-MicroLlama-siglip2-so400m-patch14-384-base-finetune"
# MODEL_NAME="MicroLlava-siglip2-so400m-patch14-384-base-finetune"

# TODO: change conv-mode here for microllama models

# siglip2 v2 not as good as v1 so I uploaded v1
# MODEL_PATH="/home/ken/workspace/TinyLLaVA_Factory/checkpoints/llava_factory/tiny-llava-MicroLlama-siglip2-so400m-patch14-384-base-finetune-v2"
# MODEL_NAME="MicroLlava-siglip2-so400m-patch14-384-base-finetune-v2"

# TODO: change conv-mode here for microllama models

# qwen3 0.6B (not working)
# MODEL_PATH="/home/ken/workspace/TinyLLaVA_Factory/checkpoints/llava_factory/tiny-llava-Qwen3-0.6B-siglip2-so400m-patch14-384-qwen3-0_5b_base-finetune-v1"
# MODEL_NAME="MicroLlava-qwen3-0.6B-siglip2-so400m-patch14-384-base-finetune"
# EVAL_DIR="/home/ken/workspace/TinyLLaVA_Factory/data/eval"
# CONV_MODE="qwen3_instruct"

# qwen3 0.6B
MODEL_PATH="checkpoints/llava_factory/tiny-llava-Qwen3-0.6B-base-siglip2-so400m-patch14-384-qwen3-0_6b_base-finetune"
MODEL_NAME="MicroLlava-qwen3-0.6B-base-siglip2-so400m-patch14-384-base-finetune-working"
CONV_MODE="qwen3_base"
EVAL_DIR="/home/ken/workspace/TinyLLaVA_Factory/data/eval"
# this does not work for qwen3
# --temperature 0 \

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m tinyllava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file $EVAL_DIR/vqav2/$SPLIT.jsonl \
        --image-folder $EVAL_DIR/vqav2/test2015 \
        --answers-file $EVAL_DIR/vqav2/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
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
