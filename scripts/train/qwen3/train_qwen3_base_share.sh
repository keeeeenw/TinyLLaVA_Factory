DATA_ROOT=/home/ken/workspace/TinyLLaVA_Factory/data
# FINETUNE_DATA_PATH=$DATA_ROOT/text_files/llava_v1_5_mix665k_cleaned_data.json #finetune annotation file path - no ocr_vqa - run remove_dataset.py on llava_v1_5_mix665k.json
FINETUNE_DATA_PATH=$DATA_ROOT/text_files/llava_v1_5_mix665k_cleaned_data_w_ocr_vqa.json #finetune annotation file path - w ocr_vqa but entries removed due to unable to download image ~ 356 images missing
IMAGE_PATH=$DATA_ROOT/llava/llava_pretrain/images #pretrain image dir
DATA_PATH=$DATA_ROOT/text_files/blip_laion_cc_sbu_558k.json #pretrain annotation file path
FINETUNE_IMAGE_PATH=$DATA_ROOT/ #finetune image dir

LLM_VERSION=Qwen/Qwen3-0.6B-base # llm path in huggingface
VT_VERSION=google/siglip2-so400m-patch14-384 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
# TODO: double check if we need to change converstaion here
# Qwen3 uses ChatML-like formatting with <|im_start|>, <|im_end|>, and does not use USER: or ASSISTANT:
CONV_VERSION=qwen3_base #chat template, other options are: phi, llama, gemmma, etc
VERSION=qwen3-0_6b_base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
# TODO: we can try to increase max length because qwen3 supports a lot more than this.
MODEL_MAX_LENGTH=2048 #max model length for llm

# Still required
# bash scripts/train/qwen3/pretrain_qwen3.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"

# No need to run this.
# bash scripts/train/qwen3/finetune_qwen3.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"

# Removed 356 entries from OCR dataset that cannot be downloaded. 
SHARE_FINETUNE_DATA_PATH=$DATA_ROOT/text_files/cleaned_sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k_ocr_reduced.json
SHARE_FINETUNE_IMAGE_PATH=$DATA_ROOT/ #finetune image dir
# For train_phi_share.sh, they first run pretrain_share.sh, we will pick up from pretrain.sh directly. 
# because we don't have the full SAM dataset
# this will pickup from pretrain_qwen3.sh
bash scripts/train/qwen3/finetune_qwen3_base_share.sh "$SHARE_FINETUNE_DATA_PATH" "$SHARE_FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"