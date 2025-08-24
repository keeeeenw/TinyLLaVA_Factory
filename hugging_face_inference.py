from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model from hugging face directly (once they are uploaded)
# hf_path = 'keeeeenw/MicroLlava'
# hf_path = 'keeeeenw/MicroLlava-Qwen3-0.6B-base-siglip2-so400m'

# Load model from the freshly synced local path after training for the first time
# because they don't exist on hugging face yet.
# for 'keeeeenw/MicroLlava'
# hf_path = "/home/ken/workspace/TinyLLaVA_Factory/checkpoints/llava_factory/tiny-llava-MicroLlama-siglip2-so400m-patch14-384-base-finetune-huggingface"
# for 'keeeeenw/MicroLlava-Qwen3-0.6B-base-siglip2-so400m'
hf_path = "/home/ken/workspace/TinyLLaVA_Factory/checkpoints/llava_factory/tiny-llava-Qwen3-0.6B-base-siglip2-so400m-patch14-384-qwen3-0_6b_base-finetune-huggingface"
model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)

# model.cuda() # if no cuda is used, it takes ~78 seconds inference on my consumer CPU. 
config = model.config
tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False, model_max_length = config.tokenizer_model_max_length,padding_side = config.tokenizer_padding_side)
prompt="What are the things I should be cautious about when I visit here?"
image_url="https://llava-vl.github.io/static/images/view.jpg"
output_text, genertaion_time = model.chat(prompt=prompt,
                                          image=image_url,
                                          tokenizer=tokenizer)

print('model output:', output_text)
print('runing time:', genertaion_time)
