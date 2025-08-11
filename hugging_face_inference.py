# from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
# import torch
# from PIL import Image
# import requests
# from io import BytesIO

# repo_id = "keeeeenw/MicroLlava-siglip-so400m-patch14-384-base-finetune"   # change to your repo

# user_prompt = "What are the things I should be cautious about when I visit here?"
# image_url = "https://llava-vl.github.io/static/images/view.jpg"

# # Load image from URL
# response = requests.get(image_url, timeout=30)
# response.raise_for_status()
# image = Image.open(BytesIO(response.content)).convert("RGB")

# # Load components
# tokenizer = AutoTokenizer.from_pretrained(repo_id)
# processor = AutoProcessor.from_pretrained(repo_id)  # expects image processor + tokenizer
# model = AutoModelForCausalLM.from_pretrained(
#     repo_id,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     trust_remote_code=True  # set True if your repo includes custom code
# )

# # Prepare inputs for a VLM
# # Many TinyLLaVA style processors accept both "text" and "images"
# inputs = processor(text=user_prompt, images=image, return_tensors="pt").to(model.device)

# # Generate
# with torch.inference_mode():
#     output_ids = model.generate(
#         **inputs,
#         max_new_tokens=128,
#         temperature=0.2,
#         do_sample=False
#     )

# # Decode
# # If your processor bundles a tokenizer, you can use processor.tokenizer instead
# generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# print(generated_text)

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

hf_path = 'keeeeenw/MicroLlava-siglip-so400m-patch14-384-base-finetune'

# Sync local cache with remote, downloading only new/changed files
# hf_path = snapshot_download(
#     repo_id=hf_path,
#     local_dir=None,
#     local_dir_use_symlinks=True, # Saves disk space if supported
#     force_download=True,        # Don't redownload everything
#     resume_download=True         # Continue or fetch only missing files
# )

# Load model from the freshly synced local path
# hf_path = "/home/ken/workspace/TinyLLaVA_Factory/checkpoints/llava_factory/tiny-llava-MicroLlama-siglip-so400m-patch14-384-base-finetune-huggingface"
model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)

# model.cuda()
config = model.config
tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False, model_max_length = config.tokenizer_model_max_length,padding_side = config.tokenizer_padding_side)
prompt="What are the things I should be cautious about when I visit here?"
image_url="https://llava-vl.github.io/static/images/view.jpg"
output_text, genertaion_time = model.chat(prompt=prompt,
                                          image=image_url,
                                          tokenizer=tokenizer)

print('model output:', output_text)
print('runing time:', genertaion_time)
