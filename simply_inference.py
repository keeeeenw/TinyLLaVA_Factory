from tinyllava.eval.run_tiny_llava import eval_model

# model_path = "/home/ken/workspace/TinyLLaVA_Factory/checkpoints/llava_factory/tiny-llava-MicroLlama-siglip-so400m-patch14-384-base-finetune/"
# siglip2 v1
# model_path = "/home/ken/workspace/TinyLLaVA_Factory/checkpoints/llava_factory/tiny-llava-MicroLlama-siglip2-so400m-patch14-384-base-finetune/"
# siglip2 v2 does not work as well as v1
# model_path = "/home/ken/workspace/TinyLLaVA_Factory/checkpoints/llava_factory/tiny-llava-MicroLlama-siglip2-so400m-patch14-384-base-finetune-v2/"
# conv_mode = "llama" # or llama, gemma, etc
model_path = "/home/ken/workspace/TinyLLaVA_Factory/checkpoints/llava_factory/tiny-llava-Qwen3-0.6B-siglip2-so400m-patch14-384-qwen3-0_5b_base-finetune/"
conv_mode = "qwen3_instruct" # or llama, gemma, etc
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model": None,
    "query": prompt,
    "conv_mode": conv_mode,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)