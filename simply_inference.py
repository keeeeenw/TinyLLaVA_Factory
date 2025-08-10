from tinyllava.eval.run_tiny_llava import eval_model

model_path = "/home/ken/workspace/TinyLLaVA_Factory/checkpoints/llava_factory/tiny-llava-MicroLlama-siglip-so400m-patch14-384-base-finetune/"
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"
conv_mode = "llama" # or llama, gemma, etc

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