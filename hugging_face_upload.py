from huggingface_hub import create_repo, upload_folder
repo_id = "keeeeenw/MicroLlava-Qwen3-0.6B-base-siglip2-so400m"
create_repo(repo_id, private=True, exist_ok=True)
upload_folder(folder_path="checkpoints/llava_factory/tiny-llava-Qwen3-0.6B-base-siglip2-so400m-patch14-384-qwen3-0_6b_base-finetune-huggingface", repo_id=repo_id, repo_type="model")