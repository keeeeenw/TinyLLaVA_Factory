from huggingface_hub import create_repo, upload_folder
repo_id = "keeeeenw/MicroLlava-siglip-so400m-patch14-384-base-finetune"
create_repo(repo_id, private=True, exist_ok=True)
upload_folder(folder_path="checkpoints/llava_factory/tiny-llava-MicroLlama-siglip-so400m-patch14-384-base-finetune-huggingface", repo_id=repo_id, repo_type="model")